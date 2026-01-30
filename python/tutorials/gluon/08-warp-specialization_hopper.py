"""
Warp Specialization
===================

This tutorial covers warp specialization. In typical GPU kernels, all the warps
in the kernel are performing parallel slices of the same task. Warp
specialization, however, is a technique where different warps in the kernel are
doing completely different tasks.

With warp specialization, we can overlap execution of independent parts of the
kernel by placing the work in different warps. This minimizes the critical path
in each warp, and we rely on the warp scheduler to dynamically schedule the
warps. We can also overlap non-async operations that exercise different parts of
the hardware without relying on precise SASS-level instruction interleaving.

However, warp specialization comes at the cost of additional synchronization
overhead, potentially higher shared memory usage for communicating data, and
higher overall register pressure.

Warp specialization in Gluon is only supported on Hopper and newer GPUs.
"""

import pytest
import torch
import triton
import importlib
from functools import partial
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from triton.language.core import _aggregate as aggregate
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import tma, mbarrier, fence_async_shared
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    tensor_memory_descriptor,
    allocate_tensor_memory,
    get_tmem_reg_layout,
    tcgen05_mma,
    tcgen05_commit,
)

if torch.cuda.is_available():
    from triton._C.libtriton import nvidia
    cublas_workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    cublas = nvidia.cublas.CublasLt(cublas_workspace)
else:
    cublas = None

# Re-use utilities from the previous tutorial.
t3 = importlib.import_module("03-async-copy")
t4 = importlib.import_module("04-tma")
t7 = importlib.import_module("07-persistence")


def is_hopper_or_newer():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] >= 9


def is_blackwell():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] == 10


if __name__ == "__main__" and not is_hopper_or_newer():
    raise RuntimeError("This tutorial requires Hopper or newer NVIDIA GPU")

# Let's revisit our elementwise add kernel and implement a warp-specialized
# version. In a warp-specialized kernel, groups of warps that perform a specific
# task are called "partitions", and each can have a different number of warps
# and registers.
#
# First, we need to decide what the partitions will be and how many registers
# they will get. One of the benefits of warp specialization is that partitions
# that only use scalar values require only 1 warp and often very few registers.
# For example, we can have one partition that just issues async TMA loads and
# one partition that just issues TMA stores, each with 1 warp and 24 registers,
# the minimum number of registers we can assign to a warp.
#
# Then we have one compute partition, with either 4 or 8 warps, which performs
# the vector addition. Estimating the right register allocation is difficult,
# and often involves trial and error, profiling, and autotuning. We will need to
# use mbarriers to signal between the partitions using producer-consumer pairs.
#
# To write a warp-specialized kernel, we need to write a separate function for
# each partition. One of the partitions must be chosen as the "default"
# partition and it always has the same number of warps as `num_warps` passed to
# the kernel. The other partitions, i.e. the "worker" partitions, can have
# different numbers of warps. The signature of the worker partition functions
# must all be the same. Only the default partition can accept tensor arguments.
#
# To quickly sketch out the partitions: load partition will fetch inputs to smem
# and signal the compute partition. The compute partition will consume the
# operands and send them to the store partition over smem.
#
# Recall that we need fence_async_shared to synchronize the async and generic
# proxies. This also applies if the buffer accesses are initiated in different
# partitions, even when they are sequenced by mbarrier.arrive:
#
# ```python
# smem.store(value)  # in partition A
# fence_async_shared()
# mbarrier.arrive(bar, count=1)
#
# mbarrier.wait(bar, phase=0)  # in partition B
# tma.async_copy_shared_to_global(desc, [0, 0], smem)
# ```
#
# A fence is needed somewhere between the shared memory store and the TMA store.
#
# ```python
# value = smem.load()
# mbarrier.arrive(bar, count=1)
#
# mbarrier.wait(bar, phase=0)
# fence_async_shared()
# tma.async_copy_global_to_shared(desc, [0, 0], bar, smem)
# ```
#
# A fence is needed somewhere between the shared memory load and the TMA load.


@gluon.jit
def load_partition(descs, barriers, buffers, xoff, numel, YBLOCK: gl.constexpr):
    # Unpack the arguments.
    a_desc, b_desc, c_desc = descs
    load_empty_bars, load_ready_bars, c_empty_bars, c_ready_bars = barriers
    a_bufs, b_bufs, c_bufs = buffers
    xnumel, ynumel = numel

    num_buffers: gl.constexpr = a_bufs.type.shape[0]

    # All the partitions need to have the same number of inner loop iterations.
    for i in range(gl.cdiv(ynumel, YBLOCK)):
        index = i % num_buffers
        phase = i // num_buffers & 1
        a_buf = a_bufs.index(index)
        b_buf = b_bufs.index(index)
        load_empty_bar = load_empty_bars.index(index)
        load_ready_bar = load_ready_bars.index(index)

        # Wait for the current buffers to be empty. Recall that mbarriers are
        # initialized to phase 1 complete, so we wait starting with phase 1 to
        # allow the producer to begin filling the pipeline.
        mbarrier.wait(load_empty_bar, phase ^ 1)

        # Okay, a_buf and b_buf are empty. Issue the TMA loads, and have them
        # signal the operand buffers as ready when they complete.
        yoff = i * YBLOCK
        mbarrier.expect(load_ready_bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(a_desc, [xoff, yoff], load_ready_bar, a_buf)
        tma.async_copy_global_to_shared(b_desc, [xoff, yoff], load_ready_bar, b_buf)


@gluon.jit
def store_partition(descs, barriers, buffers, xoff, numel, YBLOCK: gl.constexpr):
    a_desc, b_desc, c_desc = descs
    load_empty_bars, load_ready_bars, c_empty_bars, c_ready_bars = barriers
    a_bufs, b_bufs, c_bufs = buffers
    xnumel, ynumel = numel

    # This partition consumes the addition result, passed over smem, and stores
    # them to global memory.
    num_buffers: gl.constexpr = c_bufs.type.shape[0]
    # We will keep `num_buffers-1` stores in flight by software pipelining.
    outstanding_stores: gl.constexpr = num_buffers - 1

    for i in range(gl.cdiv(ynumel, YBLOCK)):
        index = i % num_buffers
        phase = i // num_buffers & 1
        c_buf = c_bufs.index(index)
        c_ready_bar = c_ready_bars.index(index)

        # Wait for the compute partition to produce c.
        mbarrier.wait(c_ready_bar, phase)
        yoff = i * YBLOCK
        tma.async_copy_shared_to_global(c_desc, [xoff, yoff], c_buf)

        tma.store_wait(outstanding_stores)
        c_empty_bar = c_empty_bars.index((i - outstanding_stores) % num_buffers)
        # Signal the compute partition that the buffer `outstanding_stores`
        # iterations ago is consumed, predicated on there having been at least
        # that many outstanding stores.
        mbarrier.arrive(c_empty_bar, count=1, pred=i >= outstanding_stores)

    # Since we waited for the last value of c, all the other partitions have
    # exited by now. We just need to wait the stores to complete.
    tma.store_wait(0)


# The default partition can have a different signature than the worker partition
# functions.
@gluon.jit
def compute_partition(barriers, buffers, ynumel, YBLOCK: gl.constexpr, layout: gl.constexpr):
    load_empty_bars, load_ready_bars, c_empty_bars, c_ready_bars = barriers
    a_bufs, b_bufs, c_bufs = buffers

    num_load_buffers: gl.constexpr = a_bufs.type.shape[0]
    num_store_buffers: gl.constexpr = c_bufs.type.shape[0]

    for i in range(gl.cdiv(ynumel, YBLOCK)):
        load_index = i % num_load_buffers
        load_phase = i // num_load_buffers & 1
        a_buf = a_bufs.index(load_index)
        b_buf = b_bufs.index(load_index)
        load_ready_bar = load_ready_bars.index(load_index)
        load_empty_bar = load_empty_bars.index(load_index)

        # Wait for the operands then consume them.
        mbarrier.wait(load_ready_bar, load_phase)
        a_val = a_buf.load(layout)
        b_val = b_buf.load(layout)
        # Fence before signalling the load partitions so the TMA load is
        # ordered with the shared load.
        fence_async_shared()
        mbarrier.arrive(load_empty_bar, count=1)

        c_val = a_val + b_val

        store_idx = i % num_store_buffers
        store_phase = i // num_store_buffers & 1
        c_buf = c_bufs.index(store_idx)
        c_empty_bar = c_empty_bars.index(store_idx)
        c_ready_bar = c_ready_bars.index(store_idx)

        mbarrier.wait(c_empty_bar, store_phase ^ 1)
        c_buf.store(c_val)
        # Fence to order with TMA store.
        fence_async_shared()
        mbarrier.arrive(c_ready_bar, count=1)


@gluon.jit
def gemm_warp_specialized_kernel(  #
        a_desc, b_desc, c_desc,  #
        M, N, K, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,  #
        num_load_buffers: gl.constexpr, num_store_buffers: gl.constexpr, num_warps: gl.constexpr):
    # Pick a layout that makes it easy to avoid bank conflicts.
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, num_warps], [1, 0])

    # Allocate all the buffers and barriers.
    a_bufs = gl.allocate_shared_memory(a_desc.dtype, [num_load_buffers] + a_desc.block_type.shape, a_desc.layout)
    b_bufs = gl.allocate_shared_memory(b_desc.dtype, [num_load_buffers] + b_desc.block_type.shape, b_desc.layout)
    c_bufs = gl.allocate_shared_memory(c_desc.dtype, [num_store_buffers] + c_desc.block_type.shape, c_desc.layout)
    load_empty_bars = gl.allocate_shared_memory(gl.int64, [num_load_buffers, 1], mbarrier.MBarrierLayout())
    load_ready_bars = gl.allocate_shared_memory(gl.int64, [num_load_buffers, 1], mbarrier.MBarrierLayout())
    c_empty_bars = gl.allocate_shared_memory(gl.int64, [num_store_buffers, 1], mbarrier.MBarrierLayout())
    c_ready_bars = gl.allocate_shared_memory(gl.int64, [num_store_buffers, 1], mbarrier.MBarrierLayout())

    for i in gl.static_range(num_load_buffers):
        mbarrier.init(load_empty_bars.index(i), count=1)
        mbarrier.init(load_ready_bars.index(i), count=1)
    for i in gl.static_range(num_store_buffers):
        mbarrier.init(c_empty_bars.index(i), count=1)
        mbarrier.init(c_ready_bars.index(i), count=1)

    descs = (a_desc, b_desc, c_desc)
    barriers = (load_empty_bars, load_ready_bars, c_empty_bars, c_ready_bars)
    buffers = (a_bufs, b_bufs, c_bufs)
    # numel = (xnumel, ynumel)

    # pid = gl.program_id(0)
    # xoff = pid * XBLOCK

    # `gl.warp_specialize` declares a warp-specialized section of the kernel.
    # It accepts arguments for the default partition function, which can include
    # tensors, and the default partition function. It takes arguments for all
    # the worker partitions, which cannot include tensors, and takes a list of
    # worker partition functions. The warps and register budget for each
    # partition are passed as lists.
    #
    # Note that warp and register allocation on NVIDIA GPUs is by warpgroup,
    # which are 4 consecutive warps. The number of warps used by a kernel is
    # rounded to the nearest multiple of 4. The compiler tries to organize the
    # warps to reduce the amount of registers allocated. The default partition
    # receives whatever registers are left over, based on `maxnreg` passed to
    # the kernel.
    # gl.warp_specialize([
    #     (compute_partition, (barriers, buffers, ynumel, YBLOCK, layout)),
    #     (load_partition, (descs, barriers, buffers, xoff, numel, YBLOCK)),
    #     (store_partition, (descs, barriers, buffers, xoff, numel, YBLOCK)),
    # ], [1, 1], [24, 24])


def gemm_warp_specialized(a, b, c, BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, #
                                     num_load_buffers=2, num_store_buffers=2, num_warps=4):
    M, K = a.shape
    K, N = b.shape
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    block_shape_a = [BLOCK_M, BLOCK_K]
    block_shape_b = [BLOCK_K, BLOCK_N]
    block_shape_c = [BLOCK_M, BLOCK_N]
    layout_a = gl.NVMMASharedLayout.get_default_for(block_shape_a, gl.float16)
    layout_b = gl.NVMMASharedLayout.get_default_for(block_shape_b, gl.float16)
    layout_c = gl.NVMMASharedLayout.get_default_for(block_shape_c, gl.float16)
    a_desc = TensorDescriptor.from_tensor(a, block_shape_a, layout_a)
    b_desc = TensorDescriptor.from_tensor(b, block_shape_b, layout_b)
    c_desc = TensorDescriptor.from_tensor(c, block_shape_c, layout_c)

    # By default, a warp-specialized kernel assumes maxnreg=256, the maximum
    # allowed per thread, in order to determine how to reallocate registers.
    # We need to intentionally set the register limit. Since the kernel will
    # have `num_warps+4` warps total, register usage will be
    #
    #     maxnreg * (num_warps+4) * 32
    #
    # Keep this in mind when deciding how much occupancy you want.
    gemm_warp_specialized_kernel[grid](  #
        a_desc, b_desc, c_desc, M, N, K,  #
        BLOCK_M, BLOCK_N, BLOCK_K, num_load_buffers, num_store_buffers,  #
        num_warps=num_warps, maxnreg=128)


@pytest.mark.parametrize("M, N, K", [(2048, 2048, 2048)])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [(128, 128, 64)])
@pytest.mark.parametrize("num_load_buffers, num_store_buffers", [(1, 1), (2, 2)])
@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.skipif(not is_hopper_or_newer(), reason="Requires Hopper or newer")
def test_gemm_warp_specialized(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_load_buffers, num_store_buffers,
                                          num_warps):
    a = torch.randn(M, K, device="cuda")
    b = torch.randn(K, N, device="cuda")
    c = torch.zeros(M, N, device="cuda")
    gemm_warp_specialized(a, b, c, BLOCK_M, BLOCK_N, BLOCK_K, num_load_buffers, num_store_buffers, num_warps)
    torch.testing.assert_close(a + b, c, atol=0, rtol=0)


if __name__ == "__main__":
    print("Benchmarking gemm")
    print("============================")
    M, N, K = 32 * 1024, 32 * 1024, 4096
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.zeros(M, N, device="cuda", dtype=torch.float16)
    BT = B.T.contiguous()
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
    num_load_buffers = 3
    num_store_buffers = 1
    num_warps = 4

    ms = triton.testing.do_bench(lambda: cublas.matmul(A, BT, C))
    print(f"cublas GEMM: {t7.get_flops(ms, M, N, K):.2f} TFLOPS/s")

    ms = triton.testing.do_bench(lambda: gemm_warp_specialized(  #
        A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_load_buffers, num_store_buffers, num_warps))
    print(f"Gluon GEMM: {t7.get_flops(ms, M, N, K):.2f} TFLOPS/s")
    # print()