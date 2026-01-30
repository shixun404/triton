#!/usr/bin/env python3
import argparse
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

# @triton.jit
# def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    # group_id = tile_id // num_pid_in_group
    # first_pid_m = group_id * GROUP_SIZE_M
    # group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    # pid_m = first_pid_m + (tile_id % group_size_m)
    # pid_n = (tile_id % num_pid_in_group) // group_size_m
    # return pid_m, pid_n

@triton.jit
def _compute_pid(tile_id, num_pid_n, num_pid_m, GROUP_SIZE_M: tl.constexpr):
    # linearize within the (GROUP_SIZE_M, num_pid_n) rectangle
    pid_n = (tile_id % (GROUP_SIZE_M * num_pid_n)) // GROUP_SIZE_M
    pid_m_in_group = (tile_id % (GROUP_SIZE_M * num_pid_n)) % GROUP_SIZE_M

    group_id = tile_id // (GROUP_SIZE_M * num_pid_n)
    pid_m = group_id * GROUP_SIZE_M + pid_m_in_group
    return pid_m, pid_n

@triton.jit
def matmul_kernel_tma(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,   # A: [M, K]
    stride_bn, stride_bk,   # B: [N, K]
    stride_cm, stride_cn,   # C: [M, N]
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    WS: tl.constexpr,
):

    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[stride_am, stride_ak],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )

    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[N, K],
        strides=[stride_bn, stride_bk],
        block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
    )

    c_desc = tl.make_tensor_descriptor(
    c_ptr, shape=[M, N], strides=[stride_cm, stride_cn], 
    block_shape=[
        BLOCK_SIZE_M,
        BLOCK_SIZE_N
        ]
    )

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    tile_id = pid
    for tile_id in tl.range(pid, num_tiles, 132, 
                    warp_specialize=WS):

        # pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, 132)
        pid_m, pid_n = _compute_pid(tile_id, num_pid_n, num_pid_m, GROUP_SIZE_M)
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
        for ki in range(0, k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])          # [BM, BK]
            b = b_desc.load([offs_bn, offs_k])          # [BK, BN]
            acc = tl.dot(a, b.T, acc)
        dtype = c_ptr.dtype.element_ty

        c = acc.to(dtype)
        c_desc.store([offs_am, offs_bn], c)



@dataclass(frozen=True)
class Cfg:
    bm: int
    bn: int
    bk: int
    group_m: int
    warps: int
    stages: int
    WS: bool = False


def run_kernel(A, Bnk, C, cfg: Cfg):
    M, K = A.shape
    N = Bnk.shape[0]

    grid = (triton.cdiv(M, cfg.bm) * triton.cdiv(N, cfg.bn),)

    matmul_kernel_tma[grid](
        A, Bnk, C,
        M, N, K,
        A.stride(0), A.stride(1),
        Bnk.stride(0), Bnk.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=cfg.bm,
        BLOCK_SIZE_N=cfg.bn,
        BLOCK_SIZE_K=cfg.bk,
        GROUP_SIZE_M=cfg.group_m,
        num_warps=cfg.warps,     # launch option (ONLY ONCE)
        num_stages=cfg.stages,   # launch option (ONLY ONCE)
        WS=cfg.WS,
    )


def bench(A, Bnk, C, cfg: Cfg, iters: int, warmup: int):
    # warmup
    for _ in range(warmup):
        run_kernel(A, Bnk, C, cfg)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        run_kernel(A, Bnk, C, cfg)
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / iters

    M, K = A.shape
    N = Bnk.shape[0]
    tflops = (2.0 * M * N * K) / (ms * 1e-3) / 1e12
    return ms, tflops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    
    def alloc_fn(size, alignment, stream):
    # alignment/stream 参数这里可以先不管，Triton 只需要返回 CUDA 上的 buffer
        return torch.empty((size,), device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)


    M, N, K = 32768, 32768, 4096
    dtype = torch.bfloat16
    device = "cuda"

    # B is [N, K] (so b.T is [K, N])
    A = torch.randn((M, K), device=device, dtype=dtype)
    Bnk = torch.randn((N, K), device=device, dtype=dtype)
    C = torch.empty((M, N), device=device, dtype=dtype)

    configs = [
        Cfg(128, 128, 64, group_m=8, warps=4, stages=3, WS=True),
        # Cfg(128, 256, 64, group_m=8, warps=4, stages=3, WS=False),
        # Cfg(256, 128, 64, group_m=8, warps=8, stages=4),
        # Cfg(128, 128, 64, group_m=8, warps=8, stages=4),
        # Cfg(128, 256, 64, group_m=8, warps=4, stages=3),
        # Cfg(256, 128, 64, group_m=8, warps=4, stages=3),
    ]

    if args.check:
        ref = (A @ Bnk.T).to(dtype)
        cfg0 = configs[0]
        run_kernel(A, Bnk, C, cfg0)
        torch.cuda.synchronize()
        # bf16 tolerance (adjust if needed)
        max_abs = (C - ref).abs().max().item()
        print(f"[check] max_abs_error = {max_abs}")

    print("BM,BN,BK,GROUP_M,WARPS,STAGES,WS,ms,TFLOPs")
    for cfg in configs:
        try:
            ms, tflops = bench(A, Bnk, C, cfg, iters=args.iters, warmup=args.warmup)
            print(f"{cfg.bm},{cfg.bn},{cfg.bk},{cfg.group_m},{cfg.warps},{cfg.stages},{cfg.WS}, {ms:.6f},{tflops:.2f}")
        except Exception as e:
            # Skip all errors (e.g., OutOfResources)
            print(f"{cfg.bm},{cfg.bn},{cfg.bk},{cfg.group_m},{cfg.warps},{cfg.stages},NaN,NaN  # {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
