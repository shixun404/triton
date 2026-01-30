#!/usr/bin/env python3
# bench_triton_tma_ws.py
#
# Bench two Triton GEMM kernels:
#  1) matmul_tma_ws_kernel
#  2) matmul_tma_persistent_ws_kernel
#
# Requirements you asked:
#  - GEMM shapes:
#      (M,N,K) = (n, n, n//8) and (n, n//8, n) for n in [2048,4096,8192,16384,32768]
#  - dtype: bfloat16
#  - explore as much parameter space as possible
#  - warmup=3, itr=10
#
# Output:
#  - CSV with per-config timing + TFLOPS

import argparse
import csv
import itertools
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
import triton
import triton.language as tl

if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] in [9, 10]:
    from triton._C.libtriton import nvidia
    cublas_workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    cublas = nvidia.cublas.CublasLt(cublas_workspace)
else:
    cublas = None


# -----------------------------
# HW / feature detection
# -----------------------------
def _cc():
    p = torch.cuda.get_device_properties("cuda")
    return p.major, p.minor


def is_hopper_or_blackwell() -> bool:
    major, minor = _cc()
    # Hopper = SM90 (9.0), Blackwell = SM100 (10.0) (and beyond)
    return major >= 9


def is_blackwell() -> bool:
    major, minor = _cc()
    return major >= 10


# -----------------------------
# Kernels (adapted for BF16 C)
# -----------------------------
@triton.jit
def _compute_pid(tile_id, num_pid_n, num_pid_m, GROUP_SIZE_M: tl.constexpr):
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


def exceeds_smem_capacity(num_stages: int, BLOCK_M: int, BLOCK_N: int, BLOCK_K: int, elem_bytes: int = 2) -> bool:
    # Rough filter copied from your test; keep 228KB limit.
    # (num_stages * BK * (BM + BN) + BM*BN) * elem_bytes
    return (num_stages * BLOCK_K * (BLOCK_M + BLOCK_N) + BLOCK_M * BLOCK_N) * elem_bytes > 228 * 1024

@triton.jit
def matmul_tma_ws_kernel(  #
        a_ptr, b_ptr, c_ptr,  #
        a_stride0, a_stride1,  #
        b_stride0, b_stride1,  #
        c_stride0, c_stride1,  #
        M, N, K,  #
        num_stages: tl.constexpr,  #
        BLOCK_SIZE_M: tl.constexpr,  #
        BLOCK_SIZE_N: tl.constexpr,  #
        BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        WS: tl.constexpr,  #
):
    a_desc = tl.make_tensor_descriptor(a_ptr, shape=[M, K], strides=[a_stride0, a_stride1],
                                       block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K])
    b_desc = tl.make_tensor_descriptor(b_ptr, shape=[N, K], strides=[b_stride0, b_stride1],
                                       block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K])
    c_desc = tl.make_tensor_descriptor(c_ptr, shape=[M, N], strides=[c_stride0, c_stride1],
                                       block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N])

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m, pid_n = _compute_pid(pid, num_pid_n, num_pid_m, GROUP_SIZE_M)

    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

    off_am = pid_m * BLOCK_SIZE_M
    off_bn = pid_n * BLOCK_SIZE_N
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(k_tiles, warp_specialize=WS, num_stages=num_stages):
        off_k = k * BLOCK_SIZE_K
        a = a_desc.load((off_am, off_k))
        b = b_desc.load((off_bn, off_k))
        accumulator = tl.dot(a, b.T, accumulator)

    c = accumulator.to(tl.bfloat16)
    c_desc.store((off_am, off_bn), c)



@triton.jit
def matmul_tma_persistent_ws_kernel(  #
        a_ptr, b_ptr, c_ptr,  #
        a_stride0, a_stride1,  #
        b_stride0, b_stride1,  #
        c_stride0, c_stride1,  #
        M, N, K,  #
        num_stages: tl.constexpr,  #
        BLOCK_SIZE_M: tl.constexpr,  #
        BLOCK_SIZE_N: tl.constexpr,  #
        BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        NUM_SMS: tl.constexpr,  #
        FLATTEN: tl.constexpr,  #
        WS: tl.constexpr,  #   
):
    a_desc = tl.make_tensor_descriptor(a_ptr, shape=[M, K], strides=[a_stride0, a_stride1],
                                       block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K])
    b_desc = tl.make_tensor_descriptor(b_ptr, shape=[N, K], strides=[b_stride0, b_stride1],
                                       block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K])
    c_desc = tl.make_tensor_descriptor(c_ptr, shape=[M, N], strides=[c_stride0, c_stride1],
                                       block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N])

    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=FLATTEN, warp_specialize=WS,
                            num_stages=num_stages):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_n, num_pid_m, GROUP_SIZE_M)

        off_am = pid_m * BLOCK_SIZE_M
        off_bn = pid_n * BLOCK_SIZE_N
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            off_k = ki * BLOCK_SIZE_K
            a = a_desc.load((off_am, off_k))
            b = b_desc.load((off_bn, off_k))
            accumulator = tl.dot(a, b.T, accumulator)

        c = accumulator.to(tl.bfloat16)
        c_desc.store((off_am, off_bn), c)


# -----------------------------
# Bench harness
# -----------------------------
@dataclass(frozen=True)
class Config:
    kernel: str  # "ws" or "persistent"
    BM: int
    BN: int
    BK: int
    stages: int
    warps: int
    group_m: int
    flatten: int  # 0/1
    ws: int
    num_sms: int  # only for persistent; else 0
    


def grid_ws(M: int, N: int, meta: Dict) -> Tuple[int]:
    return (triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),)


def grid_persistent(M: int, N: int, num_sms: int, meta: Dict) -> Tuple[int]:
    tiles = triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"])
    return (min(num_sms, tiles),)


def tflops(M: int, N: int, K: int, ms: float) -> float:
    # 2*M*N*K ops, ms -> seconds
    return (2.0 * M * N * K) / (ms * 1e-3) / 1e12


def gen_param_space(M: int, N: int, K: int, sm_count: int) -> Iterable[Config]:
    # Big-ish but still sane; tune these lists if you want *even more*.
    BM_list = [64, 128, 256]
    BN_list = [64, 128]
    BK_list = [64]
    stages_list = [2, 3, 4, 5]
    warps_list = [4, 8]
    group_m_list = [8]
    ws_list = [0, 1]

    # persistent-only:
    # explore a few NUM_SMS choices to see occupancy / scheduling effects
    num_sms_list = [132]

    flatten_list = [1] if not is_blackwell() else [0, 1]

    for BM, BN, BK, stg, wp, gm, ws in itertools.product(
        BM_list, BN_list, BK_list, stages_list, warps_list, group_m_list, ws_list
    ):
        # These kernels do *not* mask OOB loads/stores; require exact tiling.
        if (M % BM) != 0 or (N % BN) != 0 or (K % BK) != 0:
            continue
        if BN > N or BM > M or BK > K:
            continue
        if exceeds_smem_capacity(stg, BM, BN, BK, elem_bytes=2):  # BF16 = 2 bytes
            continue

        # ws kernel config
        for fl in [1]:  # not used by ws kernel (kept constant)
            yield Config("ws", BM, BN, BK, stg, wp, gm, fl, ws, 0)

        # persistent kernel configs
        for fl, nsms in itertools.product(flatten_list, num_sms_list):
            yield Config("persistent", BM, BN, BK, stg, wp, gm, fl, ws, nsms)


def bench_one(
    cfg: Config,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    M: int,
    N: int,
    K: int,
    warmup: int,
    iters: int,
) -> float:
    # Use Triton’s benchmarking helper (CUDA events under the hood).
    # It returns milliseconds.
    if cfg.kernel == "ws":
        meta = dict(
            num_stages=cfg.stages,
            BLOCK_SIZE_M=cfg.BM,
            BLOCK_SIZE_N=cfg.BN,
            BLOCK_SIZE_K=cfg.BK,
            GROUP_SIZE_M=cfg.group_m,
            num_warps=cfg.warps,
            WS=cfg.ws
        )
        grid = lambda META: grid_ws(M, N, META)

        def run():
            matmul_tma_ws_kernel[grid](
                A, B, C,
                *A.stride(), *B.stride(), *C.stride(),
                M, N, K,
                **meta,
            )

    else:
        meta = dict(
            num_stages=cfg.stages,
            BLOCK_SIZE_M=cfg.BM,
            BLOCK_SIZE_N=cfg.BN,
            BLOCK_SIZE_K=cfg.BK,
            GROUP_SIZE_M=cfg.group_m,
            NUM_SMS=cfg.num_sms,
            FLATTEN=bool(cfg.flatten) and is_blackwell(),
            num_warps=cfg.warps,
            WS=cfg.ws
        )
        grid = lambda META: grid_persistent(M, N, cfg.num_sms, META)

        def run():
            matmul_tma_persistent_ws_kernel[grid](
                A, B, C,
                *A.stride(), *B.stride(), *C.stride(),
                M, N, K,
                **meta,
            )

    # Make sure compile happens before timing loops (first call inside do_bench would include compile otherwise)
    run()
    torch.cuda.synchronize()

    ms = triton.testing.do_bench(run, warmup=warmup, rep=iters)
    ref_out = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
    cublas.matmul(A, B, ref_out)
    torch.testing.assert_close(ref_out.to(torch.bfloat16), C.to(torch.bfloat16), atol=0.03, rtol=0.03)
    return float(ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="bench_tma_ws.csv")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kernel", type=str, default="both", choices=["both", "ws", "persistent"])
    ap.add_argument("--max-configs-per-shape", type=int, default=0,
                    help="0 = no limit (may take a long time). If set, keeps best-N random subset.")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    if not is_hopper_or_blackwell():
        raise RuntimeError("Requires Hopper (SM90) or newer (Blackwell, etc.) for warp_specialize + TMA.")

    torch.manual_seed(args.seed)
    device = "cuda"
    props = torch.cuda.get_device_properties(device)
    sm_count = props.multi_processor_count

    n_list = [2048, 4096, 8192, 16384, 32768]
    shapes: List[Tuple[int, int, int, str]] = []
    for n in n_list:
        shapes.append((n, n, n // 8, "n,n,n//8"))
        shapes.append((n, n // 8, n, "n,n//8,n"))

    # CSV header
    fieldnames = [
        "kernel",
        "shape_tag",
        "M", "N", "K",
        "dtype",
        "BM", "BN", "BK",
        "stages", "warps", "group_m",
        "flatten", "ws", "num_sms",
        "ms", "tflops",
    ]

    def alloc_fn(size: int, _, __):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for (M, N, K, tag) in shapes:
            # B is [N, K] per your kernel convention (dot(a, b.T))
            A = torch.randn((M, K), device=device, dtype=torch.bfloat16)
            B = torch.randn((N, K), device=device, dtype=torch.bfloat16)
            C = torch.empty((M, N), device=device, dtype=torch.bfloat16)

            cfgs = list(gen_param_space(M, N, K, sm_count))
            # optional: limit configs per shape (random subset)
            if args.max_configs_per_shape and args.max_configs_per_shape > 0 and len(cfgs) > args.max_configs_per_shape:
                g = torch.Generator(device="cpu")
                g.manual_seed(args.seed + M + N + K)
                idx = torch.randperm(len(cfgs), generator=g)[: args.max_configs_per_shape].tolist()
                cfgs = [cfgs[i] for i in idx]
            
            for cfg in cfgs:
                if args.kernel != "both" and cfg.kernel != args.kernel:
                    continue
                try:
                    ms = bench_one(cfg, A, B, C, M, N, K, warmup=args.warmup, iters=args.iters)
                except Exception as e:
                    # Skip configs that fail to compile/run
                    print(f"[SKIP] {cfg} ({e})")
                    continue

                row = dict(
                    kernel=cfg.kernel,
                    shape_tag=tag,
                    M=M, N=N, K=K,
                    dtype="bf16",
                    BM=cfg.BM, BN=cfg.BN, BK=cfg.BK,
                    stages=cfg.stages, warps=cfg.warps, group_m=cfg.group_m,
                    flatten=cfg.flatten if cfg.kernel == "persistent" else 0,
                    ws=cfg.ws,
                    num_sms=cfg.num_sms if cfg.kernel == "persistent" else 0,
                    ms=ms,
                    tflops=tflops(M, N, K, ms),
                )
                w.writerow(row)
                f.flush()

    print(f"[OK] Wrote {args.out}")


if __name__ == "__main__":
    main()
