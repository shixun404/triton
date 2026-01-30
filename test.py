#!/usr/bin/env python3
import argparse
from dataclasses import dataclass

import torch
import triton
import triton.language as tl


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
    # Descriptors
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
    # c_desc = tl.make_tensor_descriptor(
    #     c_ptr,
    #     shape=[M, N],
    #     strides=[stride_cm, stride_cn],
    #     block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    # )

    # c_desc = tl.make_tensor_descriptor(
    #     c_ptr,
    #     shape=[M, N],
    #     strides=[stride_cm, stride_cn],
    #     block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N // 4],
    # )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[stride_cm, stride_cn],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N // 2],
    )

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M

    pid_m = first_pid_m + (pid % GROUP_SIZE_M)
    pid_n = (pid % num_pid_in_group) // GROUP_SIZE_M

    # # guard (in case grid is oversized)
    # if pid_m >= num_pid_m:
    #     return

    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Your requested warp_specialize=True
    for kt in tl.range(0, k_tiles, warp_specialize=WS):
        offs_k = kt * BLOCK_SIZE_K
        a = a_desc.load([offs_am, offs_k])   # [BM, BK]
        b = b_desc.load([offs_bn, offs_k])   # [BN, BK]
        acc = tl.dot(a, b.T, acc)            # [BM,BK] x [BK,BN] -> [BM,BN]

    acc = tl.reshape(acc, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
    acc = tl.permute(acc, (0, 2, 1))
    acc0, acc1 = tl.split(acc)
    c0 = acc0.to(tl.bfloat16)
    c_desc.store([offs_am, offs_bn], c0)
    c1 = acc1.to(tl.bfloat16)
    c_desc.store([offs_am, offs_bn + BLOCK_SIZE_N // 2], c1)
    
    # acc = tl.reshape(acc, (BLOCK_SIZE_M, 4, BLOCK_SIZE_N // 4))
    # acc = tl.permute(acc, (0, 2, 1))
    # acc0, acc1, acc2, acc3 = tl.split(acc)
    # c0 = acc0.to(tl.bfloat16)
    # c_desc.store([offs_am, offs_bn], c0)
    # c1 = acc1.to(tl.bfloat16)
    # c_desc.store([offs_am, offs_bn + BLOCK_SIZE_N // 4], c1)
    # c2 = acc2.to(tl.bfloat16)
    # c_desc.store([offs_am, offs_bn + 2 * (BLOCK_SIZE_N // 4)], c2)
    # c3 = acc3.to(tl.bfloat16)
    # c_desc.store([offs_am, offs_bn + 3 * (BLOCK_SIZE_N // 4)], c3)

    # c = acc.to(tl.bfloat16)
    # c_desc.store([offs_am, offs_bn], c)


    # c0 = acc[:, 0:BLOCK_SIZE_N // 2].to(tl.bfloat16)
    # c_desc.store([offs_am, offs_bn], c0)

    # c1 = acc[:, BLOCK_SIZE_N // 2:BLOCK_SIZE_N].to(tl.bfloat16)
    # c_desc.store([offs_am, offs_bn + BLOCK_SIZE_N // 2], c1)


@dataclass(frozen=True)
class Cfg:
    bm: int
    bn: int
    bk: int
    group_m: int
    warps: int
    stages: int
    num_ctas: int   
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
        num_ctas=cfg.num_ctas,,   # launch option (ONLY ONCE)
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
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    
    def alloc_fn(size, alignment, stream):
    # alignment/stream 参数这里可以先不管，Triton 只需要返回 CUDA 上的 buffer
        return torch.empty((size,), device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)


    m, k, n = 2048, 256, 2048

    dtype = torch.bfloat16
    device = "cuda"
    csv_lines = []
    csv_lines.append("M,N,K,BM,BN,BK,GROUP_M,WARPS,STAGES,WS,ms,TFLOPs")
    print("M,N,K,BM,BN,BK,GROUP_M,WARPS,STAGES,WS,ms,TFLOPs")
    for i in [16]:
        M = i * m
        N = i * n
        K = i * k
        # B is [N, K] (so b.T is [K, N])
        A = torch.randn((M, K), device=device, dtype=dtype) * 0.1
        Bnk = torch.randn((N, K), device=device, dtype=dtype) * 0.1
        C = torch.empty((M, N), device=device, dtype=dtype)

        configs = [
            Cfg(256, 128, 64, group_m=8, warps=4, stages=4, num_ctas=2, WS=True),
            # Cfg(128, 256, 64, group_m=8, warps=4, stages=3, WS=False),
            # Cfg(256, 128, 64, group_m=8, warps=8, stages=4),
            # Cfg(128, 128, 64, group_m=8, warps=8, stages=4),
            # Cfg(128, 256, 64, group_m=8, warps=4, stages=3),
            # Cfg(256, 128, 64, group_m=8, warps=4, stages=3),
        ]

        if args.check:
            # ref = (A @ Bnk.T).to(dtype)
            # cfg0 = configs[0]
            # run_kernel(A, Bnk, C, cfg0)
            # torch.cuda.synchronize()
            # # bf16 tolerance (adjust if needed)
            # max_abs = (C - ref).abs().max().item()
            # print(C[:4, :4], ref[:4, :4], (C - ref)[:4, :4])
            # print(f"[check] max_abs_error = {max_abs}")
            # 1) ref 用 fp32 计算更稳
            ref = (A @ Bnk.T)

            # 2) 跑 kernel
            cfg0 = configs[0]
            run_kernel(A, Bnk, C, cfg0)
            torch.cuda.synchronize()

            # 3) 误差统计（在 fp32 上比）
            diff = (C - ref).abs()
            max_abs = diff.max().item()

            # 避免除 0
            denom = ref.abs().clamp_min(1e-6)
            max_rel = (diff / denom).max().item()

            # NaN/Inf 检查
            bad = torch.isnan(C).any().item() or torch.isinf(C).any().item()

            print(f"[check] max_abs={max_abs:.6g}  max_rel={max_rel:.6g}  nan_or_inf={bad}")

            # 4) 给一个常用阈值（你可以按需求调）
            # bf16：常见经验阈值 abs 1e-1 ~ 1e0，rel 1e-2 ~ 1e-1（看规模/累加长度K）
            assert not bad

        

       
        for cfg in configs:
            try:
                ms, tflops = bench(A, Bnk, C, cfg, iters=args.iters, warmup=args.warmup)
                csv_lines.append(f"{M},{N},{K},{cfg.bm},{cfg.bn},{cfg.bk},{cfg.group_m},{cfg.warps},{cfg.stages},{cfg.WS}, {ms:.6f},{tflops:.2f}")
                print(f"{M},{N},{K},{cfg.bm},{cfg.bn},{cfg.bk},{cfg.group_m},{cfg.warps},{cfg.stages},{cfg.WS}, {ms:.6f},{tflops:.2f}")
            except Exception as e:
                # Skip all errors (e.g., OutOfResources)
                print(f"{M},{N},{K},{cfg.bm},{cfg.bn},{cfg.bk},{cfg.group_m},{cfg.warps},{cfg.stages},NaN,NaN  # {type(e).__name__}: {e}")
        import time
        time.sleep(2)

    csv_output = "\n".join(csv_lines)
    with open("/home/tiger/Triton-distributed/result/triton_dist_3.6_WS=1.csv", "w") as f:
        f.write(csv_output)

if __name__ == "__main__":
    main()
