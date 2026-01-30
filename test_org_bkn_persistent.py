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
    num_stages: tl.constexpr,
    WS: tl.constexpr,
    FLATTEN: tl.constexpr,
    SUBTILE: tl.constexpr,
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
        shape=[K, N],
        strides=[stride_bk, stride_bn],
        block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[stride_cm, stride_cn],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N // 2 if SUBTILE else BLOCK_SIZE_N],
    )


    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles = num_pid_m * num_pid_n

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    
    for pid in tl.range(start_pid, num_tiles, 132, num_stages=num_stages, warp_specialize=WS, flatten=FLATTEN):
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
        for kt in tl.range(k_tiles):
            offs_k = kt * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])   # [BM, BK]
            b = b_desc.load([offs_k, offs_bn])   # [BN, BK]
            acc = tl.dot(a, b, acc)            # [BM,BK] x [BK,BN] -> [BM,BN]

        if SUBTILE:
            acc = tl.reshape(acc, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            c0 = acc0.to(tl.bfloat16)
            c_desc.store([offs_am, offs_bn], c0)
            c1 = acc1.to(tl.bfloat16)
            c_desc.store([offs_am, offs_bn + BLOCK_SIZE_N // 2], c1)
        else:
            c = acc.to(tl.bfloat16)
            c_desc.store([offs_am, offs_bn], c)


    # c_desc = tl.make_tensor_descriptor(
    #     c_ptr,
    #     shape=[M, N],
    #     strides=[stride_cm, stride_cn],
    #     block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N // 4],
    # )
    # c_desc = tl.make_tensor_descriptor(
    #     c_ptr,
    #     shape=[M, N],
    #     strides=[stride_cm, stride_cn],
    #     block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N // 2],
    # )


    # acc = tl.reshape(acc, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
    # acc = tl.permute(acc, (0, 2, 1))
    # acc0, acc1 = tl.split(acc)
    # c0 = acc0.to(tl.bfloat16)
    # c_desc.store([offs_am, offs_bn], c0)
    # c1 = acc1.to(tl.bfloat16)
    # c_desc.store([offs_am, offs_bn + BLOCK_SIZE_N // 2], c1)
    
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
    num_ctas: int = 1   
    WS: bool = False
    FLATTEN: bool = False
    SUBTILE: bool = False


def run_kernel(A, Bkn, C, cfg: Cfg):
    M, K = A.shape
    N = Bkn.shape[1]

    # grid = (triton.cdiv(M, cfg.bm) * triton.cdiv(N, cfg.bn),)
    grid = (132,)

    matmul_kernel_tma[grid](
        A, Bkn, C,
        M, N, K,
        A.stride(0), A.stride(1),
        Bkn.stride(1), Bkn.stride(0),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=cfg.bm,
        BLOCK_SIZE_N=cfg.bn,
        BLOCK_SIZE_K=cfg.bk,
        GROUP_SIZE_M=cfg.group_m,
        num_warps=cfg.warps,     # launch option (ONLY ONCE)
        num_stages=cfg.stages,   # launch option (ONLY ONCE)
        num_ctas=cfg.num_ctas,   # launch option (ONLY ONCE)
        WS=cfg.WS,
        FLATTEN=cfg.FLATTEN,
        SUBTILE=cfg.SUBTILE,
    )

def bench(A, Bkn, C, cfg: Cfg, iters: int, warmup: int):
    # warmup
    for _ in range(warmup):
        run_kernel(A, Bkn, C, cfg)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        run_kernel(A, Bkn, C, cfg)
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / iters

    M, K = A.shape
    N = Bkn.shape[1]
    tflops = (2.0 * M * N * K) / (ms * 1e-3) / 1e12
    return ms, tflops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--group_m", type=int, default=8)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    iter_list = [100, 100, 10, 5, args.iters]
    multiplier_list = [1, 2, 4, 8, 16]

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    
    def alloc_fn(size, alignment, stream):
    # alignment/stream 参数这里可以先不管，Triton 只需要返回 CUDA 上的 buffer
        return torch.empty((size,), device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)


    # m, k, n = 2048, 2048, 256
    m, k, n = 2048, 256, 2048

    dtype = torch.bfloat16
    device = "cuda"
    csv_lines = []
    csv_lines.append("M,N,K,BM,BN,BK,GROUP_M,WARPS,STAGES,CTAS,WS,flatten,itrs,wmp,ms,TFLOPs")
    # print("M,N,K,BM,BN,BK,GROUP_M,WARPS,STAGES,CTAS,WS,flatten,itrs,wmp,ms,TFLOPs")
    header1 = (
        f"{'Problem Size':^20} | "
        f"{'Tile Shape':^22} | "
        f"{'Execution':^34} | "
        f"{'Result':^18}"
    )

    header2 = (
        f"{'M':>6} {'N':>6} {'K':>6} | "
        f"{'BM':>4} {'BN':>4} {'BK':>4} {'GROUP_M':>7} | "
        f"{'WARPS':>5} {'STAGES':>6} {'CTAS':>4} {'WS':>3} {'FLAT':>4} {'SUBTILE':>4} | "
        f"{'iters':>5} {'warmup':>7} {'ms':>9} {'TFLOPs':>8}"
    )

    print(header1)
    print(header2)
    # print("-" * len(header1))
    print("-" * len(header2))
    configs = []

    for BLOCK_M in [256, 128]:
        for BLOCK_N in [256, 128]:
            for BLOCK_K in [64]:
                for group_m in [1, 2, 4, 8]:
                        for stages in [3, 4, 5]:
                            for flatten in [False, True]:
                                configs.append(
                                Cfg(BLOCK_M, BLOCK_N, BLOCK_K, group_m, warps=8, stages=stages, num_ctas=2, WS=False, FLATTEN=flatten)
                                )
                                # if BLOCK_M == 256 and BLOCK_N == 256:    
                                #     continue
                                configs.append(
                                    Cfg(BLOCK_M, BLOCK_N, BLOCK_K, group_m, warps=8, stages=stages, num_ctas=1, WS=False, FLATTEN=flatten)
                                )
                                configs.append(
                                    Cfg(BLOCK_M, BLOCK_N, BLOCK_K, group_m, warps=4, stages=stages, num_ctas=1, WS=True, FLATTEN=flatten)
                                )
    # configs = [Cfg(256, 256, 64, group_m=8, warps=8, stages=4, num_ctas=2, WS=False)]
    # configs = [Cfg(128, 256, 64, group_m=8, warps=4, stages=3, num_ctas=1, WS=True)]
    # configs = [Cfg(128, 256, 64, group_m=8, warps=8, stages=3, num_ctas=1, WS=False, FLATTEN=True)]
    configs = [

        Cfg(128, 256, 64, group_m=args.group_m, warps=8, stages=3, num_ctas=1, WS=False, FLATTEN=False, SUBTILE=False),
        # Cfg(128, 256, 64, group_m=args.group_m, warps=8, stages=3, num_ctas=1, WS=False, FLATTEN=False, SUBTILE=True),
        # Cfg(128, 256, 64, group_m=args.group_m, warps=8, stages=4, num_ctas=1, WS=False, FLATTEN=False, SUBTILE=True),
        # # Cfg(128, 256, 64, group_m=args.group_m, warps=8, stages=4, num_ctas=1, WS=False, FLATTEN=False, SUBTILE=False),
        Cfg(128, 256, 64, group_m=args.group_m, warps=8, stages=3, num_ctas=1, WS=False, FLATTEN=True, SUBTILE=False),
        # Cfg(128, 256, 64, group_m=args.group_m, warps=8, stages=3, num_ctas=1, WS=False, FLATTEN=True, SUBTILE=True),
        # Cfg(128, 256, 64, group_m=args.group_m, warps=8, stages=4, num_ctas=1, WS=False, FLATTEN=True, SUBTILE=True),
        # Cfg(256, 256, 64, group_m=args.group_m, warps=8, stages=3, num_ctas=2, WS=False, FLATTEN=True, SUBTILE=False),
        # # Cfg(256, 256, 64, group_m=args.group_m, warps=8, stages=3, num_ctas=2, WS=False, FLATTEN=True, SUBTILE=True), # compilation failed 
        # # Cfg(256, 256, 64, group_m=args.group_m, warps=8, stages=4, num_ctas=2, WS=False, FLATTEN=True, SUBTILE=False), # shared memory 
        # # Cfg(256, 256, 64, group_m=args.group_m, warps=8, stages=4, num_ctas=2, WS=False, FLATTEN=True, SUBTILE=True), # compilation failed 
        
        
     
        # Cfg(128, 256, 64, group_m=args.group_m, warps=8, stages=3, num_ctas=1, WS=True, FLATTEN=False),
        # Cfg(128, 256, 64, group_m=args.group_m, warps=8, stages=3, num_ctas=1, WS=True, FLATTEN=False, SUBTILE=True),
        Cfg(128, 256, 64, group_m=args.group_m, warps=8, stages=3, num_ctas=1, WS=True, FLATTEN=True),
        Cfg(128, 256, 64, group_m=args.group_m, warps=8, stages=3, num_ctas=1, WS=True, FLATTEN=True, SUBTILE=True),
     
        # Cfg(128, 256, 64, group_m=8, warps=8, stages=4, num_ctas=1, WS=False, FLATTEN=True),
        ]

    # for id in range(len(iter_list)): 
    for id in [4]: 
        i = multiplier_list[id]
        args.iters = iter_list[id]
        M = i * m
        N = i * n
        K = i * k
        # B is [N, K] (so b.T is [K, N])
        A = torch.randn((M, K), device=device, dtype=dtype) * 0.1
        Bnk = torch.randn((N, K), device=device, dtype=dtype) * 0.1
        Bkn = Bnk.T.contiguous() 
        C = torch.empty((M, N), device=device, dtype=dtype)

        
        if args.check:
            ref = (A @ Bkn)

            # 2) 跑 kernel
            cfg0 = configs[0]
            # run_kernel(A, Bnk, C, cfg0)
            run_kernel(A, Bkn, C, cfg0)
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

        

        # seed += 1
        for cfg in configs:
            A = torch.randn((M, K), device=device, dtype=dtype) * 0.1
            Bnk = torch.randn((N, K), device=device, dtype=dtype) * 0.1
            Bkn = Bnk.T.contiguous() 
            current_cfg = f"{M},{N},{K},{cfg.bm},{cfg.bn},{cfg.bk},{cfg.group_m},{cfg.warps},{cfg.stages},{cfg.num_ctas},{cfg.WS},{cfg.FLATTEN},{args.iters},{args.warmup},"
            try:
                ms, tflops = bench(A, Bkn, C, cfg, iters=args.iters, warmup=args.warmup)
                # ms, tflops = bench(A, Bnk, C, cfg, iters=args.iters, warmup=args.warmup)
                current_cfg += f"{ms:.6f},{tflops:.2f}"
                csv_lines.append(current_cfg)
                line = (
                    f"{M:6d} {N:6d} {K:6d} | "
                    f"{cfg.bm:4d} {cfg.bn:4d} {cfg.bk:4d} {cfg.group_m:7d} | "
                    f"{cfg.warps:5d} {cfg.stages:6d} {cfg.num_ctas:4d} "
                    f"{int(cfg.WS):3d} {int(cfg.FLATTEN):4d}  {int(cfg.SUBTILE):6d} | "
                    f"{args.iters:5d} {args.warmup:7d} {ms:9.4f} {tflops:8.2f}"
                )
                print(line)
            except Exception as e:
                # Skip all errors (e.g., OutOfResources)
                current_cfg += f"NaN,NaN"
                print(current_cfg)
                print(f"Error: {type(e).__name__}: {e}")
                # assert 0
            if args.save:
                csv_output = "\n".join(csv_lines)
                with open(f"/home/tiger/triton/result/triton_dist_3.6_persistent.csv", "w") as f:
                    f.write(csv_output)
            torch.cuda.synchronize()
            import time
            time.sleep(20)
            torch.cuda.synchronize()
    if args.save:
        csv_output = "\n".join(csv_lines)
        with open(f"/home/tiger/triton/result/triton_dist_3.6_persistent.csv", "w") as f:
            f.write(csv_output)

if __name__ == "__main__":
    main()
