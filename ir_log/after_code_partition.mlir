module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel_tma(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %c0_i32 = arith.constant 0 : i32
    %1 = ttg.memdesc_index %0[%c0_i32] : !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.init_barrier %1, 1 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %c1_i32 = arith.constant 1 : i32
    %2 = ttg.memdesc_index %0[%c1_i32] : !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.init_barrier %2, 1 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %c2_i32 = arith.constant 2 : i32
    %3 = ttg.memdesc_index %0[%c2_i32] : !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.init_barrier %3, 1 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %c3_i32 = arith.constant 3 : i32
    %4 = ttg.memdesc_index %0[%c3_i32] : !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.init_barrier %4, 1 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %5 = ttg.local_alloc : () -> !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %c0_i32_0 = arith.constant 0 : i32
    %6 = ttg.memdesc_index %5[%c0_i32_0] : !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.init_barrier %6, 1 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %c1_i32_1 = arith.constant 1 : i32
    %7 = ttg.memdesc_index %5[%c1_i32_1] : !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.init_barrier %7, 1 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %c2_i32_2 = arith.constant 2 : i32
    %8 = ttg.memdesc_index %5[%c2_i32_2] : !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.init_barrier %8, 1 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %c3_i32_3 = arith.constant 3 : i32
    %9 = ttg.memdesc_index %5[%c3_i32_3] : !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.init_barrier %9, 1 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %10 = ttg.local_alloc : () -> !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %c0_i32_4 = arith.constant 0 : i32
    %11 = ttg.memdesc_index %10[%c0_i32_4] : !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.init_barrier %11, 1 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %c1_i32_5 = arith.constant 1 : i32
    %12 = ttg.memdesc_index %10[%c1_i32_5] : !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.init_barrier %12, 1 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %c2_i32_6 = arith.constant 2 : i32
    %13 = ttg.memdesc_index %10[%c2_i32_6] : !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.init_barrier %13, 1 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %c3_i32_7 = arith.constant 3 : i32
    %14 = ttg.memdesc_index %10[%c3_i32_7] : !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.init_barrier %14, 1 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %15 = nvws.create_token {loadType = 2 : i32, numBuffers = 4 : i32} : tensor<4x!nvws.token>
    %16 = nvws.create_token {loadType = 2 : i32, numBuffers = 4 : i32} : tensor<4x!nvws.token>
    %17 = nvws.create_token {loadType = 2 : i32, numBuffers = 4 : i32} : tensor<4x!nvws.token>
    %18 = nvws.create_token {loadType = 2 : i32, numBuffers = 4 : i32} : tensor<4x!nvws.token>
    %19 = ttg.local_alloc : () -> !ttg.memdesc<4x128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>
    %20 = ttg.local_alloc : () -> !ttg.memdesc<4x128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>
    %21 = ttg.local_alloc : () -> !ttg.memdesc<4x128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>
    ttg.warp_specialize(%arg8, %arg2, %arg3, %arg4, %arg5, %10, %19, %0, %21, %15, %17, %5, %20, %18, %16) attributes {requestedRegisters = array<i32: 232, 232>}
    default {
      %c1_i64 = arith.constant {async_task_id = array<i32: 0>} 1 : i64
      %c8_i32 = arith.constant {async_task_id = array<i32: 0>} 8 : i32
      %c256_i32 = arith.constant {async_task_id = array<i32: 0>} 256 : i32
      %c128_i32 = arith.constant {async_task_id = array<i32: 0>} 128 : i32
      %c64_i32 = arith.constant {async_task_id = array<i32: 0>} 64 : i32
      %c0_i32_8 = arith.constant {async_task_id = array<i32: 0>} 0 : i32
      %c1_i32_9 = arith.constant {async_task_id = array<i32: 0>} 1 : i32
      %c127_i32 = arith.constant {async_task_id = array<i32: 0>} 127 : i32
      %c63_i32 = arith.constant {async_task_id = array<i32: 0>} 63 : i32
      %22 = arith.extsi %arg6 {async_task_id = array<i32: 0>} : i32 to i64
      %23 = tt.make_tensor_descriptor %arg0, [%arg3, %arg5], [%22, %c1_i64] {async_task_id = array<i32: 0>} : <bf16>, <tensor<128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>>
      %24 = tt.make_tensor_descriptor %arg0, [%arg3, %arg5], [%22, %c1_i64] {async_task_id = array<i32: 0>} : <bf16>, <tensor<128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>>
      %25 = arith.extsi %arg7 {async_task_id = array<i32: 0>} : i32 to i64
      %26 = tt.make_tensor_descriptor %arg1, [%arg4, %arg5], [%25, %c1_i64] {async_task_id = array<i32: 0>} : <bf16>, <tensor<128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>>
      %27 = tt.get_program_id x {async_task_id = array<i32: 0>} : i32
      %28 = arith.addi %arg4, %c127_i32 {async_task_id = array<i32: 0>} : i32
      %29 = arith.divsi %28, %c128_i32 {async_task_id = array<i32: 0>} : i32
      %30 = arith.muli %29, %c8_i32 {async_task_id = array<i32: 0>} : i32
      %31 = arith.divsi %27, %30 {async_task_id = array<i32: 0>} : i32
      %32 = arith.muli %31, %c8_i32 {async_task_id = array<i32: 0>} : i32
      %33 = arith.remsi %27, %c8_i32 {async_task_id = array<i32: 0>} : i32
      %34 = arith.addi %32, %33 {async_task_id = array<i32: 0>} : i32
      %35 = arith.remsi %27, %30 {async_task_id = array<i32: 0>} : i32
      %36 = arith.divsi %35, %c8_i32 {async_task_id = array<i32: 0>} : i32
      %37 = arith.addi %arg5, %c63_i32 {async_task_id = array<i32: 0>} : i32
      %38 = arith.divsi %37, %c64_i32 {async_task_id = array<i32: 0>} : i32
      %39 = arith.muli %34, %c256_i32 {async_task_id = array<i32: 0>} : i32
      %40 = arith.addi %39, %c128_i32 {async_task_id = array<i32: 0>} : i32
      %41 = arith.muli %36, %c128_i32 {async_task_id = array<i32: 0>} : i32
      %c0_i64 = arith.constant {async_task_id = array<i32: 0>} 0 : i64
      %42 = scf.for %arg9 = %c0_i32_8 to %38 step %c1_i32_9 iter_args(%arg10 = %c0_i64) -> (i64)  : i32 {
        %43 = arith.muli %arg9, %c64_i32 {async_task_id = array<i32: 0>} : i32
        %c4_i32 = arith.constant {async_task_id = array<i32: 0>} 4 : i32
        %44 = arith.extsi %c4_i32 {async_task_id = array<i32: 0>} : i32 to i64
        %45 = arith.divui %arg10, %44 {async_task_id = array<i32: 0>} : i64
        %46 = arith.muli %45, %44 {async_task_id = array<i32: 0>} : i64
        %47 = arith.subi %arg10, %46 {async_task_id = array<i32: 0>} : i64
        %48 = arith.trunci %47 {async_task_id = array<i32: 0>} : i64 to i32
        %c1_i64_10 = arith.constant {async_task_id = array<i32: 0>} 1 : i64
        %49 = arith.andi %45, %c1_i64_10 {async_task_id = array<i32: 0>} : i64
        %50 = arith.trunci %49 {async_task_id = array<i32: 0>} : i64 to i1
        %c4_i32_11 = arith.constant {async_task_id = array<i32: 0>} 4 : i32
        %51 = arith.extsi %c4_i32_11 {async_task_id = array<i32: 0>} : i32 to i64
        %52 = arith.divui %arg10, %51 {async_task_id = array<i32: 0>} : i64
        %53 = arith.muli %52, %51 {async_task_id = array<i32: 0>} : i64
        %54 = arith.subi %arg10, %53 {async_task_id = array<i32: 0>} : i64
        %55 = arith.trunci %54 {async_task_id = array<i32: 0>} : i64 to i32
        %c1_i64_12 = arith.constant {async_task_id = array<i32: 0>} 1 : i64
        %56 = arith.andi %52, %c1_i64_12 {async_task_id = array<i32: 0>} : i64
        %57 = arith.trunci %56 {async_task_id = array<i32: 0>} : i64 to i1
        nvws.producer_acquire %15, %55, %57 {async_task_id = array<i32: 0>} : tensor<4x!nvws.token>, i32, i1
        %58 = ttg.memdesc_index %10[%55] {async_task_id = array<i32: 0>} : !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
        %true = arith.constant {async_task_id = array<i32: 0>} true
        ttng.barrier_expect %58, 16384 {async_task_id = array<i32: 0>}, %true : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
        %59 = ttg.memdesc_index %19[%55] {async_task_id = array<i32: 0>} : !ttg.memdesc<4x128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>
        ttng.async_tma_copy_global_to_local %23[%39, %43] %59, %58, %true {async_task_id = array<i32: 0>} : !tt.tensordesc<tensor<128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>>, !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>
        %c4_i32_13 = arith.constant {async_task_id = array<i32: 0>} 4 : i32
        %60 = arith.extsi %c4_i32_13 {async_task_id = array<i32: 0>} : i32 to i64
        %61 = arith.divui %arg10, %60 {async_task_id = array<i32: 0>} : i64
        %62 = arith.muli %61, %60 {async_task_id = array<i32: 0>} : i64
        %63 = arith.subi %arg10, %62 {async_task_id = array<i32: 0>} : i64
        %64 = arith.trunci %63 {async_task_id = array<i32: 0>} : i64 to i32
        %c1_i64_14 = arith.constant {async_task_id = array<i32: 0>} 1 : i64
        %65 = arith.andi %61, %c1_i64_14 {async_task_id = array<i32: 0>} : i64
        %66 = arith.trunci %65 {async_task_id = array<i32: 0>} : i64 to i1
        %c4_i32_15 = arith.constant {async_task_id = array<i32: 0>} 4 : i32
        %67 = arith.extsi %c4_i32_15 {async_task_id = array<i32: 0>} : i32 to i64
        %68 = arith.divui %arg10, %67 {async_task_id = array<i32: 0>} : i64
        %69 = arith.muli %68, %67 {async_task_id = array<i32: 0>} : i64
        %70 = arith.subi %arg10, %69 {async_task_id = array<i32: 0>} : i64
        %71 = arith.trunci %70 {async_task_id = array<i32: 0>} : i64 to i32
        %c1_i64_16 = arith.constant {async_task_id = array<i32: 0>} 1 : i64
        %72 = arith.andi %68, %c1_i64_16 {async_task_id = array<i32: 0>} : i64
        %73 = arith.trunci %72 {async_task_id = array<i32: 0>} : i64 to i1
        nvws.producer_acquire %18, %71, %73 {async_task_id = array<i32: 0>} : tensor<4x!nvws.token>, i32, i1
        nvws.producer_acquire %17, %71, %73 {async_task_id = array<i32: 0>} : tensor<4x!nvws.token>, i32, i1
        %74 = ttg.memdesc_index %0[%71] {async_task_id = array<i32: 0>} : !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
        %true_17 = arith.constant {async_task_id = array<i32: 0>} true
        ttng.barrier_expect %74, 16384 {async_task_id = array<i32: 0>}, %true_17 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
        %75 = ttg.memdesc_index %21[%71] {async_task_id = array<i32: 0>} : !ttg.memdesc<4x128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>
        ttng.async_tma_copy_global_to_local %26[%41, %43] %75, %74, %true_17 {async_task_id = array<i32: 0>} : !tt.tensordesc<tensor<128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>>, !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>
        %c4_i32_18 = arith.constant {async_task_id = array<i32: 0>} 4 : i32
        %76 = arith.extsi %c4_i32_18 {async_task_id = array<i32: 0>} : i32 to i64
        %77 = arith.divui %arg10, %76 {async_task_id = array<i32: 0>} : i64
        %78 = arith.muli %77, %76 {async_task_id = array<i32: 0>} : i64
        %79 = arith.subi %arg10, %78 {async_task_id = array<i32: 0>} : i64
        %80 = arith.trunci %79 {async_task_id = array<i32: 0>} : i64 to i32
        %c1_i64_19 = arith.constant {async_task_id = array<i32: 0>} 1 : i64
        %81 = arith.andi %77, %c1_i64_19 {async_task_id = array<i32: 0>} : i64
        %82 = arith.trunci %81 {async_task_id = array<i32: 0>} : i64 to i1
        %c4_i32_20 = arith.constant {async_task_id = array<i32: 0>} 4 : i32
        %83 = arith.extsi %c4_i32_20 {async_task_id = array<i32: 0>} : i32 to i64
        %84 = arith.divui %arg10, %83 {async_task_id = array<i32: 0>} : i64
        %85 = arith.muli %84, %83 {async_task_id = array<i32: 0>} : i64
        %86 = arith.subi %arg10, %85 {async_task_id = array<i32: 0>} : i64
        %87 = arith.trunci %86 {async_task_id = array<i32: 0>} : i64 to i32
        %c1_i64_21 = arith.constant {async_task_id = array<i32: 0>} 1 : i64
        %88 = arith.andi %84, %c1_i64_21 {async_task_id = array<i32: 0>} : i64
        %89 = arith.trunci %88 {async_task_id = array<i32: 0>} : i64 to i1
        nvws.producer_acquire %16, %87, %89 {async_task_id = array<i32: 0>} : tensor<4x!nvws.token>, i32, i1
        %90 = ttg.memdesc_index %5[%87] {async_task_id = array<i32: 0>} : !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
        %true_22 = arith.constant {async_task_id = array<i32: 0>} true
        ttng.barrier_expect %90, 16384 {async_task_id = array<i32: 0>}, %true_22 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
        %91 = ttg.memdesc_index %20[%87] {async_task_id = array<i32: 0>} : !ttg.memdesc<4x128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>
        ttng.async_tma_copy_global_to_local %24[%40, %43] %91, %90, %true_22 {async_task_id = array<i32: 0>} : !tt.tensordesc<tensor<128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>>, !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>
        %c1_i64_23 = arith.constant {async_task_id = array<i32: 0>} 1 : i64
        %92 = arith.addi %arg10, %c1_i64_23 {async_task_id = array<i32: 0>} : i64
        scf.yield {async_task_id = array<i32: 0>} %92 : i64
      } {async_task_id = array<i32: 0>}
      ttg.warp_yield
    }
    partition0(%arg9: i32, %arg10: !tt.ptr<bf16>, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>, %arg15: !ttg.memdesc<4x128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>, %arg16: !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>, %arg17: !ttg.memdesc<4x128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>, %arg18: tensor<4x!nvws.token>, %arg19: tensor<4x!nvws.token>, %arg20: !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>, %arg21: !ttg.memdesc<4x128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>, %arg22: tensor<4x!nvws.token>, %arg23: tensor<4x!nvws.token>) num_warps(4) {
      %c1_i64 = arith.constant {async_task_id = array<i32: 1>} 1 : i64
      %c8_i32 = arith.constant {async_task_id = array<i32: 1>} 8 : i32
      %c256_i32 = arith.constant {async_task_id = array<i32: 1>} 256 : i32
      %c128_i32 = arith.constant {async_task_id = array<i32: 1>} 128 : i32
      %c64_i32 = arith.constant {async_task_id = array<i32: 1>} 64 : i32
      %c0_i32_8 = arith.constant {async_task_id = array<i32: 1>} 0 : i32
      %c1_i32_9 = arith.constant {async_task_id = array<i32: 1>} 1 : i32
      %c127_i32 = arith.constant {async_task_id = array<i32: 1>} 127 : i32
      %c63_i32 = arith.constant {async_task_id = array<i32: 1>} 63 : i32
      %cst = arith.constant {async_task_id = array<i32: 1>} dense<0.000000e+00> : tensor<128x128xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>>
      %22 = arith.extsi %arg9 {async_task_id = array<i32: 1>} : i32 to i64
      %23 = tt.make_tensor_descriptor %arg10, [%arg11, %arg12], [%22, %c1_i64] {async_task_id = array<i32: 1>} : <bf16>, <tensor<128x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>>
      %24 = tt.get_program_id x {async_task_id = array<i32: 1>} : i32
      %25 = arith.addi %arg12, %c127_i32 {async_task_id = array<i32: 1>} : i32
      %26 = arith.divsi %25, %c128_i32 {async_task_id = array<i32: 1>} : i32
      %27 = arith.muli %26, %c8_i32 {async_task_id = array<i32: 1>} : i32
      %28 = arith.divsi %24, %27 {async_task_id = array<i32: 1>} : i32
      %29 = arith.muli %28, %c8_i32 {async_task_id = array<i32: 1>} : i32
      %30 = arith.remsi %24, %c8_i32 {async_task_id = array<i32: 1>} : i32
      %31 = arith.addi %29, %30 {async_task_id = array<i32: 1>} : i32
      %32 = arith.remsi %24, %27 {async_task_id = array<i32: 1>} : i32
      %33 = arith.divsi %32, %c8_i32 {async_task_id = array<i32: 1>} : i32
      %34 = arith.addi %arg13, %c63_i32 {async_task_id = array<i32: 1>} : i32
      %35 = arith.divsi %34, %c64_i32 {async_task_id = array<i32: 1>} : i32
      %36 = arith.muli %31, %c256_i32 {async_task_id = array<i32: 1>} : i32
      %37 = arith.muli %33, %c128_i32 {async_task_id = array<i32: 1>} : i32
      %c0_i64 = arith.constant {async_task_id = array<i32: 1>} 0 : i64
      %38:2 = scf.for %arg24 = %c0_i32_8 to %35 step %c1_i32_9 iter_args(%arg25 = %cst, %arg26 = %c0_i64) -> (tensor<128x128xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>>, i64)  : i32 {
        %c4_i32 = arith.constant {async_task_id = array<i32: 1>} 4 : i32
        %41 = arith.extsi %c4_i32 {async_task_id = array<i32: 1>} : i32 to i64
        %42 = arith.divui %arg26, %41 {async_task_id = array<i32: 1>} : i64
        %43 = arith.muli %42, %41 {async_task_id = array<i32: 1>} : i64
        %44 = arith.subi %arg26, %43 {async_task_id = array<i32: 1>} : i64
        %45 = arith.trunci %44 {async_task_id = array<i32: 1>} : i64 to i32
        %c1_i64_10 = arith.constant {async_task_id = array<i32: 1>} 1 : i64
        %46 = arith.andi %42, %c1_i64_10 {async_task_id = array<i32: 1>} : i64
        %47 = arith.trunci %46 {async_task_id = array<i32: 1>} : i64 to i1
        %c4_i32_11 = arith.constant {async_task_id = array<i32: 1>} 4 : i32
        %48 = arith.extsi %c4_i32_11 {async_task_id = array<i32: 1>} : i32 to i64
        %49 = arith.divui %arg26, %48 {async_task_id = array<i32: 1>} : i64
        %50 = arith.muli %49, %48 {async_task_id = array<i32: 1>} : i64
        %51 = arith.subi %arg26, %50 {async_task_id = array<i32: 1>} : i64
        %52 = arith.trunci %51 {async_task_id = array<i32: 1>} : i64 to i32
        %c1_i64_12 = arith.constant {async_task_id = array<i32: 1>} 1 : i64
        %53 = arith.andi %49, %c1_i64_12 {async_task_id = array<i32: 1>} : i64
        %54 = arith.trunci %53 {async_task_id = array<i32: 1>} : i64 to i1
        %c4_i32_13 = arith.constant {async_task_id = array<i32: 1>} 4 : i32
        %55 = arith.extsi %c4_i32_13 {async_task_id = array<i32: 1>} : i32 to i64
        %56 = arith.divui %arg26, %55 {async_task_id = array<i32: 1>} : i64
        %57 = arith.muli %56, %55 {async_task_id = array<i32: 1>} : i64
        %58 = arith.subi %arg26, %57 {async_task_id = array<i32: 1>} : i64
        %59 = arith.trunci %58 {async_task_id = array<i32: 1>} : i64 to i32
        %c1_i64_14 = arith.constant {async_task_id = array<i32: 1>} 1 : i64
        %60 = arith.andi %56, %c1_i64_14 {async_task_id = array<i32: 1>} : i64
        %61 = arith.trunci %60 {async_task_id = array<i32: 1>} : i64 to i1
        %c4_i32_15 = arith.constant {async_task_id = array<i32: 1>} 4 : i32
        %62 = arith.extsi %c4_i32_15 {async_task_id = array<i32: 1>} : i32 to i64
        %63 = arith.divui %arg26, %62 {async_task_id = array<i32: 1>} : i64
        %64 = arith.muli %63, %62 {async_task_id = array<i32: 1>} : i64
        %65 = arith.subi %arg26, %64 {async_task_id = array<i32: 1>} : i64
        %66 = arith.trunci %65 {async_task_id = array<i32: 1>} : i64 to i32
        %c1_i64_16 = arith.constant {async_task_id = array<i32: 1>} 1 : i64
        %67 = arith.andi %63, %c1_i64_16 {async_task_id = array<i32: 1>} : i64
        %68 = arith.trunci %67 {async_task_id = array<i32: 1>} : i64 to i1
        %69 = ttg.memdesc_index %arg14[%52] {async_task_id = array<i32: 1>} : !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
        %70 = arith.extsi %54 {async_task_id = array<i32: 1>} : i1 to i32
        ttng.wait_barrier %69, %70 {async_task_id = array<i32: 1>} : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
        %71 = ttg.memdesc_index %arg15[%52] {async_task_id = array<i32: 1>} : !ttg.memdesc<4x128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>
        %72 = ttg.local_load %71 {async_task_id = array<i32: 1>} : !ttg.memdesc<128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory> -> tensor<128x64xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        %73 = ttg.local_alloc %72 {async_task_id = array<i32: 1>} : (tensor<128x64xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>
        %74 = ttg.memdesc_index %arg16[%66] {async_task_id = array<i32: 1>} : !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
        %75 = arith.extsi %68 {async_task_id = array<i32: 1>} : i1 to i32
        ttng.wait_barrier %74, %75 {async_task_id = array<i32: 1>} : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
        %76 = ttg.memdesc_index %arg17[%66] {async_task_id = array<i32: 1>} : !ttg.memdesc<4x128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>
        %77 = ttg.local_load %76 {async_task_id = array<i32: 1>} : !ttg.memdesc<128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory> -> tensor<128x64xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        %78 = ttg.local_alloc %77 {async_task_id = array<i32: 1>} : (tensor<128x64xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>
        %79 = ttg.memdesc_trans %76 {async_task_id = array<i32: 1>, order = array<i32: 1, 0>} : !ttg.memdesc<128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory> -> !ttg.memdesc<64x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>, #ttg.shared_memory>
        %80 = ttng.warp_group_dot %71, %79, %arg25 {async_task_id = array<i32: 1>, inputPrecision = 0 : i32} : !ttg.memdesc<128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory> * !ttg.memdesc<64x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>, #ttg.shared_memory> -> tensor<128x128xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>>
        nvws.consumer_release %arg18, %52 {async_task_id = array<i32: 1>} : tensor<4x!nvws.token>, i32
        nvws.consumer_release %arg19, %66 {async_task_id = array<i32: 1>} : tensor<4x!nvws.token>, i32
        %c1_i64_17 = arith.constant {async_task_id = array<i32: 1>} 1 : i64
        %81 = arith.addi %arg26, %c1_i64_17 {async_task_id = array<i32: 1>} : i64
        scf.yield {async_task_id = array<i32: 1>} %80, %81 : tensor<128x128xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>>, i64
      } {async_task_id = array<i32: 1>}
      %39 = arith.truncf %38#0 {async_task_id = array<i32: 1>} : tensor<128x128xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>> to tensor<128x128xbf16, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>>
      %40 = ttg.convert_layout %39 {async_task_id = array<i32: 1>} : tensor<128x128xbf16, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>> -> tensor<128x128xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
      tt.descriptor_store %23[%36, %37], %40 {async_task_id = array<i32: 1>} : !tt.tensordesc<tensor<128x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>>, tensor<128x128xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
      ttg.warp_return
    }
    partition1(%arg9: i32, %arg10: !tt.ptr<bf16>, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>, %arg15: !ttg.memdesc<4x128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>, %arg16: !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>, %arg17: !ttg.memdesc<4x128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>, %arg18: tensor<4x!nvws.token>, %arg19: tensor<4x!nvws.token>, %arg20: !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>, %arg21: !ttg.memdesc<4x128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>, %arg22: tensor<4x!nvws.token>, %arg23: tensor<4x!nvws.token>) num_warps(4) {
      %c1_i64 = arith.constant {async_task_id = array<i32: 2>} 1 : i64
      %c8_i32 = arith.constant {async_task_id = array<i32: 2>} 8 : i32
      %c256_i32 = arith.constant {async_task_id = array<i32: 2>} 256 : i32
      %c128_i32 = arith.constant {async_task_id = array<i32: 2>} 128 : i32
      %c64_i32 = arith.constant {async_task_id = array<i32: 2>} 64 : i32
      %c0_i32_8 = arith.constant {async_task_id = array<i32: 2>} 0 : i32
      %c1_i32_9 = arith.constant {async_task_id = array<i32: 2>} 1 : i32
      %c127_i32 = arith.constant {async_task_id = array<i32: 2>} 127 : i32
      %c63_i32 = arith.constant {async_task_id = array<i32: 2>} 63 : i32
      %cst = arith.constant {async_task_id = array<i32: 2>} dense<0.000000e+00> : tensor<128x128xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>>
      %22 = arith.extsi %arg9 {async_task_id = array<i32: 2>} : i32 to i64
      %23 = tt.make_tensor_descriptor %arg10, [%arg11, %arg12], [%22, %c1_i64] {async_task_id = array<i32: 2>} : <bf16>, <tensor<128x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>>
      %24 = tt.get_program_id x {async_task_id = array<i32: 2>} : i32
      %25 = arith.addi %arg12, %c127_i32 {async_task_id = array<i32: 2>} : i32
      %26 = arith.divsi %25, %c128_i32 {async_task_id = array<i32: 2>} : i32
      %27 = arith.muli %26, %c8_i32 {async_task_id = array<i32: 2>} : i32
      %28 = arith.divsi %24, %27 {async_task_id = array<i32: 2>} : i32
      %29 = arith.muli %28, %c8_i32 {async_task_id = array<i32: 2>} : i32
      %30 = arith.remsi %24, %c8_i32 {async_task_id = array<i32: 2>} : i32
      %31 = arith.addi %29, %30 {async_task_id = array<i32: 2>} : i32
      %32 = arith.remsi %24, %27 {async_task_id = array<i32: 2>} : i32
      %33 = arith.divsi %32, %c8_i32 {async_task_id = array<i32: 2>} : i32
      %34 = arith.addi %arg13, %c63_i32 {async_task_id = array<i32: 2>} : i32
      %35 = arith.divsi %34, %c64_i32 {async_task_id = array<i32: 2>} : i32
      %36 = arith.muli %31, %c256_i32 {async_task_id = array<i32: 2>} : i32
      %37 = arith.addi %36, %c128_i32 {async_task_id = array<i32: 2>} : i32
      %38 = arith.muli %33, %c128_i32 {async_task_id = array<i32: 2>} : i32
      %c0_i64 = arith.constant {async_task_id = array<i32: 2>} 0 : i64
      %39:2 = scf.for %arg24 = %c0_i32_8 to %35 step %c1_i32_9 iter_args(%arg25 = %cst, %arg26 = %c0_i64) -> (tensor<128x128xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>>, i64)  : i32 {
        %c4_i32 = arith.constant {async_task_id = array<i32: 2>} 4 : i32
        %42 = arith.extsi %c4_i32 {async_task_id = array<i32: 2>} : i32 to i64
        %43 = arith.divui %arg26, %42 {async_task_id = array<i32: 2>} : i64
        %44 = arith.muli %43, %42 {async_task_id = array<i32: 2>} : i64
        %45 = arith.subi %arg26, %44 {async_task_id = array<i32: 2>} : i64
        %46 = arith.trunci %45 {async_task_id = array<i32: 2>} : i64 to i32
        %c1_i64_10 = arith.constant {async_task_id = array<i32: 2>} 1 : i64
        %47 = arith.andi %43, %c1_i64_10 {async_task_id = array<i32: 2>} : i64
        %48 = arith.trunci %47 {async_task_id = array<i32: 2>} : i64 to i1
        %c4_i32_11 = arith.constant {async_task_id = array<i32: 2>} 4 : i32
        %49 = arith.extsi %c4_i32_11 {async_task_id = array<i32: 2>} : i32 to i64
        %50 = arith.divui %arg26, %49 {async_task_id = array<i32: 2>} : i64
        %51 = arith.muli %50, %49 {async_task_id = array<i32: 2>} : i64
        %52 = arith.subi %arg26, %51 {async_task_id = array<i32: 2>} : i64
        %53 = arith.trunci %52 {async_task_id = array<i32: 2>} : i64 to i32
        %c1_i64_12 = arith.constant {async_task_id = array<i32: 2>} 1 : i64
        %54 = arith.andi %50, %c1_i64_12 {async_task_id = array<i32: 2>} : i64
        %55 = arith.trunci %54 {async_task_id = array<i32: 2>} : i64 to i1
        %c4_i32_13 = arith.constant {async_task_id = array<i32: 2>} 4 : i32
        %56 = arith.extsi %c4_i32_13 {async_task_id = array<i32: 2>} : i32 to i64
        %57 = arith.divui %arg26, %56 {async_task_id = array<i32: 2>} : i64
        %58 = arith.muli %57, %56 {async_task_id = array<i32: 2>} : i64
        %59 = arith.subi %arg26, %58 {async_task_id = array<i32: 2>} : i64
        %60 = arith.trunci %59 {async_task_id = array<i32: 2>} : i64 to i32
        %c1_i64_14 = arith.constant {async_task_id = array<i32: 2>} 1 : i64
        %61 = arith.andi %57, %c1_i64_14 {async_task_id = array<i32: 2>} : i64
        %62 = arith.trunci %61 {async_task_id = array<i32: 2>} : i64 to i1
        %c4_i32_15 = arith.constant {async_task_id = array<i32: 2>} 4 : i32
        %63 = arith.extsi %c4_i32_15 {async_task_id = array<i32: 2>} : i32 to i64
        %64 = arith.divui %arg26, %63 {async_task_id = array<i32: 2>} : i64
        %65 = arith.muli %64, %63 {async_task_id = array<i32: 2>} : i64
        %66 = arith.subi %arg26, %65 {async_task_id = array<i32: 2>} : i64
        %67 = arith.trunci %66 {async_task_id = array<i32: 2>} : i64 to i32
        %c1_i64_16 = arith.constant {async_task_id = array<i32: 2>} 1 : i64
        %68 = arith.andi %64, %c1_i64_16 {async_task_id = array<i32: 2>} : i64
        %69 = arith.trunci %68 {async_task_id = array<i32: 2>} : i64 to i1
        %70 = ttg.memdesc_index %arg20[%67] {async_task_id = array<i32: 2>} : !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
        %71 = arith.extsi %69 {async_task_id = array<i32: 2>} : i1 to i32
        ttng.wait_barrier %70, %71 {async_task_id = array<i32: 2>} : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
        %72 = ttg.memdesc_index %arg21[%67] {async_task_id = array<i32: 2>} : !ttg.memdesc<4x128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>
        %73 = ttg.local_load %72 {async_task_id = array<i32: 2>} : !ttg.memdesc<128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory> -> tensor<128x64xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        %74 = ttg.local_alloc %73 {async_task_id = array<i32: 2>} : (tensor<128x64xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>
        %75 = ttg.memdesc_index %arg16[%53] {async_task_id = array<i32: 2>} : !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
        %76 = arith.extsi %55 {async_task_id = array<i32: 2>} : i1 to i32
        ttng.wait_barrier %75, %76 {async_task_id = array<i32: 2>} : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
        %77 = ttg.memdesc_index %arg17[%53] {async_task_id = array<i32: 2>} : !ttg.memdesc<4x128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>
        %78 = ttg.local_load %77 {async_task_id = array<i32: 2>} : !ttg.memdesc<128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory> -> tensor<128x64xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        %79 = ttg.local_alloc %78 {async_task_id = array<i32: 2>} : (tensor<128x64xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>
        %80 = ttg.memdesc_trans %77 {async_task_id = array<i32: 2>, order = array<i32: 1, 0>} : !ttg.memdesc<128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory> -> !ttg.memdesc<64x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>, #ttg.shared_memory>
        %81 = ttng.warp_group_dot %72, %80, %arg25 {async_task_id = array<i32: 2>, inputPrecision = 0 : i32} : !ttg.memdesc<128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory> * !ttg.memdesc<64x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>, #ttg.shared_memory> -> tensor<128x128xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>>
        nvws.consumer_release %arg22, %53 {async_task_id = array<i32: 2>} : tensor<4x!nvws.token>, i32
        nvws.consumer_release %arg23, %67 {async_task_id = array<i32: 2>} : tensor<4x!nvws.token>, i32
        %c1_i64_17 = arith.constant {async_task_id = array<i32: 2>} 1 : i64
        %82 = arith.addi %arg26, %c1_i64_17 {async_task_id = array<i32: 2>} : i64
        scf.yield {async_task_id = array<i32: 2>} %81, %82 : tensor<128x128xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>>, i64
      } {async_task_id = array<i32: 2>}
      %40 = arith.truncf %39#0 {async_task_id = array<i32: 2>} : tensor<128x128xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>> to tensor<128x128xbf16, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>>
      %41 = ttg.convert_layout %40 {async_task_id = array<i32: 2>} : tensor<128x128xbf16, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>> -> tensor<128x128xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
      tt.descriptor_store %23[%37, %38], %41 {async_task_id = array<i32: 2>} : !tt.tensordesc<tensor<128x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>>, tensor<128x128xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
      ttg.warp_return
    } : (i32, !tt.ptr<bf16>, i32, i32, i32, !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>, !ttg.memdesc<4x128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>, !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>, !ttg.memdesc<4x128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>, tensor<4x!nvws.token>, tensor<4x!nvws.token>, !ttg.memdesc<4x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>, !ttg.memdesc<4x128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>, tensor<4x!nvws.token>, tensor<4x!nvws.token>) -> ()
    tt.return
  }
}
