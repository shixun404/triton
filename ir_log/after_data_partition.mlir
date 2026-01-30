module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel_tma(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c1_i64 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 1 : i64
    %c8_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 8 : i32
    %c256_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 256 : i32
    %c128_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 128 : i32
    %c64_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 64 : i32
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
    %c1_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 1 : i32
    %c127_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 127 : i32
    %c63_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 63 : i32
    %cst = arith.constant {async_task_id = array<i32: 1, 2>} dense<0.000000e+00> : tensor<128x128xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>>
    %0 = arith.extsi %arg6 {async_task_id = array<i32: 0>} : i32 to i64
    %1 = tt.make_tensor_descriptor %arg0, [%arg3, %arg5], [%0, %c1_i64] {async_task_id = array<i32: 0>} : <bf16>, <tensor<128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>>
    %2 = tt.make_tensor_descriptor %arg0, [%arg3, %arg5], [%0, %c1_i64] {async_task_id = array<i32: 0>} : <bf16>, <tensor<128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>>
    %3 = arith.extsi %arg7 {async_task_id = array<i32: 0>} : i32 to i64
    %4 = tt.make_tensor_descriptor %arg1, [%arg4, %arg5], [%3, %c1_i64] {async_task_id = array<i32: 0>} : <bf16>, <tensor<128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>>
    %5 = arith.extsi %arg8 {async_task_id = array<i32: 1, 2>} : i32 to i64
    %6 = tt.make_tensor_descriptor %arg2, [%arg3, %arg4], [%5, %c1_i64] {async_task_id = array<i32: 1>} : <bf16>, <tensor<128x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>>
    %7 = tt.make_tensor_descriptor %arg2, [%arg3, %arg4], [%5, %c1_i64] {async_task_id = array<i32: 2>} : <bf16>, <tensor<128x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>>
    %8 = tt.get_program_id x {async_task_id = array<i32: 0, 1, 2>} : i32
    %9 = arith.addi %arg4, %c127_i32 {async_task_id = array<i32: 0, 1, 2>} : i32
    %10 = arith.divsi %9, %c128_i32 {async_task_id = array<i32: 0, 1, 2>} : i32
    %11 = arith.muli %10, %c8_i32 {async_task_id = array<i32: 0, 1, 2>} : i32
    %12 = arith.divsi %8, %11 {async_task_id = array<i32: 0, 1, 2>} : i32
    %13 = arith.muli %12, %c8_i32 {async_task_id = array<i32: 0, 1, 2>} : i32
    %14 = arith.remsi %8, %c8_i32 {async_task_id = array<i32: 0, 1, 2>} : i32
    %15 = arith.addi %13, %14 {async_task_id = array<i32: 0, 1, 2>} : i32
    %16 = arith.remsi %8, %11 {async_task_id = array<i32: 0, 1, 2>} : i32
    %17 = arith.divsi %16, %c8_i32 {async_task_id = array<i32: 0, 1, 2>} : i32
    %18 = arith.addi %arg5, %c63_i32 {async_task_id = array<i32: 0, 1, 2>} : i32
    %19 = arith.divsi %18, %c64_i32 {async_task_id = array<i32: 0, 1, 2>} : i32
    %20 = arith.muli %15, %c256_i32 {async_task_id = array<i32: 0, 1, 2>} : i32
    %21 = arith.addi %20, %c128_i32 {async_task_id = array<i32: 2>} : i32
    %22 = arith.addi %20, %c128_i32 {async_task_id = array<i32: 0>} : i32
    %23 = arith.muli %17, %c128_i32 {async_task_id = array<i32: 0, 1, 2>} : i32
    %24:2 = scf.for %arg9 = %c0_i32 to %19 step %c1_i32 iter_args(%arg10 = %cst, %arg11 = %cst) -> (tensor<128x128xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>>, tensor<128x128xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>>)  : i32 {
      %29 = arith.muli %arg9, %c64_i32 {async_task_id = array<i32: 0>} : i32
      %30 = tt.descriptor_load %1[%20, %29] {async_task_id = array<i32: 0>} : !tt.tensordesc<tensor<128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>> -> tensor<128x64xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %31 = tt.descriptor_load %2[%22, %29] {async_task_id = array<i32: 0>} : !tt.tensordesc<tensor<128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>> -> tensor<128x64xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %32 = ttg.local_alloc %30 {async_task_id = array<i32: 1>} : (tensor<128x64xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>
      %33 = ttg.local_alloc %31 {async_task_id = array<i32: 2>} : (tensor<128x64xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>
      %34 = tt.descriptor_load %4[%23, %29] {async_task_id = array<i32: 0>} : !tt.tensordesc<tensor<128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>> -> tensor<128x64xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %35 = ttg.local_alloc %34 {async_task_id = array<i32: 1, 2>} : (tensor<128x64xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>
      %36 = ttg.memdesc_trans %35 {async_task_id = array<i32: 1, 2>, order = array<i32: 1, 0>} : !ttg.memdesc<128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory> -> !ttg.memdesc<64x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>, #ttg.shared_memory>
      %37 = ttng.warp_group_dot %32, %36, %arg10 {async_task_id = array<i32: 1>, inputPrecision = 0 : i32} : !ttg.memdesc<128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory> * !ttg.memdesc<64x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>, #ttg.shared_memory> -> tensor<128x128xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>>
      %38 = ttng.warp_group_dot %33, %36, %arg11 {async_task_id = array<i32: 2>, inputPrecision = 0 : i32} : !ttg.memdesc<128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory> * !ttg.memdesc<64x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>, #ttg.shared_memory> -> tensor<128x128xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>>
      scf.yield {async_task_id = array<i32: 1, 2>} %37, %38 : tensor<128x128xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>>, tensor<128x128xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>>
    } {async_task_id = array<i32: 0, 1, 2>, tt.warp_specialize}
    %25 = arith.truncf %24#0 {async_task_id = array<i32: 1>} : tensor<128x128xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>> to tensor<128x128xbf16, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>>
    %26 = arith.truncf %24#1 {async_task_id = array<i32: 2>} : tensor<128x128xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>> to tensor<128x128xbf16, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>>
    %27 = ttg.convert_layout %25 {async_task_id = array<i32: 1>} : tensor<128x128xbf16, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>> -> tensor<128x128xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %28 = ttg.convert_layout %26 {async_task_id = array<i32: 2>} : tensor<128x128xbf16, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>> -> tensor<128x128xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    tt.descriptor_store %6[%20, %23], %27 {async_task_id = array<i32: 1>} : !tt.tensordesc<tensor<128x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>>, tensor<128x128xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    tt.descriptor_store %7[%21, %23], %28 {async_task_id = array<i32: 2>} : !tt.tensordesc<tensor<128x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>>, tensor<128x128xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    tt.return
  }
}