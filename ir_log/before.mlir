module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel_tma(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c1_i64 = arith.constant 1 : i64
    %c8_i32 = arith.constant 8 : i32
    %c256_i32 = arith.constant 256 : i32
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c127_i32 = arith.constant 127 : i32
    %c63_i32 = arith.constant 63 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>>
    %0 = arith.extsi %arg6 : i32 to i64
    %1 = tt.make_tensor_descriptor %arg0, [%arg3, %arg5], [%0, %c1_i64] : <bf16>, <tensor<256x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>>
    %2 = arith.extsi %arg7 : i32 to i64
    %3 = tt.make_tensor_descriptor %arg1, [%arg4, %arg5], [%2, %c1_i64] : <bf16>, <tensor<128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>>
    %4 = arith.extsi %arg8 : i32 to i64
    %5 = tt.make_tensor_descriptor %arg2, [%arg3, %arg4], [%4, %c1_i64] : <bf16>, <tensor<256x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>>
    %6 = tt.get_program_id x : i32
    %7 = arith.addi %arg4, %c127_i32 : i32
    %8 = arith.divsi %7, %c128_i32 : i32
    %9 = arith.muli %8, %c8_i32 : i32
    %10 = arith.divsi %6, %9 : i32
    %11 = arith.muli %10, %c8_i32 : i32
    %12 = arith.remsi %6, %c8_i32 : i32
    %13 = arith.addi %11, %12 : i32
    %14 = arith.remsi %6, %9 : i32
    %15 = arith.divsi %14, %c8_i32 : i32
    %16 = arith.addi %arg5, %c63_i32 : i32
    %17 = arith.divsi %16, %c64_i32 : i32
    %18 = arith.muli %13, %c256_i32 : i32
    %19 = arith.muli %15, %c128_i32 : i32
    %20 = scf.for %arg9 = %c0_i32 to %17 step %c1_i32 iter_args(%arg10 = %cst) -> (tensor<256x128xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>>)  : i32 {
      %23 = arith.muli %arg9, %c64_i32 : i32
      %24 = tt.descriptor_load %1[%18, %23] : !tt.tensordesc<tensor<256x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>> -> tensor<256x64xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %25 = ttg.local_alloc %24 : (tensor<256x64xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<256x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>
      %26 = tt.descriptor_load %3[%19, %23] : !tt.tensordesc<tensor<128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>> -> tensor<128x64xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %27 = ttg.local_alloc %26 : (tensor<128x64xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>
      %28 = ttg.memdesc_trans %27 {order = array<i32: 1, 0>} : !ttg.memdesc<128x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory> -> !ttg.memdesc<64x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>, #ttg.shared_memory>
      %29 = ttng.warp_group_dot %25, %28, %arg10 {inputPrecision = 0 : i32} : !ttg.memdesc<256x64xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory> * !ttg.memdesc<64x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>, #ttg.shared_memory> -> tensor<256x128xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>>
      scf.yield %29 : tensor<256x128xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>>
    } {tt.warp_specialize}
    %21 = arith.truncf %20 : tensor<256x128xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>> to tensor<256x128xbf16, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>>
    %22 = ttg.convert_layout %21 : tensor<256x128xbf16, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>> -> tensor<256x128xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    tt.descriptor_store %5[%18, %19], %22 : !tt.tensordesc<tensor<256x128xbf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>>, tensor<256x128xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    tt.return
  }
}
