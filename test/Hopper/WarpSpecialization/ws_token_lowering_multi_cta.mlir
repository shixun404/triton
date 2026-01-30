// RUN: triton-opt %s -split-input-file --nvgpu-test-ws-token-lowering=num-consumer-groups=1 | FileCheck %s

// CHECK-LABEL: @token_lowering_multi_cta
// CHECK: ttg.local_alloc
// CHECK: ttng.init_barrier
// CHECK: ttng.init_barrier
// CHECK: gpu.barrier
// CHECK: ttng.cluster_arrive
// CHECK: ttng.cluster_wait
// CHECK-NOT: nvws.create_token
// CHECK-NOT: nvws.producer_acquire
// CHECK-NOT: nvws.producer_commit

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @token_lowering_multi_cta() {
    %tok = nvws.create_token {loadType = 1 : i32, numBuffers = 2 : i32} : tensor<2x!nvws.token>
    %c0 = arith.constant 0 : i32
    %false = arith.constant false

    // Use the token via explicit capture in a partition region, mimicking warp_specialize=true.
    ttg.warp_specialize(%tok) attributes {warpGroupStartIds = array<i32: 4>} {
      default {
        ttg.warp_yield
      }
      partition0(%t : tensor<2x!nvws.token>) num_warps(4) {
        nvws.producer_acquire %t, %c0, %false {async_task_id = array<i32: 0>} : tensor<2x!nvws.token>, i32, i1
        nvws.producer_commit %t, %c0 {async_task_id = array<i32: 0>} : tensor<2x!nvws.token>, i32
        ttg.warp_return
      }
    } : (tensor<2x!nvws.token>) -> ()

    tt.return
  }
}


