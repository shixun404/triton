#include "Utility.h"
#include "mlir/Pass/Pass.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir {

// Implemented in WSLowerToken.cpp.
void doTokenLowering(triton::FuncOp &funcOp, unsigned numConsumerGroups);

#define GEN_PASS_DEF_NVGPUTESTWSTOKENLOWERING
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

class NVGPUTestWSTokenLoweringPass
    : public impl::NVGPUTestWSTokenLoweringBase<NVGPUTestWSTokenLoweringPass> {
public:
  using impl::NVGPUTestWSTokenLoweringBase<
      NVGPUTestWSTokenLoweringPass>::NVGPUTestWSTokenLoweringBase;

  void runOnOperation() override {
    getOperation()->walk([&](triton::FuncOp funcOp) {
      doTokenLowering(funcOp, numConsumerGroups);
    });
  }
};

} // namespace mlir


