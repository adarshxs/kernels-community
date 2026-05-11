#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("relu_and_probe(Tensor! out, Tensor input, str code, int iters) -> str");
#if defined(CPU_KERNEL)
  ops.impl("relu_and_probe", torch::kCPU, &relu_and_probe);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
