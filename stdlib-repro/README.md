# stdlib-repro

Tiny Kernel Hub repro for the packaged-runtime segfaults seen in DeepGEMM's JIT path.

The op is intentionally boring:

- run ReLU on a CPU tensor
- parse `#include` lines with `std::regex`
- format a hash with `std::ostringstream`, `std::hex`, `std::setw`, and `std::setfill`

Build/upload it through `kernel-builder`, then load it with `kernels.get_kernel(...)` in the same environment that used the DeepGEMM artifact.
