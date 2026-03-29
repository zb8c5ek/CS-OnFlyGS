#ifdef _WIN32
#  include <windows.h>
#  undef small
#  undef min
#  undef max
#  undef near
#  undef far
#endif

#include <torch/extension.h>
#include "ssim.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fusedssim", &fusedssim);
  m.def("fusedssim_backward", &fusedssim_backward);
}
