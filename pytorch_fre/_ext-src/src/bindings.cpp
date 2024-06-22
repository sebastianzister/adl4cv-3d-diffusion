#include "interpolate.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("three_nn", &three_nn);
  m.def("three_interpolate", &three_interpolate);
  m.def("three_interpolate_grad", &three_interpolate_grad);
}
