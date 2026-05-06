#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

py::array_t<double> diffuse3d(py::array_t<double> input, double rate) {
    auto buf = input.request();
    auto result = py::array_t<double>(buf.size);
    auto rbuf = result.request();

    auto* in = static_cast<double*>(buf.ptr);
    auto* out = static_cast<double*>(rbuf.ptr);

    auto shape = buf.shape;
    py::ssize_t Z = shape[0];
    py::ssize_t Y = shape[1];
    py::ssize_t X = shape[2];

    auto idx = [X, Y](py::ssize_t z, py::ssize_t y, py::ssize_t x) {
        return z * Y * X + y * X + x;
    };

    for (py::ssize_t z = 0; z < Z; ++z) {
        for (py::ssize_t y = 0; y < Y; ++y) {
            for (py::ssize_t x = 0; x < X; ++x) {
                double center = in[idx(z, y, x)];
                double sum = 0;
                int count = 0;
                for (int dz = -1; dz <= 1; ++dz) {
                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dx = -1; dx <= 1; ++dx) {
                            if (dx || dy || dz) {
                                int nx = x + dx, ny = y + dy, nz = z + dz;
                                if (0 <= nx && nx < X && 0 <= ny && ny < Y && 0 <= nz && nz < Z) {
                                    sum += in[idx(nz, ny, nx)];
                                    count++;
                                }
                            }
                        }
                    }
                }
                double lap = (sum - count * center) / count;
                out[idx(z, y, x)] = center + rate * lap;
            }
        }
    }

    result.resize({Z, Y, X});
    return result;
}

PYBIND11_MODULE(diffusion3d_cpp, m) {
    m.def("diffuse3d", &diffuse3d, "3D diffusion", py::arg("input"), py::arg("rate"));
}