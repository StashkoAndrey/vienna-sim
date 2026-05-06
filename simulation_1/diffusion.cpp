#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


namespace py = pybind11;

py::array_t<float> diffuse(py::array_t<float> input, float rate) {
    auto buf = input.request();
    auto result = py::array_t<float>(buf.size);
    auto rbuf = result.request();

    float* ptr = static_cast<float*>(buf.ptr);
    float* rptr = static_cast<float*>(rbuf.ptr);

    py::ssize_t rows = buf.shape[0];
    py::ssize_t cols = buf.shape[1];

    auto index = [cols](py::ssize_t i, py::ssize_t j) { return i * cols + j; };

    for (py::ssize_t i = 0; i < rows; ++i) {
        for (py::ssize_t j = 0; j < cols; ++j) {
            float C = ptr[index(i, j)];
            float N = (i > 0) ? ptr[index(i-1, j)] : C;
            float S = (i < rows - 1) ? ptr[index(i+1, j)] : C;
            float W = (j > 0) ? ptr[index(i, j-1)] : C;
            float E = (j < cols - 1) ? ptr[index(i, j+1)] : C;

            float laplacian = N + S + W + E - 4 * C;
            rptr[index(i, j)] = C + rate * laplacian;
        }
    }
    result.resize({rows, cols});
    return result;
}

PYBIND11_MODULE(diffusion_cpp, m) {
    m.def("diffuse", &diffuse, "Diffuse nutrients across a 2D grid");
}