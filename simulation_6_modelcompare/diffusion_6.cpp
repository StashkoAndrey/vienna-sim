#include "diffusion_6.hpp"
#include <vector>

// Simple 2D discrete Laplacian-style diffusion
void diffuse(std::vector<std::vector<float>>& grid, float rate) {
    int N = grid.size();
    if (N == 0) return;
    int M = grid[0].size();

    std::vector<std::vector<float>> original = grid;

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            float self = original[i][j];
            float sum_neighbors = 0.0f;
            int count = 0;

            if (i > 0)        { sum_neighbors += original[i - 1][j]; ++count; }
            if (i < N - 1)    { sum_neighbors += original[i + 1][j]; ++count; }
            if (j > 0)        { sum_neighbors += original[i][j - 1]; ++count; }
            if (j < M - 1)    { sum_neighbors += original[i][j + 1]; ++count; }

            grid[i][j] = self + rate * (sum_neighbors - count * self);
        }
    }
}
