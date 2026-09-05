

#include "convolution.h"

#include <vector>

void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    const int tile_h = 64;
    const int tile_w = 64;

    for (int oy0 = 0; oy0 < H; oy0 += tile_h) {
        const int y_end = oy0 + tile_h < H ? oy0 + tile_h : H;
        for (int ox0 = 0; ox0 < W; ox0 += tile_w) {
            const int x_end = ox0 + tile_w < W ? ox0 + tile_w : W;
            const int tile_rows = y_end - oy0;
            const int tile_cols = x_end - ox0;

            std::vector<float> tile(tile_rows * tile_cols, 0.0f);

            for (int ky = 0; ky < K; ++ky) {
               for (int kx = 0; kx < K; ++kx) {
                   const float weight = ker[ky * K + kx];
                   for (int oy = 0; oy < tile_rows; ++oy) {
                       const int in_y = (oy0 + oy + ky) * in_stride + (ox0 + kx);
                       const float* in_row = in + in_y;
                       float* tile_row = tile.data() + oy * tile_cols;
                       for (int ox = 0; ox < tile_cols; ++ox) {
                           tile_row[ox] += in_row[ox] * weight;
                       }
                   }
               }
            }

            for (int oy = 0; oy < tile_rows; ++oy) {
               for (int ox = 0; ox < tile_cols; ++ox) {
                   out[(oy0 + oy) * W + (ox0 + ox)] = tile[oy * tile_cols + ox];
               }
            }
        }
    }
}
