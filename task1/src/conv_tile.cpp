#include "convolution.h"

void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    const int tile_h = 64; // horizontal strip height

    for (int oy0 = 0; oy0 < H; oy0 += tile_h) {
        const int y_end = oy0 + tile_h < H ? oy0 + tile_h : H;
        const int tile_rows = y_end - oy0;

        // Zero the output rows in this horizontal strip
        for (int oy = 0; oy < tile_rows; ++oy) {
            float* out_row = out + (oy0 + oy) * W;
            for (int ox = 0; ox < W; ++ox) out_row[ox] = 0.0f;
        }

        // Accumulate directly into out for each kernel tap (no temp buffer)
        for (int ky = 0; ky < K; ++ky) {
            for (int kx = 0; kx < K; ++kx) {
                const float weight = ker[ky * K + kx];
                for (int oy = 0; oy < tile_rows; ++oy) {
                    const float* in_row = in + (oy0 + oy + ky) * in_stride + kx;
                    float* out_row = out + (oy0 + oy) * W;
                    for (int ox = 0; ox < W; ++ox) {
                        out_row[ox] += in_row[ox] * weight;
                    }
                }
            }
        }
    }
}