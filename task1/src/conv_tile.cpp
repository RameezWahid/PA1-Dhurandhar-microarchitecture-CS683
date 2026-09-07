// conv_tile.cpp  STAGE 3: REORDERING + UNROLLING + STRIP TILING

#include "convolution.h"

void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;
    const int tile_h = 64;

    for (int oy0 = 0; oy0 < H; oy0 += tile_h) {
        const int y_end = oy0 + tile_h < H ? oy0 + tile_h : H;

        for (int oy = oy0; oy < y_end; ++oy)
            for (int ox = 0; ox < W; ++ox) out[oy * W + ox] = 0.0f;

        for (int ky = 0; ky < K; ++ky) {
            for (int kx = 0; kx < K; ++kx) {
                const float weight = ker[ky * K + kx];
                for (int oy = oy0; oy < y_end; ++oy) {
                    const float* in_row = in + (oy + ky) * in_stride + kx;
                    float* out_row = out + oy * W;
                    int ox = 0;
                    for (; ox + 3 < W; ox += 4) {
                        float acc0 = out_row[ox + 0];
                        float acc1 = out_row[ox + 1];
                        float acc2 = out_row[ox + 2];
                        float acc3 = out_row[ox + 3];
                        acc0 += in_row[ox + 0] * weight;
                        acc1 += in_row[ox + 1] * weight;
                        acc2 += in_row[ox + 2] * weight;
                        acc3 += in_row[ox + 3] * weight;
                        out_row[ox + 0] = acc0;
                        out_row[ox + 1] = acc1;
                        out_row[ox + 2] = acc2;
                        out_row[ox + 3] = acc3;
                    }
                    for (; ox < W; ++ox) out_row[ox] += in_row[ox] * weight;
                }
            }
        }
    }
}
