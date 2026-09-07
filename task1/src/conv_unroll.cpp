// conv_unroll.cpp  STAGE 2: REORDERING + LOOP UNROLLING
// Keep the reordered ky, kx, oy, ox traversal, then unroll the unit-stride
// output-column loop by four elements.

#include "convolution.h"

void conv_unroll(const float* in, float* out, const float* ker,
                 int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int i = 0; i < H * W; ++i) out[i] = 0.0f;

    for (int ky = 0; ky < K; ++ky) {
        for (int kx = 0; kx < K; ++kx) {
            const float weight = ker[ky * K + kx];
            for (int oy = 0; oy < H; ++oy) {
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
