#include "convolution.h"

void conv_unroll(const float* in, float* out, const float* ker,
                 int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int oy = 0; oy < H; ++oy) {
        for (int ox = 0; ox < W; ++ox) {
            float acc = 0.0f;

            for (int ky = 0; ky < K; ++ky) {
                const int in_y = (oy + ky) * in_stride + ox;
                int kx = 0;

                for (; kx + 3 < K; kx += 4) {
                    acc += in[in_y + kx + 0] * ker[ky * K + kx + 0];
                    acc += in[in_y + kx + 1] * ker[ky * K + kx + 1];
                    acc += in[in_y + kx + 2] * ker[ky * K + kx + 2];
                    acc += in[in_y + kx + 3] * ker[ky * K + kx + 3];
                }

                for (; kx < K; ++kx) {
                    acc += in[in_y + kx] * ker[ky * K + kx];
                }
            }

            out[oy * W + ox] = acc;
        }
    }
}