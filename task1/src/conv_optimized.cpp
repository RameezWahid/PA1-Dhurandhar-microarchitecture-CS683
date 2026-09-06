// conv_optimized.cpp  STAGE 5: PUT IT ALL TOGETHER
// Combine tiling + SIMD (AVX2) + reasonable loop ordering to maximize locality and
// throughput. This implementation tiles the output, and for each tile computes
// outputs 8 columns at a time using AVX2 _mm256_fmadd_ps.

#include <immintrin.h>
#include <cstring>

#include "convolution.h"

void conv_optimized(const float* in, float* out, const float* ker,
                    int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    const int tile_h = 128;
    const int tile_w = W;  // full width

    // Precompute vectorized weights to avoid repeated _mm256_set1_ps
    const int KK = K * K;
    __m256 vweights[25]; // supports up to K=5
    for (int i = 0; i < KK; ++i) {
        vweights[i] = _mm256_set1_ps(ker[i]);
    }

    for (int oy0 = 0; oy0 < H; oy0 += tile_h) {
        const int y_end = oy0 + tile_h < H ? oy0 + tile_h : H;
        const int tile_rows = y_end - oy0;
        int oy = 0;
        // Process two output rows at a time to increase register reuse and amortize weight loads
        for (; oy + 1 < tile_rows; oy += 2) {
            int ox = 0;
            for (; ox + 7 < W; ox += 8) {
                __m256 vacc0 = _mm256_setzero_ps();
                __m256 vacc1 = _mm256_setzero_ps();
                int wi = 0;
                for (int ky = 0; ky < K; ++ky) {
                    for (int kx = 0; kx < K; ++kx) {
                        const float* in_ptr0 = in + (oy0 + oy + ky) * in_stride + (ox + kx);
                        const float* in_ptr1 = in + (oy0 + oy + 1 + ky) * in_stride + (ox + kx);
                        __m256 vin0 = _mm256_loadu_ps(in_ptr0);
                        __m256 vin1 = _mm256_loadu_ps(in_ptr1);
                        __m256 vw = vweights[wi];
                        vacc0 = _mm256_fmadd_ps(vin0, vw, vacc0);
                        vacc1 = _mm256_fmadd_ps(vin1, vw, vacc1);
                        ++wi;
                    }
                }
                _mm256_storeu_ps(out + (oy0 + oy) * W + ox, vacc0);
                _mm256_storeu_ps(out + (oy0 + oy + 1) * W + ox, vacc1);
            }
            // scalar tail for the two rows
            for (; ox < W; ++ox) {
                float acc0 = 0.0f;
                float acc1 = 0.0f;
                for (int ky = 0; ky < K; ++ky) {
                    const int in_y0 = (oy0 + oy + ky) * in_stride + ox;
                    const int in_y1 = (oy0 + oy + 1 + ky) * in_stride + ox;
                    const float* row0 = in + in_y0;
                    const float* row1 = in + in_y1;
                    const float* k_ptr = ker + ky * K;
                    for (int kx = 0; kx < K; ++kx) {
                        acc0 += row0[kx] * k_ptr[kx];
                        acc1 += row1[kx] * k_ptr[kx];
                    }
                }
                out[(oy0 + oy) * W + ox] = acc0;
                out[(oy0 + oy + 1) * W + ox] = acc1;
            }
        }
        // Handle last row if tile_rows is odd
        for (; oy < tile_rows; ++oy) {
            int ox = 0;
            for (; ox + 7 < W; ox += 8) {
                __m256 vacc = _mm256_setzero_ps();
                int wi = 0;
                for (int ky = 0; ky < K; ++ky) {
                    for (int kx = 0; kx < K; ++kx) {
                        const float* in_ptr = in + (oy0 + oy + ky) * in_stride + (ox + kx);
                        __m256 vin = _mm256_loadu_ps(in_ptr);
                        vacc = _mm256_fmadd_ps(vin, vweights[wi], vacc);
                        ++wi;
                    }
                }
                _mm256_storeu_ps(out + (oy0 + oy) * W + ox, vacc);
            }
            for (; ox < W; ++ox) {
                float acc = 0.0f;
                for (int ky = 0; ky < K; ++ky) {
                    const int in_y = (oy0 + oy + ky) * in_stride + ox;
                    const float* row_ptr = in + in_y;
                    const float* k_ptr = ker + ky * K;
                    for (int kx = 0; kx < K; ++kx) {
                        acc += row_ptr[kx] * k_ptr[kx];
                    }
                }
                out[(oy0 + oy) * W + ox] = acc;
            }
        }
    }
}
