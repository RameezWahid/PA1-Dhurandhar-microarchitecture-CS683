// conv_optimized.cpp  STAGE 5: PUT IT ALL TOGETHER
// Combine tiling + SIMD (AVX2) + reasonable loop ordering to maximize locality and
// throughput. This implementation tiles the output, and for each tile computes
// outputs 8 columns at a time using AVX2 _mm256_fmadd_ps.

#include <immintrin.h>
#include <cstring>

#include "convolution.h"

// Set to 0 to benchmark the preserved generic no-ACC path for K=3.
#ifndef CONV_USE_ACC_K3
#define CONV_USE_ACC_K3 1
#endif

// Computes eight adjacent K=3 outputs.  Two independent accumulators shorten
// the FMA dependency chain while keeping the source compact.
static inline __m256 conv3_acc_vector(const float* r0, int stride,
                                      __m256 w0, __m256 w1, __m256 w2,
                                      __m256 w3, __m256 w4, __m256 w5,
                                      __m256 w6, __m256 w7, __m256 w8) {
    const float* r1 = r0 + stride;
    const float* r2 = r1 + stride;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(r0),     w0, acc0);
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + 1), w1, acc0);
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + 2), w2, acc0);
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(r1),     w3, acc0);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + 1), w4, acc1);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + 2), w5, acc1);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(r2),     w6, acc1);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + 1), w7, acc1);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + 2), w8, acc1);
    return _mm256_add_ps(acc0, acc1);
}

void conv_optimized(const float* in, float* out, const float* ker,
                    int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

#if CONV_USE_ACC_K3
    // K=3 ACC fast path 

    if (K == 3) {
        const __m256 w0 = _mm256_set1_ps(ker[0]);
        const __m256 w1 = _mm256_set1_ps(ker[1]);
        const __m256 w2 = _mm256_set1_ps(ker[2]);
        const __m256 w3 = _mm256_set1_ps(ker[3]);
        const __m256 w4 = _mm256_set1_ps(ker[4]);
        const __m256 w5 = _mm256_set1_ps(ker[5]);
        const __m256 w6 = _mm256_set1_ps(ker[6]);
        const __m256 w7 = _mm256_set1_ps(ker[7]);
        const __m256 w8 = _mm256_set1_ps(ker[8]);
        int oy = 0;
        for (; oy + 3 < H; oy += 4) {
            for (int ox = 0; ox < W; ox += 8) {
                const float* r0 = in + oy * in_stride + ox;
                const __m256 a0 = conv3_acc_vector(r0, in_stride, w0, w1, w2, w3, w4, w5, w6, w7, w8);
                const __m256 a1 = conv3_acc_vector(r0 + in_stride, in_stride, w0, w1, w2, w3, w4, w5, w6, w7, w8);
                const __m256 a2 = conv3_acc_vector(r0 + 2 * in_stride, in_stride, w0, w1, w2, w3, w4, w5, w6, w7, w8);
                const __m256 a3 = conv3_acc_vector(r0 + 3 * in_stride, in_stride, w0, w1, w2, w3, w4, w5, w6, w7, w8);
                _mm256_storeu_ps(out + oy * W + ox, a0);
                _mm256_storeu_ps(out + (oy + 1) * W + ox, a1);
                _mm256_storeu_ps(out + (oy + 2) * W + ox, a2);
                _mm256_storeu_ps(out + (oy + 3) * W + ox, a3);
            }
        }
        for (; oy < H; ++oy) {
            for (int ox = 0; ox < W; ox += 8) {
                const __m256 a = conv3_acc_vector(in + oy * in_stride + ox, in_stride,
                                                   w0, w1, w2, w3, w4, w5, w6, w7, w8);
                _mm256_storeu_ps(out + oy * W + ox, a);
            }
        }
        return;
    }
#endif

    // Generic no-ACC strip-tiled AVX2 path 

    const int tile_h = 64;

    for (int oy0 = 0; oy0 < H; oy0 += tile_h) {
        const int y_end = oy0 + tile_h < H ? oy0 + tile_h : H;
        const int tile_rows = y_end - oy0;
        int oy = 0;
        // Process four output rows at a time to increase register reuse and amortize weight loads
        for (; oy + 3 < tile_rows; oy += 4) {
            int ox = 0;
            for (; ox + 7 < W; ox += 8) {
                __m256 vacc0 = _mm256_setzero_ps();
                __m256 vacc1 = _mm256_setzero_ps();
                __m256 vacc2 = _mm256_setzero_ps();
                __m256 vacc3 = _mm256_setzero_ps();
                int wi = 0;
                for (int ky = 0; ky < K; ++ky) {
                    const float* in_ptr0 = in + (oy0 + oy + ky) * in_stride + ox;
                    const float* in_ptr1 = in_ptr0 + in_stride;
                    const float* in_ptr2 = in_ptr1 + in_stride;
                    const float* in_ptr3 = in_ptr2 + in_stride;
                    for (int kx = 0; kx < K; ++kx) {
                        __m256 vin0 = _mm256_loadu_ps(in_ptr0 + kx);
                        __m256 vin1 = _mm256_loadu_ps(in_ptr1 + kx);
                        __m256 vin2 = _mm256_loadu_ps(in_ptr2 + kx);
                        __m256 vin3 = _mm256_loadu_ps(in_ptr3 + kx);
                        __m256 vw = _mm256_set1_ps(ker[wi]);
                        vacc0 = _mm256_fmadd_ps(vin0, vw, vacc0);
                        vacc1 = _mm256_fmadd_ps(vin1, vw, vacc1);
                        vacc2 = _mm256_fmadd_ps(vin2, vw, vacc2);
                        vacc3 = _mm256_fmadd_ps(vin3, vw, vacc3);
                        ++wi;
                    }
                }
                _mm256_storeu_ps(out + (oy0 + oy) * W + ox, vacc0);
                _mm256_storeu_ps(out + (oy0 + oy + 1) * W + ox, vacc1);
                _mm256_storeu_ps(out + (oy0 + oy + 2) * W + ox, vacc2);
                _mm256_storeu_ps(out + (oy0 + oy + 3) * W + ox, vacc3);
            }
            // scalar tail for the four rows
            for (; ox < W; ++ox) {
                float acc0 = 0.0f;
                float acc1 = 0.0f;
                float acc2 = 0.0f;
                float acc3 = 0.0f;
                for (int ky = 0; ky < K; ++ky) {
                    const int in_y0 = (oy0 + oy + ky) * in_stride + ox;
                    const int in_y1 = (oy0 + oy + 1 + ky) * in_stride + ox;
                    const int in_y2 = (oy0 + oy + 2 + ky) * in_stride + ox;
                    const int in_y3 = (oy0 + oy + 3 + ky) * in_stride + ox;
                    const float* row0 = in + in_y0;
                    const float* row1 = in + in_y1;
                    const float* row2 = in + in_y2;
                    const float* row3 = in + in_y3;
                    const float* k_ptr = ker + ky * K;
                    for (int kx = 0; kx < K; ++kx) {
                        acc0 += row0[kx] * k_ptr[kx];
                        acc1 += row1[kx] * k_ptr[kx];
                        acc2 += row2[kx] * k_ptr[kx];
                        acc3 += row3[kx] * k_ptr[kx];
                    }
                }
                out[(oy0 + oy) * W + ox] = acc0;
                out[(oy0 + oy + 1) * W + ox] = acc1;
                out[(oy0 + oy + 2) * W + ox] = acc2;
                out[(oy0 + oy + 3) * W + ox] = acc3;
            }
        }
        // Handle last row if tile_rows is odd
        for (; oy < tile_rows; ++oy) {
            int ox = 0;
            for (; ox + 7 < W; ox += 8) {
                __m256 vacc = _mm256_setzero_ps();
                int wi = 0;
                for (int ky = 0; ky < K; ++ky) {
                    const float* in_ptr = in + (oy0 + oy + ky) * in_stride + ox;
                    for (int kx = 0; kx < K; ++kx) {
                        __m256 vin = _mm256_loadu_ps(in_ptr + kx);
                        vacc = _mm256_fmadd_ps(vin, _mm256_set1_ps(ker[wi]), vacc);
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
