// matmul_optimized.cpp  STAGE 3: PUT IT ALL TOGETHER

#include <immintrin.h>
#include <algorithm>
#include "matmul.h"

static inline float sum_vector(__m256 vec) {
    __m128 low = _mm256_castps256_ps128(vec);
    __m128 high = _mm256_extractf128_ps(vec, 1);
    __m128 sum = _mm_add_ps(low, high);
    __m128 shuf = _mm_movehdup_ps(sum);
    __m128 sum2 = _mm_add_ps(sum, shuf);
    shuf = _mm_movehl_ps(shuf, sum2);
    __m128 sum1 = _mm_add_ss(sum2, shuf);
    return _mm_cvtss_f32(sum1);
}


static inline void dot2x2(const float* a0, const float* a1,
                          const float* b0, const float* b1,
                          int kk, int k_max,
                          float& s00, float& s01, float& s10, float& s11) {
    __m256 acc00 = _mm256_setzero_ps();
    __m256 acc01 = _mm256_setzero_ps();
    __m256 acc10 = _mm256_setzero_ps();
    __m256 acc11 = _mm256_setzero_ps();

    int k = kk;
    for (; k + 8 <= k_max; k += 8) {
        _mm_prefetch(reinterpret_cast<const char*>(a0 + k + 64), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(a1 + k + 64), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(b0 + k + 64), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(b1 + k + 64), _MM_HINT_T0);

        __m256 va0 = _mm256_loadu_ps(a0 + k);
        __m256 va1 = _mm256_loadu_ps(a1 + k);
        __m256 vb0 = _mm256_loadu_ps(b0 + k);
        __m256 vb1 = _mm256_loadu_ps(b1 + k);

        // 4 loads feed 4 FMAs: each loaded vector is reused once.
        acc00 = _mm256_fmadd_ps(va0, vb0, acc00);
        acc01 = _mm256_fmadd_ps(va0, vb1, acc01);
        acc10 = _mm256_fmadd_ps(va1, vb0, acc10);
        acc11 = _mm256_fmadd_ps(va1, vb1, acc11);
    }

    s00 += sum_vector(acc00);
    s01 += sum_vector(acc01);
    s10 += sum_vector(acc10);
    s11 += sum_vector(acc11);

    for (; k < k_max; k++) {
        s00 += a0[k] * b0[k];
        s01 += a0[k] * b1[k];
        s10 += a1[k] * b0[k];
        s11 += a1[k] * b1[k];
    }
}

void matmul_optimized(const float* A, const float* B, float* C,
                      int M, int N, int K, int lda, int ldb, int ldc) {

    const int block_size = 64; 

    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            C[i * ldc + j] = 0.0f;

    for (int ii = 0; ii < M; ii += block_size) {
        int i_max = std::min(ii + block_size, M);

        for (int jj = 0; jj < N; jj += block_size) {
            int j_max = std::min(jj + block_size, N);

            for (int kk = 0; kk < K; kk += block_size) {
                int k_max = std::min(kk + block_size, K);

                int i = ii;
                for (; i + 2 <= i_max; i += 2) {
                    const float* a0 = A + i * lda;
                    const float* a1 = A + (i + 1) * lda;

                    int j = jj;
                    for (; j + 2 <= j_max; j += 2) {
                        const float* b0 = B + j * ldb;
                        const float* b1 = B + (j + 1) * ldb;

                        float s00 = 0.f, s01 = 0.f, s10 = 0.f, s11 = 0.f;
                        dot2x2(a0, a1, b0, b1, kk, k_max, s00, s01, s10, s11);

                        C[i * ldc + j]           += s00;
                        C[i * ldc + (j + 1)]     += s01;
                        C[(i + 1) * ldc + j]     += s10;
                        C[(i + 1) * ldc + (j+1)] += s11;
                    }
                    // leftover single column in this row-pair (N % 2 != 0)
                    for (; j < j_max; j++) {
                        const float* b_row = B + j * ldb;
                        float s0 = 0.f, s1 = 0.f;
                        int k = kk;
                        __m256 acc0 = _mm256_setzero_ps();
                        __m256 acc1 = _mm256_setzero_ps();
                        for (; k + 8 <= k_max; k += 8) {
                            __m256 vb = _mm256_loadu_ps(b_row + k);
                            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a0 + k), vb, acc0);
                            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a1 + k), vb, acc1);
                        }
                        s0 += sum_vector(acc0);
                        s1 += sum_vector(acc1);
                        for (; k < k_max; k++) { 
                            s0 += a0[k]*b_row[k]; s1 += a1[k]*b_row[k]; 
                        }

                        C[i * ldc + j]       += s0;
                        C[(i+1) * ldc + j]   += s1;
                    }
                }
                // leftover single row (M % 2 != 0)
                for (; i < i_max; i++) {
                    const float* a_row = A + i * lda;
                    for (int j = jj; j < j_max; j++) {
                        const float* b_row = B + j * ldb;
                        __m256 acc = _mm256_setzero_ps();
                        int k = kk;
                        for (; k + 8 <= k_max; k += 8)
                            acc = _mm256_fmadd_ps(_mm256_loadu_ps(a_row + k), _mm256_loadu_ps(b_row + k), acc);
                        float s = sum_vector(acc);
                        for (; k < k_max; k++) s += a_row[k]*b_row[k];
                        C[i * ldc + j] += s;
                    }
                }
            }
        }
    }
}