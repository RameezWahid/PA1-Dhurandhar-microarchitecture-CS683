// matmul_prefetch.cpp  STAGE 2: CACHE BLOCKING + SOFTWARE PREFETCHING

#include <immintrin.h>

#include "matmul.h"

void matmul_prefetch(const float *A, const float *B, float *C,
                     int M, int N, int K, int lda, int ldb, int ldc)
{
    // TODO(student): replace this placeholder with your cache-blocked SIMD + prefetch
    // implementation.
    matmul_naive(A, B, C, M, N, K, lda, ldb, ldc);
}

void matmul_simd_baseline(const float *A, const float *B, float *C,
                          int M, int N, int K, int lda, int ldb, int ldc)
{
    for (int i = 0; i < M; ++i)
    {
        const float *rowA = A + static_cast<long>(i) * lda;
        for (int j = 0; j < N; j++)
        {
            const float *rowB = B + static_cast<long>(j) * ldb;
            // Accumulagtor holding 8 partial sum
            __m256 sum_vec = _mm256_setzero_ps();
            int p = 0;
            for (; p + 8 <= K; p += 8)
            {
                __m256 a_chunk = _mm256_loadu_ps(rowA + p);
                __m256 b_chunk = _mm256_loadu_ps(rowB + p);

                sum_vec = _mm256_fmadd_ps(a_chunk, b_chunk, sum_vec);
            }

            // collapse all 8 numbers into one
            float lanes[8];
            _mm256_storeu_ps(lanes, sum_vec);
            float dot_product = 0.0f;
            for (int lane = 0; lane < 8; lane++)
            {
                dot_product += lanes[lane];
            }

            for (; p < K; p++)
            {
                dot_product += rowA[p] * rowB[p];
            }

            C[i * ldc + j] = dot_product;
        }
    }
}