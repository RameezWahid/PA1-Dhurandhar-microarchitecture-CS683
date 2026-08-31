// matmul_prefetch.cpp  STAGE 2: CACHE BLOCKING + SOFTWARE PREFETCHING

#include <immintrin.h>
#include <iostream>

#include "matmul.h"

static const int TILE = 64;

static float simd_dot(const float *row_A, const float *row_B, int K)
{
    __m256 sum_vec = _mm256_setzero_ps();

    int p = 0;
    for (; p + 8 <= K; p += 8)
    {
        __m256 a_chunk = _mm256_loadu_ps(row_A + p);
        __m256 b_chunk = _mm256_loadu_ps(row_B + p);
        sum_vec = _mm256_fmadd_ps(a_chunk, b_chunk, sum_vec);
    }

    float lanes[8];
    _mm256_storeu_ps(lanes, sum_vec);
    float dot_product = 0.0f;
    for (int lane = 0; lane < 8; ++lane)
    {
        dot_product += lanes[lane];
    }

    for (; p < K; ++p)
    {
        dot_product += row_A[p] * row_B[p];
    }

    return dot_product;
}

void matmul_prefetch(const float *A, const float *B, float *C,
                     int M, int N, int K, int lda, int ldb, int ldc)
{
    long write_count = 0;
    for (int i_tile = 0; i_tile < M; i_tile += TILE)
    {
        int imax = std ::min(i_tile + TILE, M); // handle cases when M is not divisible by TILE
        for (int j_tile = 0; j_tile < N; j_tile += TILE)
        {
            int jmax = std ::min(j_tile + TILE, N); // handle when N is not divisible by TILE

            for (int i = i_tile; i < imax; i++)
            {
                const float *row_A = A + static_cast<long>(i) * lda;

                for (int j = j_tile; j < jmax; j++)
                {
                    const float *row_B = B + static_cast<long>(j) * ldb;
                    C[static_cast<long>(i) * ldc + j] = simd_dot(row_A, row_B, K);
                    write_count++;
                }
            }
        }
    }
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