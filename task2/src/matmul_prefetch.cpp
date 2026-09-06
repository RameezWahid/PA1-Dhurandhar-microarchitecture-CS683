// matmul_prefetch.cpp  STAGE 2: CACHE BLOCKING + SOFTWARE PREFETCHING

#include <immintrin.h>
#include <iostream>
#include <cstdlib>
#include <cstring>

#include "matmul.h"

static const int TILE = 48;

// Reads PREFETCH_DISTANCE from an environment variable so we can sweep it
// without recompiling each time. Falls back to 16 if not set.
static int get_prefetch_distance()
{
    const char *env_val = std::getenv("PF_DIST");
    if (env_val != nullptr)
    {
        return std::atoi(env_val);
    }
    return 16; // default distance
}

// Reads PF_HINT from an environment variable and maps it to the matching
// _MM_HINT_* constant. Accepts "T0", "T1", "T2", "NTA". Falls back to T0.
static int get_prefetch_hint()
{
    const char *env_val = std::getenv("PF_HINT");
    if (env_val == nullptr)
    {
        return _MM_HINT_T0; // default hint
    }

    if (std::strcmp(env_val, "T0") == 0)
        return _MM_HINT_T0;
    if (std::strcmp(env_val, "T1") == 0)
        return _MM_HINT_T1;
    if (std::strcmp(env_val, "T2") == 0)
        return _MM_HINT_T2;
    if (std::strcmp(env_val, "NTA") == 0)
        return _MM_HINT_NTA;

    return _MM_HINT_T0; // fall back to default
}

static float simd_dot(const float *row_A, const float *row_B, int K,
                      int prefetch_distance, int prefetch_hint)
{
    __m256 sum_vec = _mm256_setzero_ps();

    int p = 0;
    for (; p + 8 <= K; p += 8)
    {
        if (p + prefetch_distance + 8 <= K)
        {
            _mm_prefetch(reinterpret_cast<const char *>(row_A + p + prefetch_distance), prefetch_hint);
            _mm_prefetch(reinterpret_cast<const char *>(row_B + p + prefetch_distance), prefetch_hint);
        }

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
    int prefetch_distance = get_prefetch_distance();
    int prefetch_hint = get_prefetch_hint();

    for (int i_tile = 0; i_tile < M; i_tile += TILE)
    {
        int imax = std::min(i_tile + TILE, M);
        for (int j_tile = 0; j_tile < N; j_tile += TILE)
        {
            int jmax = std::min(j_tile + TILE, N);

            for (int i = i_tile; i < imax; i++)
            {
                const float *row_A = A + static_cast<long>(i) * lda;

                for (int j = j_tile; j < jmax; j++)
                {
                    const float *row_B = B + static_cast<long>(j) * ldb;
                    C[static_cast<long>(i) * ldc + j] =
                        simd_dot(row_A, row_B, K, prefetch_distance, prefetch_hint);
                }
            }
        }
    }
}

// Base line simd
void matmul_simd_baseline(const float *A, const float *B, float *C,
                          int M, int N, int K, int lda, int ldb, int ldc)
{
    for (int i = 0; i < M; ++i)
    {
        const float *rowA = A + static_cast<long>(i) * lda;
        for (int j = 0; j < N; j++)
        {
            const float *rowB = B + static_cast<long>(j) * ldb;
            __m256 sum_vec = _mm256_setzero_ps();
            int p = 0;
            for (; p + 8 <= K; p += 8)
            {
                __m256 a_chunk = _mm256_loadu_ps(rowA + p);
                __m256 b_chunk = _mm256_loadu_ps(rowB + p);
                sum_vec = _mm256_fmadd_ps(a_chunk, b_chunk, sum_vec);
            }

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