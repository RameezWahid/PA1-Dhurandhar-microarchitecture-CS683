// matmul_optimized.cpp  STAGE 3: PUT IT ALL TOGETHER
//
// This is the graded function AND the kernel that gets injected into llama.cpp. Combine
// everything you have learned across the whole assignment  loop reordering, register
// blocking and unrolling (Task 1 / Stage 1 here), cache tiling and software prefetch
// (Stage 2)  and TUNE it to be as fast as you can. Your speedup over matmul_naive determines
// your score (see the tier table the harness prints), and this same function will power a
// real LLM inference via `make llama-demo`.

#include <immintrin.h>
#include <algorithm>
#include "matmul.h"

static inline float sum_vector(__m256 vec){
    __m128 low = _mm256_castps256_ps128(vec);
    __m128 high = _mm256_extractf128_ps(vec, 1);
    __m128 sum = _mm_add_ps(low, high);
    __m128 shuf = _mm_movehdup_ps(sum);
    __m128 sum2 = _mm_add_ps(sum, shuf);
    shuf = _mm_movehl_ps(shuf, sum2);
    __m128 sum1 = _mm_add_ss(sum2, shuf);
    return _mm_cvtss_f32(sum1);
}

// 4 rows of A (a0..a3) x 2 rows of B (b0,b1). 8 named accumulators.
static inline void dot4x2(const float* a0, const float* a1, const float* a2, const float* a3, const float* b0, const float* b1, int kk, int k_max, float& s00, float& s01, float& s10, float& s11, float& s20, float& s21, float& s30, float& s31) {
    __m256 acc00 = _mm256_setzero_ps(); 
    __m256 acc01 = _mm256_setzero_ps();
    __m256 acc10 = _mm256_setzero_ps();
    __m256 acc11 = _mm256_setzero_ps();
    __m256 acc20 = _mm256_setzero_ps();
    __m256 acc21 = _mm256_setzero_ps();
    __m256 acc30 = _mm256_setzero_ps();
    __m256 acc31 = _mm256_setzero_ps();

    int k = kk;
    for (; k + 8 <= k_max; k += 8) {
        _mm_prefetch(reinterpret_cast<const char*>(a0 + k + 64), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(a1 + k + 64), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(a2 + k + 64), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(a3 + k + 64), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(b0 + k + 64), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(b1 + k + 64), _MM_HINT_T0);

        __m256 va0 = _mm256_loadu_ps(a0 + k);
        __m256 va1 = _mm256_loadu_ps(a1 + k);
        __m256 va2 = _mm256_loadu_ps(a2 + k);
        __m256 va3 = _mm256_loadu_ps(a3 + k);
        __m256 vb0 = _mm256_loadu_ps(b0 + k);
        __m256 vb1 = _mm256_loadu_ps(b1 + k);

        // 6 loads feed 8 FMAs.
        acc00 = _mm256_fmadd_ps(va0, vb0, acc00);
        acc01 = _mm256_fmadd_ps(va0, vb1, acc01);
        acc10 = _mm256_fmadd_ps(va1, vb0, acc10);
        acc11 = _mm256_fmadd_ps(va1, vb1, acc11);
        acc20 = _mm256_fmadd_ps(va2, vb0, acc20);
        acc21 = _mm256_fmadd_ps(va2, vb1, acc21);
        acc30 = _mm256_fmadd_ps(va3, vb0, acc30);
        acc31 = _mm256_fmadd_ps(va3, vb1, acc31);
    }

    s00 += sum_vector(acc00); s01 += sum_vector(acc01);
    s10 += sum_vector(acc10); s11 += sum_vector(acc11);
    s20 += sum_vector(acc20); s21 += sum_vector(acc21);
    s30 += sum_vector(acc30); s31 += sum_vector(acc31);

    for (; k < k_max; k++) {
        s00 += a0[k]*b0[k]; s01 += a0[k]*b1[k];
        s10 += a1[k]*b0[k]; s11 += a1[k]*b1[k];
        s20 += a2[k]*b0[k]; s21 += a2[k]*b1[k];
        s30 += a3[k]*b0[k]; s31 += a3[k]*b1[k];
    }
}

// 4 rows of A vs 1 column of B (N % 2 leftover).
static inline void dot4x1(const float* a0, const float* a1, const float* a2, const float* a3, const float* b_row, int kk, int k_max, float& s0, float& s1, float& s2, float& s3){

    __m256 acc0 = _mm256_setzero_ps(); 
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    int k = kk;

    for (; k + 8 <= k_max; k += 8) {

        __m256 vb = _mm256_loadu_ps(b_row + k);
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a0 + k), vb, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a1 + k), vb, acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a2 + k), vb, acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a3 + k), vb, acc3);
    }

    s0 += sum_vector(acc0);
    s1 += sum_vector(acc1);
    s2 += sum_vector(acc2);
    s3 += sum_vector(acc3);

    for (; k < k_max; k++) {
        s0 += a0[k]*b_row[k]; 
        s1 += a1[k]*b_row[k];
        s2 += a2[k]*b_row[k]; 
        s3 += a3[k]*b_row[k];
    }
}

// 1 row of A vs 2 columns of B (M % 4 leftover).
static inline void dot1x2(const float* a_row, const float* b0, const float* b1, int kk, int k_max, float& s0, float& s1){

    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    int k = kk;
    for (; k + 8 <= k_max; k += 8){

        __m256 va = _mm256_loadu_ps(a_row + k);
        acc0 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b0 + k), acc0);
        acc1 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b1 + k), acc1);

    }
    s0 += sum_vector(acc0); 
    s1 += sum_vector(acc1);
    for (; k < k_max; k++){
         s0 += a_row[k]*b0[k]; 
         s1 += a_row[k]*b1[k]; 
         
        }
}

// 1 row vs 1 row (final leftover corner).
static inline float dot1x1(const float* a_row, const float* b_row, int kk, int k_max){
    __m256 acc = _mm256_setzero_ps();
    int k = kk;
    for (; k + 8 <= k_max; k += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a_row + k), _mm256_loadu_ps(b_row + k), acc);
    float s = sum_vector(acc);
    for (; k < k_max; k++) s += a_row[k] * b_row[k];
    return s;
}

void matmul_optimized(const float* A, const float* B, float* C,
                      int M, int N, int K, int lda, int ldb, int ldc){

    const int block_size = 256; 

    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            C[i * ldc + j] = 0.0f;

  
    for (int jj = 0; jj < N; jj += block_size){
        int j_max = std::min(jj + block_size, N);

        for (int kk = 0; kk < K; kk += block_size) {
            int k_max = std::min(kk + block_size, K);

            for (int ii = 0; ii < M; ii += block_size){
                int i_max = std::min(ii + block_size, M);

                int i = ii;
                for (; i + 4 <= i_max; i += 4){
                    const float* a0 = A + (i+0)*lda;
                    const float* a1 = A + (i+1)*lda;
                    const float* a2 = A + (i+2)*lda;
                    const float* a3 = A + (i+3)*lda;

                    int j = jj;
                    for (; j + 2 <= j_max; j += 2){
                        const float* b0 = B + (j+0)*ldb;
                        const float* b1 = B + (j+1)*ldb;
                        float s00=0, s01=0, s10=0, s11=0, s20=0, s21=0, s30=0, s31=0;
                        dot4x2(a0, a1, a2, a3, b0, b1, kk, k_max,
                              s00, s01, s10, s11, s20, s21, s30, s31);
                        C[(i+0)*ldc + (j+0)] += s00;
                        C[(i+0)*ldc + (j+1)] += s01;
                        C[(i+1)*ldc + (j+0)] += s10;
                        C[(i+1)*ldc + (j+1)] += s11;
                        C[(i+2)*ldc + (j+0)] += s20;
                        C[(i+2)*ldc + (j+1)] += s21;
                        C[(i+3)*ldc + (j+0)] += s30;
                        C[(i+3)*ldc + (j+1)] += s31;
                    }
                    // leftover 1 column
                    for (; j < j_max; j++){
                        const float* b_row = B + j * ldb;
                        float s0=0,s1=0,s2=0,s3=0;
                        dot4x1(a0, a1, a2, a3, b_row, kk, k_max, s0, s1, s2, s3);
                        C[(i+0)*ldc + j] += s0;
                        C[(i+1)*ldc + j] += s1;
                        C[(i+2)*ldc + j] += s2;
                        C[(i+3)*ldc + j] += s3;
                    }
                }
                // leftover 1-3 rows
                for (; i < i_max; i++){
                    const float* a_row = A + i * lda;
                    int j = jj;
                    for (; j + 2 <= j_max; j += 2) {
                        const float* b0 = B + j * ldb;
                        const float* b1 = B + (j + 1) * ldb;
                        float s0 = 0.f, s1 = 0.f;
                        dot1x2(a_row, b0, b1, kk, k_max, s0, s1);
                        C[i * ldc + j]     += s0;
                        C[i * ldc + j + 1] += s1;
                    }
                    for (; j < j_max; j++){
                        const float* b_row = B + j * ldb;
                        C[i * ldc + j] += dot1x1(a_row, b_row, kk, k_max);
                    }
                }
            }
        }
    }
}