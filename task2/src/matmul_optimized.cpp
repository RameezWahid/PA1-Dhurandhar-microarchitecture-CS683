// matmul_optimized.cpp  STAGE 3: PUT IT ALL TOGETHER
//
// This is the graded function AND the kernel that gets injected into llama.cpp. Combine
// everything you have learned across the whole assignment  loop reordering, register
// blocking and unrolling (Task 1 / Stage 1 here), cache tiling and software prefetch
// (Stage 2)  and TUNE it to be as fast as you can. Your speedup over matmul_naive determines
// your score (see the tier table the harness prints), and this same function will power a
// real LLM inference via `make llama-demo`.

#include <immintrin.h>

#include "matmul.h"
#include <algorithm>


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

void matmul_optimized(const float* A, const float* B, float* C,
                      int M, int N, int K, int lda, int ldb, int ldc) {
    
    const int block_size = 64; // Example block size, can be tuned for performance


    for(int i = 0; i < M; i++)
            for(int j = 0; j < N; j++)
                        C[i * ldc + j] = 0.0f;

    
     // Step 1: TILING.

     for(int ii = 0; ii < M; ii += block_size){
        int i_max = std::min(ii + block_size, M);

        for(int jj = 0; jj < N; jj += block_size){
            int j_max = std::min(jj + block_size, N);

            for(int kk = 0; kk < K; kk += block_size){
                int k_max = std::min(kk + block_size, K);

                // Step 2: inside one small tile, do the actual multiply-add,
                for(int i = ii; i < i_max; i++){
                    const float* a_row = A + i * lda;

                    for(int j = jj; j < j_max; j++){
                        const float* b_row = B + j * ldb;

                          // Step 3 + Step 3b (UNROLLING): two independent
                        // accumulators, 16 floats processed per iteration
                        // instead of 8, so the two FMAs below don't have
                        // to wait on each other (breaks the dependency
                        // chain -> more instruction-level parallelism).


                         __m256 acc0 = _mm256_setzero_ps();
                        __m256 acc1 = _mm256_setzero_ps();
                        
                        int k = kk;
                        for(; k + 16 <= k_max; k += 16){
                             // Step 4: PREFETCH -- ask for data 64 floats ahead
                            _mm_prefetch(reinterpret_cast<const char*>(a_row + k + 64), _MM_HINT_T0);
                            _mm_prefetch(reinterpret_cast<const char*>(b_row + k + 64), _MM_HINT_T0);

                           __m256 a_vec0 = _mm256_loadu_ps(a_row + k);
                            __m256 b_vec0 = _mm256_loadu_ps(b_row + k);
                            acc0 = _mm256_fmadd_ps(a_vec0, b_vec0, acc0);
 
                            __m256 a_vec1 = _mm256_loadu_ps(a_row + k + 8);
                            __m256 b_vec1 = _mm256_loadu_ps(b_row + k + 8);
                            acc1 = _mm256_fmadd_ps(a_vec1, b_vec1, acc1);
                        }
                            // Any remaining full 8-chunk that didn't fit the
                        // 16-wide unrolled loop (i.e. k_max - k is 8..15).

                           for (; k + 8 <= k_max; k += 8) {
                            __m256 a_vec = _mm256_loadu_ps(a_row + k);
                            __m256 b_vec = _mm256_loadu_ps(b_row + k);
                            acc0 = _mm256_fmadd_ps(a_vec, b_vec, acc0);
                        }
                        float partial_sum = sum_vector(acc0) + sum_vector(acc1);

                         // Step 5: leftover elements (< 8) that don't fill
                        // a full vector.
                         for (; k < k_max; k++) {
                            partial_sum += a_row[k] * b_row[k];
                        }

                        // Step 6: add this K-chunk's contribution to C[i][j].
                        C[i * ldc + j] += partial_sum;

                        


                    }
                    

                }

            }
        }
     }



    // matmul_naive(A, B, C, M, N, K, lda, ldb, ldc);
}
