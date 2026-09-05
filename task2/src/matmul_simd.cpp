// matmul_simd.cpp  STAGE 1: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "matmul.h"

void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO(student): replace this placeholder with your register-tiled AVX2 implementation.
    const int simd_width = 8;
    float answer[simd_width],sum=0;
    for(int i=0;i<M ;i++){
        for(int j=0;j<N;j++){
            sum=0;
            __m256 acc =  _mm256_setzero_ps();
            const float* a = A + static_cast<long>(i) * lda;
            const float* b = B + static_cast<long>(j) * ldb;
            
            for (int p = 0; p < K; p+=simd_width) {
                __m256 input_a =  _mm256_loadu_ps(a + p);
                __m256 input_b =  _mm256_loadu_ps(b + p);
                acc=  _mm256_fmadd_ps(input_a,input_b,acc);
            }
             _mm256_storeu_ps(answer,acc);
           for(int k=0;k<simd_width;k++){
            sum+=answer[k];
           }
           C[static_cast<long>(i) * ldc + j] = sum;
        }
    }
}


//Code for 128bit SIMD implementation.
// void matmul_simd(const float* A, const float* B, float* C,
//                  int M, int N, int K, int lda, int ldb, int ldc) {
//     // TODO(student): replace this placeholder with your register-tiled AVX2 implementation.
//     const int simd_width = 4;
//     float answer[simd_width],sum=0;
//     for(int i=0;i<M ;i++){
//         for(int j=0;j<N;j++){
//              sum=0;  
//             __m256 acc =  _mm256_setzero_ps();
//             const float* a = A + static_cast<long>(i) * lda;
//             const float* b = B + static_cast<long>(j) * ldb;
            
//             for (int p = 0; p < K; p+=simd_width) {
//                 __m256 input_a =  _mm256_loadu_ps(a + p);
//                 __m256 input_b =  _mm256_loadu_ps(b + p);
//                 acc=  _mm256_fmadd_ps(input_a,input_b,acc);
//             }
//              _mm256_storeu_ps(answer,acc);
//            for(int k=0;k<simd_width;k++){
        //     sum+=answer[k];
        //    }
//            C[static_cast<long>(i) * ldc + j] = sum;
//         }
//     }
// }


