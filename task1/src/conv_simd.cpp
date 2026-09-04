// conv_simd.cpp  STAGE 4: SIMD with AVX2 intrinsics
#include <immintrin.h>
#include <emmintrin.h>

#include "convolution.h"

void conv_simd(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    // TODO(student): replace this placeholder with your AVX2 implementation.
    const int p = K / 2;
    const int in_stride = W + 2 * p;
    const int simd_stride = 4;
    for (int oy=0;oy<H; ++oy){
        for(int ox=0; ox<W ; ox+=simd_stride){
            __m128 acc = _mm_setzero_ps();
            for(int ky=0 ; ky<K ; ++ky){
                for(int kx=0; kx<K ; ++kx){
                    float weight= ker[ky*K+kx];
                    __m128 input = _mm_loadu_ps(in + (oy+ky)*in_stride + (ox+kx));
                    __m128 kernel_input = _mm_set_ps1(weight);
                    __m128 mul_output = _mm_mul_ps(input, kernel_input);
                    acc = _mm_add_ps(acc, mul_output);
                }
            }
            _mm_storeu_ps((out + oy*W +ox ),acc );
            
        }
    }
}
