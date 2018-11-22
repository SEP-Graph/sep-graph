//
// Created by liang on 3/9/18.
//

#ifndef GROUTE_ATOMIC_UTILS_H
#define GROUTE_ATOMIC_UTILS_H

namespace utils{
    __device__ static float atomicMax(float* address, float val)
    {
        int* address_as_i = (int*) address;
        int old = *address_as_i, assumed;
        do {
            assumed = old;
            old = ::atomicCAS(address_as_i, assumed,
                              __float_as_int(::fmaxf(val, __int_as_float(assumed))));
        } while (assumed != old);
        return __int_as_float(old);
    }

    __device__ static float atomicMin(float* address, float val)
    {
        int* address_as_i = (int*) address;
        int old = *address_as_i, assumed;
        do {
            assumed = old;
            old = ::atomicCAS(address_as_i, assumed,
                              __float_as_int(::fminf(val, __int_as_float(assumed))));
        } while (assumed != old);
        return __int_as_float(old);
    }
}

#endif //GROUTE_ATOMIC_UTILS_H
