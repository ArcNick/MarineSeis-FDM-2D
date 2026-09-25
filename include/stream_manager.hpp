#ifndef STREAM_MANAGER_HPP
#define STREAM_MANAGER_HPP

#include "cuda_check.h"

class StreamManager {
public:
    cudaStream_t vel;
    cudaStream_t sig;
    cudaStream_t vel_solid;
    cudaStream_t vel_fluid;
    cudaStream_t sig_solid;
    cudaStream_t sig_fluid;

    StreamManager() {
        CHECK(cudaStreamCreate(&vel));
        CHECK(cudaStreamCreate(&sig));
        CHECK(cudaStreamCreate(&vel_solid));
        CHECK(cudaStreamCreate(&vel_fluid));
        CHECK(cudaStreamCreate(&sig_solid));
        CHECK(cudaStreamCreate(&sig_fluid));
    }
    ~StreamManager() {
        CHECK(cudaStreamDestroy(vel));
        CHECK(cudaStreamDestroy(sig));
        CHECK(cudaStreamDestroy(vel_solid));
        CHECK(cudaStreamDestroy(vel_fluid));
        CHECK(cudaStreamDestroy(sig_solid));
        CHECK(cudaStreamDestroy(sig_fluid));
    }
};

#endif