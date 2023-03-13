/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
 
 #include "LayerNormPlugin.h"

using namespace nvinfer1;

PluginFieldCollection LayerNormPluginCreator::fc_{};
std::vector<PluginField> LayerNormPluginCreator::attr_;

__global__ void layerNormKernel(float *pInput, float *gamma, float *beta, float *pOutput)
{
    // nDim=768
    const int tx = threadIdx.x, index = blockIdx.x * 768 + threadIdx.x;

    __shared__ float temp[256];

    float value0 = pInput[index];
    float value1 = pInput[index + 256];
    float value2 = pInput[index + 512];

    temp[tx] = value0 + value1 + value2;
    __syncthreads();

    for (int stride = 128; stride >= 1; stride /= 2)
    {
        if (tx < stride)
        {
            temp[tx] += temp[tx + stride];
        }
        __syncthreads();
    }
    float mean = temp[0] / 768;
    __syncthreads();

    temp[tx] = (value0 - mean) * (value0 - mean) + (value1 - mean) * (value1 - mean) + (value2 - mean) * (value2 - mean);
    __syncthreads();

    for (int stride = 128; stride >= 1; stride /= 2)
    {
        if (tx < stride)
        {
            temp[tx] += temp[tx + stride];
        }
        __syncthreads();
    }
    float var = temp[0] / 768;

    pOutput[index]       = (value0 - mean) * rsqrtf(var + 6e-6) * gamma[tx] + beta[tx];
    pOutput[index + 256] = (value1 - mean) * rsqrtf(var + 6e-6) * gamma[tx + 256] + beta[tx + 256];
    pOutput[index + 512] = (value2 - mean) * rsqrtf(var + 6e-6) * gamma[tx + 512] + beta[tx + 512];
}

int32_t LayerNormPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    WHERE_AM_I();
    const int nBlock = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1];

    layerNormKernel <<<nBlock, 256, 0, stream>>>(
        (float *)inputs[0], (float *)inputs[1], (float *)inputs[2], (float *)outputs[0]);
    return 0;
}

REGISTER_TENSORRT_PLUGIN(LayerNormPluginCreator);

