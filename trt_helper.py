#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import torch
import tensorrt as trt
import numpy as np
import ctypes
import math
import time

from typing import Optional, Tuple

import pycuda.driver as cuda
import pycuda.autoinit

class TrtNetworkHelper():
    """TensorRT Network Definition helper for Pytorch"""
    def __init__(self, network, plugin_registry, logger):
        self.network = network
        self.plugin_registry = plugin_registry
        self.logger = logger

        self.input_num = 0

    def set_layer_name(self, layer, name):
        """
        Tool function. Set the name of trt layer or plugin and print output shapes.
        """
        if not layer:
            raise RuntimeError("Could not name")

        layer.name = str(self.network.num_layers) + "_" + name
        for i in range(0, layer.num_outputs):
            shape = layer.get_output(i).shape
            self.logger.log(trt.Logger.INFO, "[Network] " + layer.name + ", output[" + str(i) + "] shape= " + str(shape))

        return None

    def check_trt_layer(self, trt_layer):
        """
        Tool function. check trt layer,
        """
        if not trt_layer:
            raise RuntimeError("add " + str(trt_layer) + " failed!")

        for i in range(0, trt_layer.num_outputs):
            shape = trt_layer.get_output(i).shape
            # print("===> layer: ", trt_layer.name, trt.volume(shape))
            # 如果没有自动推导出shape，则插入算子失败
            if len(shape) is 1:
                raise RuntimeError("add " + trt_layer.name + " failed!")

    def layer_post_process(self, trt_layer, layer_name, precision):
        """
        Tool function. set precision, set_layer_name and check_trt_layer
        """
        if precision is not None:
            trt_layer.precision = precision

        self.set_layer_name(trt_layer, layer_name)
        self.check_trt_layer(trt_layer)

    def addInput(self, name, dtype, shape):
        if name is None:
            name = "input" + str(self.input_num)

        self.input_num = self.input_num + 1

        trt_input = self.network.add_input(name=name, dtype=dtype, shape=shape)
        if not trt_input:
            raise RuntimeError("addInput failed!")

        self.logger.log(trt.Logger.INFO, "[Network] add input:" + name + ", shape=" + str(shape))

        return trt_input

    def markOutput(self, x: trt.ITensor):
        self.network.mark_output(x)
        self.logger.log(trt.Logger.INFO, "[Network] mark output:" + x.name + ", shape=" + str(x.shape))

    def addEmbedding(self, indices, weight, layer_name=None, precision=None):
        # 自动命名为 (Unnamed Layer* 0)[Constant]，名字中带序号和类型
        constant_layer = self.network.add_constant(weight.shape, trt.Weights(weight))
        gather_layer = self.network.add_gather(constant_layer.get_output(0),
                                               indices, axis=0)

        if layer_name is None:
            # 对于重复的名字，TRT会自动添加数字前标
            layer_name = "nn.Embedding"
        else:
            layer_name = "nn.Embedding." + layer_name

        self.layer_post_process(gather_layer, layer_name, precision)
        return gather_layer.get_output(0)

    def addGELU(self, x, layer_name=None, precision=None):
        POW = self.network.add_constant((1, 1, 1), trt.Weights(np.ascontiguousarray([3.0], dtype=np.float32)))
        MULTIPLY = self.network.add_constant((1, 1, 1), trt.Weights(np.ascontiguousarray([0.044715], dtype=np.float32)))
        SQRT = self.network.add_constant((1, 1, 1), trt.Weights((np.ascontiguousarray([0.79788456080286535587989211986876], dtype=np.float32))))
        ONE = self.network.add_constant((1, 1, 1), trt.Weights((np.ascontiguousarray([1.0], dtype=np.float32))))
        HALF = self.network.add_constant((1, 1, 1), trt.Weights((np.ascontiguousarray([0.5], dtype=np.float32))))
        X_pow = self.network.add_elementwise(x, POW.get_output(0), trt.ElementWiseOperation.POW)
        X_pow_t = X_pow.get_output(0)
        X_mul = self.network.add_elementwise(X_pow_t, MULTIPLY.get_output(0), trt.ElementWiseOperation.PROD)
        X_add = self.network.add_elementwise(x, X_mul.get_output(0), trt.ElementWiseOperation.SUM)
        X_sqrt = self.network.add_elementwise(X_add.get_output(0), SQRT.get_output(0), trt.ElementWiseOperation.PROD)
        X_sqrt_tensor = X_sqrt.get_output(0)
        X_tanh = self.network.add_activation(X_sqrt_tensor, trt.ActivationType.TANH)
        X_tanh_tensor = X_tanh.get_output(0)
        X_one = self.network.add_elementwise(X_tanh_tensor, ONE.get_output(0), trt.ElementWiseOperation.SUM)
        CDF = self.network.add_elementwise(X_one.get_output(0), HALF.get_output(0), trt.ElementWiseOperation.PROD)
        gelu_layer = self.network.add_elementwise(CDF.get_output(0), x, trt.ElementWiseOperation.PROD)

        if layer_name is None:
            layer_name = "nn.GELU"
        else:
            layer_name = "nn.GELU." + layer_name

        self.layer_post_process(gelu_layer, layer_name, precision)

        return gelu_layer.get_output(0)

    def addLayerNorm(self, x, gamma, beta, layer_name=None, precision=None):
        inputTensorList = []
        inputTensorList.append(x)

        layernorm_plugin = None
        for c in self.plugin_registry.plugin_creator_list:
            if c.name == 'MyLayerNorm':
                layernorm_plugin = c.create_plugin(c.name, trt.PluginFieldCollection([]))
            # if c.name == 'LayerNorm':
                gamma_const = self.addConstant(np.ascontiguousarray([[gamma]]), "LayerNorm.Gamma")
                beta_const = self.addConstant(np.ascontiguousarray([[beta]]), "LayerNorm.Beta")
            #     inputTensorList.append(gamma_const)
            #     inputTensorList.append(beta_const)
            #     layernorm_plugin = c.create_plugin(c.name, trt.PluginFieldCollection([
            #         trt.PluginField("epsilon", np.array([1.e-5], np.float32), trt.PluginFieldType.FLOAT32)
            #     ]))
        
        assert(layernorm_plugin)
        
        trt_layer = self.network.add_plugin_v2(inputTensorList, layernorm_plugin)

        if layer_name is None:
            layer_name = "LayerNorm"
        else:
            layer_name = "LayerNorm." + layer_name
        self.layer_post_process(trt_layer, layer_name, precision)

        out = trt_layer.get_output(0)
        # return out

        X_mul = self.network.add_elementwise(out, gamma_const, trt.ElementWiseOperation.PROD)
        X_add = self.network.add_elementwise(X_mul.get_output(0), beta_const, trt.ElementWiseOperation.SUM)
        return X_add.get_output(0)

    def addLinear(self, x, weight, bias, layer_name=None, precision=None, flag=False):
        if layer_name:
            layer_name = "(Linear)." + layer_name
        else:
            layer_name = "(Linear)."

        # if flag:
        #     self.markOutput(x)

        weight_const = self.addConstant(weight, 'weight.' + layer_name)
        weight_const = self.addShuffle(weight_const, None, (1, *weight_const.shape), None, "Custom.Linear.Weight_")
        matmul_out = self.addMatMul(x, weight_const, layer_name)

        # if flag:
        #     self.markOutput(matmul_out)
        #     print(f"===> bias: ", bias.shape)
        #     print(f"===> matmul_out: ", matmul_out.shape)
        #     exit(0)

        # bias_const = self.addConstant(bias, 'bias.' + layer_name)
        # bias_const = self.addShuffle(bias_const, None, matmul_out.shape, None, "Custom.Linear.Bias_")
        bias_const = self.addConstant(np.ascontiguousarray([[bias]]), 'bias.' + layer_name)
        # if flag:
        #     print("===> ", bias_const.shape)
        #     exit(0)
        out = self.addAdd(matmul_out, bias_const, layer_name, precision)
        
        # if flag:
        #     self.markOutput(out)
        
        return out

    def addReLU(self, layer, x, layer_name=None, precision=None):
        trt_layer = self.network.add_activation(x, type=trt.ActivationType.RELU)

        if layer_name is None:
            layer_name = "nn.ReLU"

        self.layer_post_process(trt_layer, layer_name, precision)

        x = trt_layer.get_output(0)
        return x

    def addSoftmax(self, x: trt.ITensor, dim: int = -1, layer_name=None, precision=None) -> trt.ITensor:
        
        softmax_layer = self.network.add_softmax(x)
        shape_len = len(x.shape)
        if dim < 0:
            dim = shape_len + dim

        # axes 以 bit mask 的方式表示，这里是（N, S, H, D）, dim=-1，则应该表示（0，0，0，1）
        softmax_layer.axes = 1 << 3
        
        if layer_name is None:
            layer_name = f"Softmax[dim={dim}]"
        else:
            layer_name = f"Softmax[dim={dim}]." + layer_name
        self.layer_post_process(softmax_layer, layer_name, precision)
        return softmax_layer.get_output(0)

    ################## unary op ###################
    def addLog(self, x: trt.ITensor, layer_name=None, precision=None):
        trt_layer = self.network.add_unary(x, trt.UnaryOperation.LOG)
        if layer_name is None:
            layer_name = "unary.log"
        else:
            layer_name = "unary.log." + layer_name

        self.layer_post_process(trt_layer, layer_name, precision)

        x = trt_layer.get_output(0)
        return x

    ################## elementwise op ###################
    def addAdd(self, a, b, layer_name=None, precision=None):
        add_layer = self.network.add_elementwise(a, b, trt.ElementWiseOperation.SUM)
        
        if layer_name is None:
            layer_name = "Add"
        else:
            layer_name = "Add." + layer_name
        self.layer_post_process(add_layer, layer_name, precision)
        return add_layer.get_output(0)

    # tensor and scalar op
    def addScale(
            self,
            x: trt.ITensor,
            scale: float,
            layer_name: str = None,
            precision: trt.DataType = None
    ) -> trt.ITensor:
        """scale"""
        scale_layer = self.network.add_scale(
            x,
            trt.ScaleMode.UNIFORM,
            None,
            trt.Weights(np.ascontiguousarray([scale], dtype=np.float32)
        ), None)
        if layer_name is None:
            layer_name = "Scale"
        else:
            layer_name = "Scale." + layer_name
        self.layer_post_process(scale_layer, layer_name, precision)
        return scale_layer.get_output(0)

    def addMatMul(self, a: trt.ITensor, b: trt.ITensor, layer_name: Optional[str] = None) -> trt.ITensor:
        # add MatMul
        matmul_layer = self.network.add_matrix_multiply(
            a, trt.MatrixOperation.NONE, b, trt.MatrixOperation.NONE)
        if layer_name is None:
            layer_name = "MatMul"
        else:
            layer_name = "MatMul." + layer_name
        self.layer_post_process(matmul_layer, layer_name, None)
        return matmul_layer.get_output(0)

    def addConstant(self, w, layer_name: Optional[str] = None) -> trt.ITensor:
        trt_layer = self.network.add_constant(w.shape, trt.Weights(np.ascontiguousarray(w)))

        if layer_name is None:
            layer_name = "trt.Constant"
        else:
            layer_name = "trt.Constant." + layer_name

        self.layer_post_process(trt_layer, layer_name, None)
        x = trt_layer.get_output(0)
        return x

    def addShuffle(
        self,
        x: trt.ITensor,
        first_transpose: trt.Permutation,
        reshape_dims: trt.Dims,
        second_transpose: trt.Permutation,
        layer_name: Optional[str] = None
    ) -> trt.ITensor:
        """"""
        trt_layer = self.network.add_shuffle(x)
        if first_transpose is not None:
            trt_layer.first_transpose = first_transpose

        if reshape_dims is not None:
            trt_layer.reshape_dims = reshape_dims

        if second_transpose is not None:
            trt_layer.second_transpose = second_transpose

        if layer_name is None:
            layer_name = "trt.Shuffle"
        else:
            layer_name = "trt.Shuffle." + layer_name

        self.layer_post_process(trt_layer, layer_name, None)

        x = trt_layer.get_output(0)
        return x


class InferHelper():
    """"""
    def __init__(self, plan_name, trt_logger):
        """"""
        self.logger = trt_logger
        self.runtime = trt.Runtime(trt_logger)
        with open(plan_name, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()
            self.context.active_optimization_profile = 0

    def infer(self, inputs: list):
        nInput = len(inputs)

        bufferD = []
        # alloc memory
        for i in range(nInput):
            bufferD.append(cuda.mem_alloc(inputs[i].nbytes))
            cuda.memcpy_htod(bufferD[i], inputs[i].ravel())
            self.context.set_binding_shape(i, tuple(inputs[i].shape))
            # print(inputs[i].nbytes)

        # for i in range(0, self.engine.num_bindings):
        #     print("get_binding_shape:" + str(self.context.get_binding_shape(i)))

        outputs = []
        for i in range(len(inputs), self.engine.num_bindings):
            outputs.append(np.zeros(self.context.get_binding_shape(i)).astype(np.float32))

        nOutput = len(outputs)
        for i in range(nOutput):
            bufferD.append(cuda.mem_alloc(outputs[i].nbytes))
            # print(outputs[i].nbytes)

        for i in range(len(inputs), self.engine.num_bindings):
            trt_output_shape = self.context.get_binding_shape(i)
            output_idx = i - len(inputs)
            if not (list(trt_output_shape) == list(outputs[output_idx].shape)):
                self.logger.log(trt.Logger.ERROR, "[Infer] output shape is error!")
                self.logger.log(trt.Logger.ERROR, "trt_output.shape = " + str(trt_output_shape))
                self.logger.log(trt.Logger.ERROR, "base_output.shape = " + str(outputs[output_idx].shape))
                assert(0)

        # warm up
        self.context.execute_v2(bufferD)

        T1 = time.perf_counter()

        self.context.execute_v2(bufferD)

        T2 =time.perf_counter()
        print("time=" + str((T2-T1) * 1000) + "ms")

        for i in range(nInput, nInput + nOutput):
            cuda.memcpy_dtoh(outputs[i - nInput].ravel(), bufferD[i])

        for i in range(0, len(outputs)):
            print("outputs.shape:" + str(outputs[i].shape))
            print("outputs.sum:" + str(outputs[i].sum()))
            # print(outputs[i])

            # print("trt_output.shape:" + str(trt_output.shape))
            # print("trt_output.sum:" + str(trt_output.sum()))
            # print(trt_output.view(-1)[0:10])
            # print("torch.allclose result:" + str(torch.allclose(base_output, trt_output, 1e-05, 1e-03)))
            # print("====================")
        return outputs
        # return torch.allclose(base_output, trt_output, 1e-05, 1e-03)
