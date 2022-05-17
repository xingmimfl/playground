#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import sys

# This sample uses an MNIST PyTorch model to create a TensorRT Inference Engine
import numpy as np
import tensorrt as trt
import onnx
from onnx import numpy_helper

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
def GiB(val):
    return val * 1 << 30

import torch
torch.set_printoptions(threshold=100000)
def setup_seed(seed=0):
    import torch
    import os
    import numpy as np
    import random
    torch.manual_seed(seed)  # 为CPU设置随机种子
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
setup_seed(666)
from common_engine import DetectMultiBackend


def load_params(onnx_path):
    model = onnx.load(onnx_path)
    weights = model.graph.initializer
    #tensor_dict = dict([(w.name, np.frombuffer(w.raw_data, np.float32).reshape(w.dims)) for w in weights])

    weights_map = {}
    for t in weights:
        weights_map[t.name] = numpy_helper.to_array(t)
    return weights_map


# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
#TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

class ModelData(object):
    INPUT_NAME = "images"
    INPUT_SHAPE = (1, 3, 736, 1280)
    OUTPUT_NAME = "output"
    OUTPUT_SIZE = 10
    DTYPE = trt.float32

def populate_network(network, onnx_name):
    # Configure the network layers based on the weights provided.
    onnx_weights = torch.load(onnx_name)
    conv1_weights = onnx_weights['conv.weight'].cpu().numpy()
    conv1_bias = onnx_weights['conv.bias'].cpu().numpy()
    input_tensor = network.add_input(name=ModelData.INPUT_NAME, dtype=ModelData.DTYPE, shape=ModelData.INPUT_SHAPE) 
    conv1 = network.add_convolution_nd(
        input=input_tensor,
        num_output_maps=128, 
        kernel_shape=(3, 3), 
        kernel=conv1_weights, 
        bias=conv1_bias)
    conv1.stride_nd = (2, 2)
    conv1.padding_nd = (1, 1)
    #conv1.precision = trt.float32
    conv1.get_output(0).name = ModelData.OUTPUT_NAME 
    network.mark_output(tensor=conv1.get_output(0))


def build_engine(onnx_name):
    # For more information on TRT basics, refer to the introductory samples.
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(EXPLICIT_BATCH)
    config = builder.create_builder_config()
    config.max_workspace_size = GiB(1)
    builder.max_batch_size = 1
    # Populate the network using weights from the PyTorch model.
    populate_network(network, onnx_name)
    # Build and return an engine.

    runtime = trt.Runtime(TRT_LOGGER)
    plan = builder.build_serialized_network(network, config)
    return runtime.deserialize_cuda_engine(plan)


if __name__=="__main__":
    onnx_path  =  "one_conv.pth"
    engine = build_engine(onnx_path)
    device = torch.device(0)
    half = False
    model = DetectMultiBackend(engine, device, half)
    x = torch.rand(1, 3, 736, 1280).to(device)
    print(x)
    output = model(x)
    #print(output)
    print(output)
    print("output.size:\t", output.shape)
