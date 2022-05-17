# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Common modules
"""

import json
import math
import platform
import warnings
from collections import OrderedDict, namedtuple
from copy import copy
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torch.cuda import amp

#from utils_new.datasets import exif_transpose, letterbox
#from utils_new.general import (LOGGER, check_requirements, check_suffix, check_version, colorstr, increment_path,
#                           make_divisible, non_max_suppression, scale_coords, xywh2xyxy, xyxy2xywh)
#from utils_new.plots import Annotator, colors, save_one_box
#from utils_new.torch_utils import copy_attr, time_sync


class DetectMultiBackend(nn.Module):
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(self, model='yolov5s.pt', device=torch.device('cpu'), fp16=False):
        super().__init__()
        #w = str(weights[0] if isinstance(weights, list) else weights)

        import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        stride, names = 32, [f'class{i}' for i in range(1000)]  # assign defaults
        logger = trt.Logger(trt.Logger.INFO)
        #with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
        #    model = runtime.deserialize_cuda_engine(f.read())
        #model = weights
        bindings = OrderedDict()
        print(model.num_bindings)
        for index in range(model.num_bindings):
            name = model.get_binding_name(index)
            dtype = trt.nptype(model.get_binding_dtype(index))
            shape = tuple(model.get_binding_shape(index))
            data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
            bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
            if model.binding_is_input(index) and dtype == np.float16:
                fp16 = True
        binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
        context = model.create_execution_context()
        batch_size = bindings['images'].shape[0]
        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False, val=False):
        # YOLOv5 MultiBackend inference
        b, ch, h, w = im.shape  # batch, channel, height, width
        assert im.shape == self.bindings['images'].shape, (im.shape, self.bindings['images'].shape)
        self.binding_addrs['images'] = int(im.data_ptr())
        #t1 = time_sync()
        self.context.execute_v2(list(self.binding_addrs.values()))
        #t2 = time_sync()
        #print("execute_v2_time:\t", t2-t1)
        y = self.bindings['output'].data

        if isinstance(y, np.ndarray):
            y = torch.tensor(y, device=self.device)
        return (y, []) if val else y

    def warmup(self, imgsz=(1, 3, 640, 640)):
        # Warmup model by running inference once
        if any((self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb)):  # warmup types
            if self.device.type != 'cpu':  # only warmup GPU models
                im = torch.zeros(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
                for _ in range(2 if self.jit else 1):  #
                    self.forward(im)  # warmup

    @staticmethod
    def model_type(p='path/to/model.pt'):
        # Return model type from model path, i.e. path='path/to/model.onnx' -> type=onnx
        from export import export_formats
        suffixes = list(export_formats().Suffix) + ['.xml']  # export suffixes
        check_suffix(p, suffixes)  # checks
        p = Path(p).name  # eliminate trailing separators
        pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, xml2 = (s in p for s in suffixes)
        xml |= xml2  # *_openvino_model or *.xml
        tflite &= not edgetpu  # *.tflite
        return pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs


