import os
import sys
import glob
import time
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

import torch.nn as nn
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules
quant_nn.TensorQuantizer.use_fb_fake_quant = True
quant_modules.initialize()

class MyModel(nn.Module):
        # Standard convolution
        def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
            super(MyModel, self).__init__()
            self.conv = quant_nn.QuantConv2d(c1, c2, k, s, p, groups=g, bias=True)
            self.conv_quantizer = quant_nn.TensorQuantizer(quant_nn.QuantConv2d.default_quant_desc_input)

        def forward(self, x):
            return self.conv_quantizer(self.conv(x))

if __name__=="__main__":
    model = MyModel(c1=3, c2=128, k=3, s=2, p=1)
    torch.save(model, "one_conv.pt")
    x = torch.rand(1, 3, 736, 1280)
    output = model(x)
    print(output)

    dummy_input = torch.rand(1, 3, 736, 1280)
    torch.onnx.export(
        model,
        dummy_input,
        "one_conv.onnx",
        verbose=False,
        opset_version=13,
        training=False,
        do_constant_folding=True,
        input_names=['images'],
        output_names=['output'],
    )
