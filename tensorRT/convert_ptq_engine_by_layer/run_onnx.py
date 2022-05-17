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
from common_onnx import DetectMultiBackend

if __name__=="__main__":
    onnx_name = "one_conv.onnx"  
    half = False
    device = torch.device(0)
    model = DetectMultiBackend(onnx_name, device=device,fp16=half)
    print(model)
   
    x = torch.rand(1, 3, 736, 1280)
    print(x)
    output = model(x)
    print(output)
    print(output.shape)

