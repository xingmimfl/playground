import datetime
import os
import sys
import time
import collections

import torch
import torch.utils.data
from torch import nn

from tqdm import tqdm

import torchvision
from torchvision import transforms

#running onnx, for resnet50 example

# For simplicity, import train and eval functions from the train script from torchvision instead of copything them here
# Download torchvision from https://github.com/pytorch/vision
sys.path.append("xxxxxxx/pytorch_quantization/vision/references/classification")
from train import train_one_epoch, load_data
import vision_utils as utils

import onnxruntime
import numpy as np

def evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix=''):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f'Test: {log_suffix}'
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        input_name = model.get_inputs()[0].name
        output_name = model.get_outputs()[0].name
        image = image.cpu().numpy().astype(np.float32)
        output = model.run([output_name], {input_name:image})
        output = output[0]
        output = torch.tensor(output).float().to(device)
        loss = criterion(output, target)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        # FIXME need to take into account that the datasets
        # could have been padded in distributed setup
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print(f'{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}')
    return metric_logger.acc1.global_avg

#----Create model with pretrained weight
#model = torchvision.models.resnet50(pretrained=True, progress=False)
#model.cuda()
#print(model)

model = onnxruntime.InferenceSession('quant_resnet50.onnx', providers=['CUDAExecutionProvider'])


#----Create data loader----
data_path = "xxx/imagenet"
batch_size = 1

traindir = os.path.join(data_path, 'train')
valdir = os.path.join(data_path, 'val')
_args = collections.namedtuple('mock_args', ['model', 'distributed', 'cache_dataset'])
dataset, dataset_test, train_sampler, test_sampler = load_data(traindir, valdir, _args(model='resnet50', distributed=False, cache_dataset=False))

data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size,
    sampler=train_sampler, num_workers=4, pin_memory=True)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=batch_size,
    sampler=test_sampler, num_workers=4, pin_memory=True)

#----evaluate the calibrated model---
criterion = nn.CrossEntropyLoss()
with torch.no_grad():
    evaluate(model, criterion, data_loader_test, device="cuda", print_freq=20)

