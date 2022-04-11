import os
import sys
import numpy as np
import glob
import cv2
import time
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

#----reference
#----https://github.com/NVIDIA/object-detection-tensorrt-example/tree/master/SSD_Model/utils

def load_pts(a_pts_path):
    points = []
    for a_line in open(a_pts_path):
        a_line = a_line.strip()
        array = a_line.split()
        array = [float(x) for x in array]
        points.append(array)
    points = np.asarray(points)
    return points

def bgr2y(image):
    yuv = image.copy().astype(np.float32)
    yuv[:,:,0] = 0.299*image[:,:,2] + 0.587*image[:,:,1] + 0.114*image[:,:,0]
    yuv[:,:,1] = 0.492*(image[:,:,0] - yuv[:,:,0]) + 128
    yuv[:,:,2] = 0.877*(image[:,:,2] - yuv[:,:,0]) + 128
    yuv = np.clip(yuv, 0, 255)
    y = yuv[:, :, 0][:, :, np.newaxis]
    return y

def crop_image_f(img, points):
    img_height, img_width, _ = img.shape
    x1, y1 = points.min(axis=0)
    x2, y2 = points.max(axis=0)
    orig_x1 = int(x1); orig_y1 = int(y1)
    orig_x2 = int(x2); orig_y2 = int(y2)
    cx = (x1 + x2) / 2.0; cy = (y1 + y2) / 2.0
    box_w = x2 - x1 + 1
    box_h = y2 - y1 + 1
    box_max = max(box_w, box_h) #---find max length
    box_max = box_max * 1.3 #---max_length * 1.3
    half_box_max = box_max / 2.0
    x1 = int(cx - half_box_max)
    x2 = int(cx + half_box_max)
    y1 = int(cy - half_box_max)
    y2 = int(cy + half_box_max)
    delta_x1 = 0; delta_y1 = 0
    delta_x2 = 0; delta_y2 = 0
    if x1 < 0: delta_x1 = -x1
    if y1 < 0: delta_y1 = -y1
    if (x2 > img_width -1): delta_x2 = x2 - (img_width - 1)
    if (y2 > img_height - 1): delta_y2 = y2 - (img_height - 1)
    crop_width = x2 - x1 + 1
    crop_height = y2 - y1 + 1
    crop_image = np.zeros((crop_height, crop_width, 3), dtype=np.uint8)
    crop_image[delta_y1: crop_height-delta_y2, delta_x1: crop_width-delta_x2] = img[ y1 + delta_y1: y2+1 - delta_y2, x1 + delta_x1: x2+1 - delta_x2].copy()
    return crop_image, [x1, y1, x2, y2]

#------tensorrt helper data class
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

#------tensorrt load engine---
def load_engine(trt_runtime, engine_path):
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine

#------allocates all buffers required for an engine
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        #size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        print("engine.max_batch_size:\t", engine.max_batch_size)
        size = trt.volume(engine.get_binding_shape(binding)) 
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

#----tensorRT functions for input and output---
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

#----- model data----
class ModelData(object):
    # Name of input node
    INPUT_NAME = "Input"
    # CHW format of model input
    INPUT_SHAPE = (1, 128, 128)
    # Name of output node
    OUTPUT_NAME = "NMS"

    @staticmethod
    def get_input_channels():
        return ModelData.INPUT_SHAPE[0]

    @staticmethod
    def get_input_height():
        return ModelData.INPUT_SHAPE[1]

    @staticmethod
    def get_input_width():
        return ModelData.INPUT_SHAPE[2]

#---create a inference class
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TRTInference(object):
    def __init__(self, trt_engine_path, trt_engine_datatype=trt.DataType.FLOAT, batch_size=1):
        # some of them will be needed during inference
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')

        # Initialize runtime needed for loading TensorRT engine from file
        self.trt_runtime = trt.Runtime(TRT_LOGGER)
        # TRT engine placeholder
        self.trt_engine = None

        # Display requested engine settings to stdout
        print("TensorRT inference engine settings:")
        print("  * Inference precision - {}".format(trt_engine_datatype))
        print("  * Max batch size - {}\n".format(batch_size))

        # If we get here, the file with engine exists, so we can load it
        print("Loading cached TensorRT engine from {}".format(trt_engine_path))
        self.trt_engine = load_engine(
            self.trt_runtime,
            trt_engine_path
        )

        # This allocates memory for network inputs/outputs on both CPU and GPU
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.trt_engine)

        # Execution context is needed for inference
        self.context = self.trt_engine.create_execution_context()

        # Allocate memory for multiple usage [e.g. multiple batch inference]
        input_volume = trt.volume(ModelData.INPUT_SHAPE)
        self.numpy_array = np.zeros((self.trt_engine.max_batch_size, input_volume))

    def infer(self, a_image_path):
        """Infers model on given image.
        Args:
            image_path (str): image to run object detection model on
        """

        # Load image into CPU
        a_image_name = os.path.basename(a_image_path)

        a_pts_path = a_image_path[:-4] + ".pts"
        points = load_pts(a_pts_path)

        a_image = cv2.imread(a_image_path)
        a_image_copy = a_image.copy()
        crop_image, crop_cor = crop_image_f(a_image, points)
        crop_h, crop_w, _ = crop_image.shape
        crop_image = cv2.resize(crop_image, (128, 128))

        height, width, _ = a_image_copy.shape
        a_image_copy = cv2.resize(a_image_copy, (4*width, 4*height))

        crop_image_y = bgr2y(crop_image)
        crop_image_y = crop_image_y.transpose(2, 0, 1)
        crop_image_y = np.expand_dims(crop_image_y, axis=0) / 255.0 * 4

        # Copy it into appropriate place into memory
        np.copyto(self.inputs[0].host, crop_image_y.ravel())

        # When infering on single image, we measure inference
        # time to output it to the user
        inference_start_time = time.time()

        # Fetch output from the model
        output  = do_inference(
            self.context,
            bindings=self.bindings,
            inputs=self.inputs,
            outputs=self.outputs,
            stream=self.stream
        )

        # Output inference time
        print("TensorRT inference time: {} ms".format((time.time() - inference_start_time) * 1000))

        #----
        output = output[0]/64.0
        lines_vec = []
        print("output.shape:\t", output.shape)
        for i in range(98):
            x = output[2*i] * crop_w + crop_cor[0]
            y = output[2*i+1] * crop_h + crop_cor[1]
            a_line = str(x) + " " + str(y) + "\n"
            lines_vec.append(a_line)
            x = 4 * x; y = 4 * y
            color = (0, 255, 0)
            cv2.circle(a_image_copy, (int(x), int(y)), 3, color, -1)

        target_images_dir = "WFLW_predict_tensorRT"
        if not os.path.exists(target_images_dir):
            os.makedirs(target_images_dir)
        a_target_image_path = os.path.join(target_images_dir, a_image_name)
        cv2.imwrite(a_target_image_path, a_image_copy)

        a_target_pts_path = os.path.join(target_images_dir, a_image_name)[:-4] + ".pts"
        f = open(a_target_pts_path, "w")
        f.writelines(lines_vec)
        f.close()
        return output


if __name__=="__main__":
    images_dir = ""

    #--- tensorrt inference ----
    engine_path = "landmark_lite_model_0804.engine"
    trt_inference = TRTInference(engine_path)

    count = 0
    for a_image_path in glob.iglob(images_dir + "/*.jpg"):
        print(a_image_path)
        trt_inference.infer(a_image_path)
