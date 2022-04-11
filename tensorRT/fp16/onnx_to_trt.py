import os
import numpy
import tensorrt as trt #--version 8.0.1.6, eb5de99

onnx_model_path = "landmark_lite_model_0804_0411.onnx"
TRT_LOGGER = trt.Logger()
trt_runtime = trt.Runtime(TRT_LOGGER)

if __name__=="__main__":
    with trt.Builder(TRT_LOGGER) as builder:
        network = builder.create_network(1) 
        parser = trt.OnnxParser(network, TRT_LOGGER)
        #--解析onnx文件
        with open(onnx_model_path, 'rb') as model:
            parser.parse(model.read())

        config = builder.create_builder_config()
        config.max_workspace_size=1 << 30 #--预先分配工作空间大小，即CudaEngine执行GPU时最大需要的空间

        builder.max_batch_size = 1 #--执行时最大可以使用的batch_size
        #---下面的语句和pytorch模型怎么转成onnx有关系
        #profile = builder.create_optimization_profile()
        #profile.set_shape("input", (1, 1, 128, 128), (1, 1, 128, 128), (1, 1, 128, 128))
        #config.add_optimization_profile(profile)

        engine = builder.build_serialized_network(network, config)
        engine_file_path = "landmark_lite_model_0804.engine"
        with open(engine_file_path, 'wb') as f:
            f.write(engine) #--序列化

        
