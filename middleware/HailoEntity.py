import numpy as np
from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
                            InputVStreamParams, OutputVStreamParams, FormatType,
                            HailoSchedulingAlgorithm)


class ModelEntity:
    """ Hailo计算卡所使用的hef模型文件实体类(若移植则需要重写此功能)
    """

    def __init__(self, hef_path, target, input_stream_type=FormatType.AUTO,
                 output_stream_type=FormatType.AUTO):
        """Constructor for the ModelEntity class.
            Args:
                hef_path (str): 模型文件所在路径
                target (VDevice): 计算卡实例（一个计算卡同一时刻仅允许一个Process使用，且一旦使用会被上锁）
                input_stream_type(FormatType): 模型输入数据类型，默认为FormatType.AUTO
                output_stream_type(FormatType): 模型输出数据类型，默认为FormatType.AUTO
        """
        self.modelName = hef_path
        self.hef = HEF(hef_path)
        self.configure_params = ConfigureParams.create_from_hef(hef=self.hef, interface=HailoStreamInterface.PCIe)
        self.network_groups = target.configure(self.hef, self.configure_params)
        self.network_group = self.network_groups[0]
        self.network_group_params = self.network_group.create_params()

        # Create input and output virtual streams params
        self.input_vstreams_params = InputVStreamParams.make(self.network_group, format_type=input_stream_type)
        self.output_vstreams_params = OutputVStreamParams.make(self.network_group, format_type=output_stream_type)

        # Define dataset params
        self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
        self.output_vstream_info = []
        self.image_height, self.image_width, self.channels = self.input_vstream_info.shape
        self.modelRes = []
        for i in range(0, len(self.hef.get_output_vstream_infos())):
            output_vstream = self.hef.get_output_vstream_infos()[i]
            self.output_vstream_info.append(output_vstream)
            self.modelRes.append({
                "key": output_vstream.name,
                "shape": output_vstream.shape
            })


def inferSigImg(model, frame):
    """ 推理单张图片(若移植则需要重写此功能)
        Args：
            model(ModelEntity): 模型实例
            frame(numpy): 图片
        Returns:
            dict: Output tensors of all output layers. The keys are outputs names and the values
            are output data tensors as :obj:`numpy.ndarray` (or list of :obj:`numpy.ndarray` in case of nms output and tf_nms_format=False).
    """
    image = np.resize(frame, (1, model.image_height, model.image_width, model.channels))
    with InferVStreams(model.network_group, model.input_vstreams_params,
                       model.output_vstreams_params) as infer_pipeline:
        input_data = {model.input_vstream_info.name: image}
        infer_results = infer_pipeline.infer(input_data)
        res = []
        for output_info in model.modelRes:
            res.append(infer_results[output_info["key"]])
        return infer_results


def HailoManage(sendQ, recQ, hailoLock):
    """ Hailo计算卡管理线程(若移植则需要重写此功能)
        Args：
            q(Queue): 消息队列，用于和此线程进行数据交互
    """
    # 获得计算卡实例
    target = VDevice()
    lastName = ""
    # 载入一个默认模型
    model = ModelEntity("/home/hubu/Documents/hefs/lprnet.hef", target)
    while True:
        res = sendQ.get()  # 阻塞等待其他线程传来的数据
        if lastName != res['modelName']:
            model = ModelEntity("/home/hubu/Documents/hefs/" + res["modelName"], target)  # 实例化模型
            lastName = res["modelName"]
        if "infer" == res['type']:
            hailoLock["isStreamInfer"] = True
            with model.network_group.activate(model.network_group_params):
                infer = inferSigImg(model, res["img"])  # 推理
                recQ.put(infer)  # 传回数据
            hailoLock["isStreamInfer"] = False


if __name__ == '__main__':
    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
    target = VDevice()
    print("6")
