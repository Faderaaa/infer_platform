import numpy as np
from rknnlite.api import RKNNLite
from multiprocessing import Process, Queue


class ModelEntity:
    """ Hailo计算卡所使用的hef模型文件实体类(若移植则需要重写此功能)
    """

    def __init__(self, hef_path, target):
        """Constructor for the ModelEntity class.
            Args:
                hef_path (str): 模型文件所在路径
                target (VDevice): 计算卡实例（一个计算卡同一时刻仅允许一个Process使用，且一旦使用会被上锁）
        """
        self.modelName = hef_path
        self.hef = target.load_rknn(hef_path)
        self.target = target
        self.image_height, self.image_width, self.channels = (640, 640, 3)
        self.names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                      "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                      "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                      "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                      "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
                      "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                      "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
                      "bed",
                      "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                      "microwave",
                      "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
                      "hair drier",
                      "toothbrush"]


def inferSigImg(model, frame):
    """ 推理单张图片(若移植则需要重写此功能)
        Args：
            model(ModelEntity): 模型实例
            frame(numpy): 图片
        Returns:
            dict: Output tensors of all output layers. The keys are outputs names and the values
            are output data tensors as :obj:`numpy.ndarray` (or list of :obj:`numpy.ndarray` in case of nms output and tf_nms_format=False).
    """
    image = np.resize(frame, (1, model.image_height, model.image_width, model.channels)).astype("float32")
    outputs = model.target.inference(inputs=[image])
    return outputs


def HailoManage(sendQ, recQ, streamLock):
    """ Hailo计算卡管理线程(若移植则需要重写此功能)
        Args：
            q(Queue): 消息队列，用于和此线程进行数据交互
    """    
    # 获得计算卡实例
    target = RKNNLite()
    lastName = ""
    # 载入一个默认模型
    model = ModelEntity("/root/Desktop/yolo/rknn/resnet18_for_rk3588.rknn", target)
    while True:
        res = sendQ.get()  # 阻塞等待其他线程传来的数据
        if lastName != res['modelName']:
            model = ModelEntity("/root/Desktop/yolo/rknn/" + res["modelName"], target)  # 实例化模型
            target.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
            lastName = res["modelName"]
        if "stream" == res['type']:
            streamLock["isStreamInfer"] = True
            infer = inferSigImg(model, res["img"])  # 推理
            recQ.put(infer)  # 传回数据
            streamLock["isStreamInfer"] = False


if __name__ == '__main__':
    target = RKNNLite()
    print("6")