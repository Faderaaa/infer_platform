import time
from multiprocessing import Process, Event
import onnx
import onnxruntime as ort
import cv2
import numpy as np
from scipy.spatial.distance import pdist
from sklearn import preprocessing
from utils.RTSPPush import RTSPPush


class InferModel:
    def __init__(self, model_path):
        onnx_model = onnx.load(model_path)
        try:
            onnx.checker.check_model(onnx_model)
        except Exception:
            print("Model incorrect")
        options = ort.SessionOptions()
        options.enable_profiling = True
        self.onnx_session = ort.InferenceSession(model_path)
        self.input_name = self.get_input_name()
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
        self.anchors = [[116, 90, 156, 198, 373, 326],
                        [30, 61, 62, 45, 59, 119],
                        [10, 13, 16, 30, 33, 23]
                        ]
        self.thed = 0.4

    def get_input_name(self):
        """获取输入节点名称"""
        input_name = []
        for node in self.onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, image_numpy):
        """获取输入numpy"""
        input_feed = {}
        for name in self.input_name:
            input_feed[name] = image_numpy
        return input_feed

    def inferData(self, frame):
        image = frame
        if image.shape[-1] != 160:
            image = cv2.resize(image, dsize=(160, 160), interpolation=cv2.INTER_CUBIC)
        img = image[:, :, ::-1].transpose(2, 0, 1)  # BGR2RGB和HWC2CHW
        img = img.astype(dtype=np.float32)  # onnx模型的类型是type: float32[ , , , ]
        img /= 255.0
        img = np.expand_dims(img, axis=0)  # [3, 640, 640]扩展为[1, 3, 640, 640]
        input_feed = self.get_input_feed(img.astype(np.float32))
        modelRes = self.onnx_session.run(None, input_feed)
        return modelRes


def imgStreamInfer(model_path, rtspUrl, event):
    model = InferModel(model_path)
    # print(rtspUrl)
    rtscap = cv2.VideoCapture(rtspUrl)
    width = int(rtscap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(rtscap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rtsp_p = rtspUrl + "/" + str(0)
    push = RTSPPush(width, height, rtsp_p)
    while not event.is_set():
        success, image = rtscap.read()
        img = model.inferData(image)
        push.pushData(img)
    rtscap.release()


def Manage(sendQ):
    """ Hailo计算卡管理线程(若移植则需要重写此功能)
        Args：
            q(Queue): 消息队列，用于和此线程进行数据交互
    """
    event = Event()
    inferRtsp = None
    while True:
        res = sendQ.get()  # 阻塞等待其他线程传来的数据
        if res["type"] == "startStream" and inferRtsp is None:
            inferRtsp = Process(target=imgStreamInfer, args=(res["modelName"], res["rstpUrl"], event))
            inferRtsp.start()
        elif res["type"] == "stopStream" and inferRtsp is not None:
            event.set()
            inferRtsp.join()
            inferRtsp = None
            event.clear()


if __name__ == "__main__":
    # TODO 完善第一阶段的
    onnx_path = '/home/hubu/Documents/hefs/facenet_mobilenet.onnx'
    model = InferModel(onnx_path)

    img1 = cv2.imread("/home/hubu/Documents/data/yolo_data/bus.jpg")
    outputs1 = model.inferData(img1)[0]

    img2 = cv2.imread("/home/hubu/Documents/data/yolo_data/test.jpg")
    outputs2 = model.inferData(img2)[0]

    l1 = np.linalg.norm(outputs1 - outputs2, axis=1)
    print("l1 %f" % l1)
    cosSim = 1 - pdist(np.vstack([outputs1, outputs2]), 'cosine')
    print("pdist %f" % cosSim)
    outputs1 = preprocessing.normalize(outputs1, norm='l2')
    outputs2 = preprocessing.normalize(outputs2, norm='l2')
    l1 = np.linalg.norm(outputs1 - outputs2, axis=1)
    print("after l2 l1 %f" % l1)
