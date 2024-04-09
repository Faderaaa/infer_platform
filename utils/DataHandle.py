import random
from abc import ABC, abstractmethod

import cv2
import numpy
import numpy as np


class DataHandle(ABC):
    """
    定义 DataHandle 接口
    """

    @abstractmethod
    def analyzeInferResults(self, res) -> numpy.ndarray:
        """
        解析转换模型推理结果
        """
        pass

    @abstractmethod
    def getModelType(self) -> str:
        """
        获得当前算法的模型名称
        """
        pass

    @abstractmethod
    def drawPicture(self, img: numpy.ndarray, inferResults, fps: int) -> numpy.ndarray:
        """
        渲染图片
        """
        pass

    @abstractmethod
    def reportWarningByInferResults(self, res: numpy.ndarray) -> str:
        """
        上报警告，用于在后台任务推理时，处理模型推理结果
        """
        pass


class DataHandlePool:
    """
       数据处理池，利用反射机制自动实例化当前文件下继承自DataHandle抽象类的class，
       运行时根据modelName进行匹配自动调用具体实现方法的函数
    """

    def __init__(self):
        # 获取DataHandle的所有子类
        sub_class_list = DataHandle.__subclasses__()
        self.objList = []
        for i in range(len(sub_class_list)):
            # 获取子类的类名
            class_name = sub_class_list[i].__name__
            print(class_name)
            # 导入model模块
            model_module = __import__('utils')
            m = getattr(model_module, "DataHandle")
            obj_class_name = getattr(m, class_name)
            # 实例化对象
            obj = obj_class_name()
            self.objList.append(obj)

    def matchObj(self, modelName: str):
        for obj in self.objList:
            if getattr(obj, 'getModelType')() == modelName:
                return obj

    def analyzeInferResults(self, modelName: str, res) -> numpy.ndarray:
        """
        解析转换模型推理结果
        """
        obj = self.matchObj(modelName)
        return getattr(obj, 'analyzeInferResults')(res)

    def drawPicture(self, modelName: str, img: numpy.ndarray, inferResults, fps: int) -> numpy.ndarray:
        """
        渲染图片
        """
        obj = self.matchObj(modelName)
        return getattr(obj, 'drawPicture')(img, inferResults, fps)

    def reportWarningByInferResults(self, modelName: str, res) -> str:
        """
        上报警告，用于在后台任务推理时，处理模型推理结果
        """
        obj = self.matchObj(modelName)
        return getattr(obj, 'reportWarningByInferResults')(res)


class LprnetDataHandle(DataHandle):
    """
        实现 Lprnet的DataHandle 接口(Lprnet，车牌识别模型)
    """

    def analyzeInferResults(self, res) -> numpy.ndarray:
        """
        解析转换模型推理结果(仅仅返回解析后的结果，不会绘制进图片中，主要面向内部其他扩展功能使用)
        """
        res = res['lprnet/conv31']
        res = res.tolist()
        return res

    def getModelType(self) -> str:
        """
        获得当前算法的模型名称
        """
        return 'lprnet.hef'

    def drawPicture(self, img: numpy.ndarray, inferResults, fps: int) -> numpy.ndarray:
        """
        渲染图片（将解析结果绘制到图片中，主要用于展示demo）
        """
        print(inferResults)
        for key in inferResults:
            cv2.putText(img, "inferResults[0][0][0]:" + str(inferResults[key][0][0][0]), (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, "fps:" + str(fps), (200, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 1, cv2.LINE_AA)
        img = numpy.uint8(img)
        return img

    def reportWarningByInferResults(self, res) -> str:
        """
        上报警告（用于在后台任务推理时，处理模型推理结果）
        """
        return "no Thing"


class yoloV5sDataHandle(DataHandle):
    """
        实现 yoloV5s的DataHandle 接口(yoloV5)
    """

    def __init__(self):
        self.names = ["person"
            , "bicycle"
            , "car"
            , "motorcycle"
            , "airplane"
            , "bus"
            , "train"
            , "truck"
            , "boat"
            , "traffic light"
            , "fire hydrant"
            , "stop sign"
            , "parking meter"
            , "bench"
            , "bird"
            , "cat"
            , "dog"
            , "horse"
            , "sheep"
            , "cow"
            , "elephant"
            , "bear"
            , "zebra"
            , "giraffe"
            , "backpack"
            , "umbrella"
            , "handbag"
            , "tie"
            , "suitcase"
            , "frisbee"
            , "skis"
            , "snowboard"
            , "sports ball"
            , "kite"
            , "baseball bat"
            , "baseball glove"
            , "skateboard"
            , "surfboard"
            , "tennis racket"
            , "bottle"
            , "wine glass"
            , "cup"
            , "fork"
            , "knife"
            , "spoon"
            , "bowl"
            , "banana"
            , "apple"
            , "sandwich"
            , "orange"
            , "broccoli"
            , "carrot"
            , "hot dog"
            , "pizza"
            , "donut"
            , "cake"
            , "chair"
            , "couch"
            , "potted plant"
            , "bed"
            , "dining table"
            , "toilet"
            , "tv"
            , "laptop"
            , "mouse"
            , "remote"
            , "keyboard"
            , "cell phone"
            , "microwave"
            , "oven"
            , "toaster"
            , "sink"
            , "refrigerator"
            , "book"
            , "clock"
            , "vase"
            , "scissors"
            , "teddy bear"
            , "hair drier"
            , "toothbrush"]
        self.anchors = [[116, 90, 156, 198, 373, 326],
                        [30, 61, 62, 45, 59, 119],
                        [10, 13, 16, 30, 33, 23]
                        ]
        self.thed = 0.6

    def sigmoid(self, x):
        s = 1 / (1 + np.exp(-x))
        return s

    def getBox(self, inferResults):
        bbox = []
        acx = 2
        resList = []
        for key in inferResults:
            res = inferResults[key][0]
            if res.shape[0] == 40:
                acx = 1
            if res.shape[0] == 20:
                acx = 0
            res_t = res.reshape(-1, 255) / 255.0

            for a in range(0, 3):
                slice = res_t[:, 85 * a:85 * (a + 1)]
                res_ids = np.where(self.sigmoid(slice[:, 4]) > self.thed)[0]

                for res_id in res_ids:
                    now = slice[res_id]
                    ids = np.argmax(now[5:])
                    chosen_row = int(res_id / res.shape[0])
                    chosen_col = int(res_id % res.shape[0])
                    x, y, w, h = self.sigmoid(now[:4])
                    x = (x * 2.0 - 0.5 + chosen_col) / res.shape[1]
                    y = (y * 2.0 - 0.5 + chosen_row) / res.shape[1]
                    w = (2.0 * w) * (2.0 * w) * self.anchors[acx][a * 2] / 640
                    h = (2.0 * h) * (2.0 * h) * self.anchors[acx][a * 2 + 1] / 640
                    bbox.append((ids, slice[res_id][4], x, y, w, h))
            max_bbox = {}
            for box in bbox:
                if box[0] not in max_bbox.keys() or box[1] > max_bbox[box[0]][1]:
                    max_bbox[box[0]] = box
            for keyName in max_bbox:
                resList.append(max_bbox[keyName])
        return resList

    def analyzeInferResults(self, res) -> numpy.ndarray:
        """
        解析转换模型推理结果(仅仅返回解析后的结果，不会绘制进图片中，主要面向内部其他扩展功能使用)
        """
        res = self.getBox(res)
        return np.array(res)

    def getModelType(self) -> str:
        """
        获得当前算法的模型名称
        """
        return 'yolov5s_bck.hef'

    def drawPicture(self, img: numpy.ndarray, inferResults, fps: int) -> numpy.ndarray:
        """
        渲染图片（将解析结果绘制到图片中，主要用于展示demo）
        """
        bbox = self.getBox(inferResults)
        for box in bbox:
            if box[1] > 0.5:
                img_h = img.shape[0]
                img_w = img.shape[1]
                x = box[2] * img_w
                y = box[3] * img_h
                w = box[4] * img_w
                h = box[5] * img_h
                # 左上
                pt1 = (int(x - w / 2), int(y - h / 2))
                # 右下
                pt2 = (int(x + w / 2), int(y + h / 2))
                random_int = box[0] / 80 * 255
                cv2.rectangle(img, pt1, pt2, (0, random_int, 0), 4)
                cv2.putText(img, self.names[box[0]] + ":" + str(box[1]), pt1, cv2.FONT_ITALIC, 2, (0, random_int, 0), 5)
        cv2.putText(img, str(fps), (100, 100), cv2.FONT_ITALIC, 2, (0, 255, 0), 5)
        img = numpy.uint8(img)
        return img

    def reportWarningByInferResults(self, res) -> str:
        """
        上报警告（用于在后台任务推理时，处理模型推理结果）
        """
        return "no Thing"
