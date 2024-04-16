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
    def dataPreprocessing(self, data: numpy.ndarray) -> numpy.ndarray:
        """
        数据预处理
        """
        pass

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
            if getattr(obj, 'getModelType')() == modelName.split("/")[-1]:
                return obj
        return None

    def dataPreprocessing(self, modelName: str, data: numpy.ndarray) -> numpy.ndarray:
        """
        数据预处理
        """
        obj = self.matchObj(modelName)
        assert obj is not None
        return getattr(obj, 'dataPreprocessing')(data)

    def analyzeInferResults(self, modelName: str, res) -> numpy.ndarray:
        """
        解析转换模型推理结果
        """
        obj = self.matchObj(modelName)
        assert obj is not None
        return getattr(obj, 'analyzeInferResults')(res)

    def drawPicture(self, modelName: str, img: numpy.ndarray, inferResults, fps: int) -> numpy.ndarray:
        """
        渲染图片
        """
        obj = self.matchObj(modelName)
        assert obj is not None
        return getattr(obj, 'drawPicture')(img, inferResults, fps)

    def reportWarningByInferResults(self, modelName: str, res) -> str:
        """
        上报警告，用于在后台任务推理时，处理模型推理结果
        """
        obj = self.matchObj(modelName)
        assert obj is not None
        return getattr(obj, 'reportWarningByInferResults')(res)


class yoloV5sHailoDataHandle(DataHandle):
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
        self.thed = 0.4

    def sigmoid(self, x):
        s = 1 / (1 + np.exp(-x))
        return s

    def getBox(self, inferResults):
        bbox = []
        acx = 2
        resList = []
        for res in inferResults:
            # res = inferResults[key][0]
            res = res[0]
            if res.shape[0] == 40:
                acx = 1
            if res.shape[0] == 20:
                acx = 0
            res_t = res.reshape(-1, 255)

            for a in range(0, 3):
                slice = res_t[:, 85 * a:85 * (a + 1)]
                res_ids = np.where((slice[:, 4]) > self.thed)[0]

                for res_id in res_ids:
                    now = slice[res_id]
                    ids = np.argmax(now[5:])
                    chosen_row = int(res_id / res.shape[0])
                    chosen_col = int(res_id % res.shape[0])
                    x, y, w, h = (now[:4])
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

    def dataPreprocessing(self, data: numpy.ndarray) -> numpy.ndarray:
        # data = np.resize(data.astype(np.float32), (1, 640, 640, 3))
        if data.shape[-1] != 640:
            data = cv2.resize(data, dsize=(640, 640), interpolation=cv2.INTER_CUBIC)
        img = data[:, :, ::-1]  # BGR2RGB和HWC2CHW
        img = img.astype(dtype=np.float32)  # 模型的类型是type: float32[ , , , ]
        # img /= 255.0
        img = np.expand_dims(img, axis=0)  # [3, 640, 640]扩展为[1, 3, 640, 640]
        return img

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
        return 'yolov5s.hef'

    def drawPicture(self, img: numpy.ndarray, inferResults, fps: int) -> numpy.ndarray:
        """
        渲染图片（将解析结果绘制到图片中，主要用于展示demo）
        """
        for box in inferResults:
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

                cl = int(box[0].item())
                score = box[1].item()
                cv2.rectangle(img, pt1, pt2, (255, 0, 0), 2)
                cv2.putText(img, '{0} {1:.2f}'.format(self.names[cl], score),
                            pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(img, str(fps), (100, 100), cv2.FONT_ITALIC, 2, (0, 255, 0), 5)
        img = numpy.uint8(img)
        return img

    def reportWarningByInferResults(self, res) -> str:
        """
        上报警告（用于在后台任务推理时，处理模型推理结果）
        """
        return "no Thing"


class yoloV5sOnnxDataHandle(DataHandle):
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
        self.thed = 0.4

    def xywh2xyxy(self, x):
        # [x, y, w, h] to [x1, y1, x2, y2]
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y
    def nms(self, dets, thresh):
        # dets:x1 y1 x2 y2 score class
        # x[:,n]就是取所有集合的第n个数据
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        # -------------------------------------------------------
        #   计算框的面积
        #	置信度从大到小排序
        # -------------------------------------------------------
        areas = (y2 - y1 + 1) * (x2 - x1 + 1)
        scores = dets[:, 4]
        # print(scores)
        keep = []
        index = scores.argsort()[::-1]  # np.argsort()对某维度从小到大排序
        # [::-1] 从最后一个元素到第一个元素复制一遍。倒序从而从大到小排序

        while index.size > 0:
            i = index[0]
            keep.append(i)
            # -------------------------------------------------------
            #   计算相交面积
            #	1.相交
            #	2.不相交
            # -------------------------------------------------------
            x11 = np.maximum(x1[i], x1[index[1:]])
            y11 = np.maximum(y1[i], y1[index[1:]])
            x22 = np.minimum(x2[i], x2[index[1:]])
            y22 = np.minimum(y2[i], y2[index[1:]])

            w = np.maximum(0, x22 - x11 + 1)
            h = np.maximum(0, y22 - y11 + 1)

            overlaps = w * h
            # -------------------------------------------------------
            #   计算该框与其它框的IOU，去除掉重复的框，即IOU值大的框
            #	IOU小于thresh的框保留下来
            # -------------------------------------------------------
            ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
            idx = np.where(ious <= thresh)[0]
            index = index[idx + 1]
        return keep

    def filter_box(self, org_box, conf_thres, iou_thres):  # 过滤掉无用的框
        # -------------------------------------------------------
        #   删除为1的维度
        #	删除置信度小于conf_thres的BOX
        # -------------------------------------------------------
        org_box = np.squeeze(org_box)  # 删除数组形状中单维度条目(shape中为1的维度)
        # (25200, 9)
        # […,4]：代表了取最里边一层的所有第4号元素，…代表了对:,:,:,等所有的的省略。此处生成：25200个第四号元素组成的数组
        conf = org_box[..., 4] > conf_thres  # 0 1 2 3 4 4是置信度，只要置信度 > conf_thres 的
        box = org_box[conf == True]  # 根据objectness score生成(n, 9)，只留下符合要求的框
        # -------------------------------------------------------
        #   通过argmax获取置信度最大的类别
        # -------------------------------------------------------
        cls_cinf = box[..., 5:]  # 左闭右开（5 6 7 8），就只剩下了每个grid cell中各类别的概率
        cls = []
        for i in range(len(cls_cinf)):
            cls.append(int(np.argmax(cls_cinf[i])))  # 剩下的objecctness score比较大的grid cell，分别对应的预测类别列表
        all_cls = list(set(cls))  # 去重，找出图中都有哪些类别
        # set() 函数创建一个无序不重复元素集，可进行关系测试，删除重复数据，还可以计算交集、差集、并集等。
        # -------------------------------------------------------
        #   分别对每个类别进行过滤
        #   1.将第6列元素替换为类别下标
        #	2.xywh2xyxy 坐标转换
        #	3.经过非极大抑制后输出的BOX下标
        #	4.利用下标取出非极大抑制后的BOX
        # -------------------------------------------------------
        output = []
        for i in range(len(all_cls)):
            curr_cls = all_cls[i]
            curr_cls_box = []
            curr_out_box = []
            for j in range(len(cls)):
                if cls[j] == curr_cls:
                    box[j][5] = curr_cls
                    curr_cls_box.append(box[j][:6])  # 左闭右开，0 1 2 3 4 5

            curr_cls_box = np.array(curr_cls_box)  # 0 1 2 3 4 5 分别是 x y w h score class
            curr_cls_box = self.xywh2xyxy(curr_cls_box)  # 0 1 2 3 4 5 分别是 x1 y1 x2 y2 score class
            curr_out_box = self.nms(curr_cls_box, iou_thres)  # 获得nms后，剩下的类别在curr_cls_box中的下标

            for k in curr_out_box:
                output.append(curr_cls_box[k])
        output = np.array(output)
        return output

    def getBox(self, inferResults):
        resList = self.filter_box(inferResults, self.thed, self.thed)
        return resList

    def dataPreprocessing(self, data: numpy.ndarray) -> numpy.ndarray:
        if data.shape[-1] != 640:
            data = cv2.resize(data, dsize=(640, 640), interpolation=cv2.INTER_CUBIC)
        img = data[:, :, ::-1].transpose(2, 0, 1)  # BGR2RGB和HWC2CHW
        img = img.astype(dtype=np.float32)  # onnx模型的类型是type: float32[ , , , ]
        img /= 255.0
        img = np.expand_dims(img, axis=0)  # [3, 640, 640]扩展为[1, 3, 640, 640]
        return img

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
        return 'yolov5s.onnx'

    def drawPicture(self, img: numpy.ndarray, inferResults, fps: int) -> numpy.ndarray:
        """
        渲染图片（将解析结果绘制到图片中，主要用于展示demo）
        """
        if inferResults.shape[0] != 0:
            boxes = inferResults[..., :4].astype(np.int32)  # x1 x2 y1 y2
            scores = inferResults[..., 4]
            classes = inferResults[..., 5].astype(np.int32)
            for box, score, cl in zip(boxes, scores, classes):
                top, left, right, bottom = box/640
                top, left, right, bottom = int(top * img.shape[1]), int(left * img.shape[0]),\
                                           int(right * img.shape[1]), int(bottom * img.shape[0])
                cv2.rectangle(img, (top, left), (right, bottom), (255, 0, 0), 2)
                cv2.putText(img, '{0} {1:.2f}'.format(self.names[cl], score),
                            (top, left), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img, str(fps), (100, 100), cv2.FONT_ITALIC, 2, (0, 255, 0), 5)
        return img

    def reportWarningByInferResults(self, res) -> str:
        """
        上报警告（用于在后台任务推理时，处理模型推理结果）
        """
        return "no Thing"