import time

import cv2
import numpy as np
import requests

from utils.DataHandle import DataHandlePool

dataHandle = DataHandlePool()


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def loadHef():
    param = {
        "hef": "lprnet.hef"
    }
    res = requests.post(url="http://127.0.0.1:5000/device/loadHef", json=param)
    print(res.json())
    return res


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(z):
    # 计算指数
    exp_z = np.exp(z)
    # 计算softmax
    softmax_output = exp_z / np.sum(exp_z)
    return softmax_output


def analy(res):
    # 解析输出数据
    boxes = []
    conMax = 0
    # 定义类别数量和置信度阈值
    num_classes = 80
    for re in res:
        v = res[re]
        output_data = np.array(v)
        for i in range(output_data.shape[1]):
            for j in range(output_data.shape[2]):
                # 获取当前网格单元的预测结果
                predictions = output_data[0, i, j]

                # 遍历每个锚点
                for k in range(0, 3):
                    # 获取边界框信息和置信度得分
                    box = predictions[k * (4 + 1 + num_classes):(k + 1) * (4 + 1 + num_classes)]
                    x, y, w, h = box[:4]
                    confidence = box[4]

                    # 如果置信度大于阈值，则记录类别和坐标
                    if int(confidence) >= int(conMax):
                        class_probs = box[5:]
                        class_index = np.argmax(class_probs)
                        class_name = class_index + 1  # 类别索引从1开始

                        # 将类别和坐标添加到列表中
                        maxBox = (class_name, confidence, x, y, w, h)
                        boxes.append(maxBox)
                        conMax = confidence

    # 打印所有满足条件的方框的类别和坐标
    boxes = boxes[-10:]
    for box in boxes:
        print("Class: {}, con:{}, Coordinates: ({:.2f}, {:.2f}, {:.2f}, {:.2f})".format(*box))


def infer():
    image = cv2.imread('/home/hubu/Documents/data/COCO2017/val_final/val/000000002157.jpg')
    image = cv2.resize(image, dsize=(640, 640), interpolation=cv2.INTER_CUBIC)
    image = image.tobytes()
    files = {'file': image}
    params = {
        "modelName": "yolov5s_bck.hef"
    }
    res = requests.post(url="http://127.0.0.1:5000/device/inferSig", files=files, data=params)
    infer_results = res.json()['infer_results']
    if infer_results == "err stream is inprocessing":
        print(infer_results)
        return
    infer_results = eval(infer_results)  # 13, 13, 255
    for re in infer_results:
        print(re)


def startStream():
    params = {
        "modelName": "/home/hubu/Documents/hefs/yolov5s.onnx",
        # "rstpUrl": "rtsp://admin:123456@192.168.31.68:554/ch01.264"
        "rstpUrl": "rtsp://127.0.0.1:8554/chan1/sub/av_stream"
    }
    res = requests.post(url="http://127.0.0.1:5000/device/startStream", data=params)
    print(res.text)
    return res

def stopStream():
    res = requests.get(url="http://127.0.0.1:5000/device/stopStream")
    print(res.text)
    return res

def showRtsp():
    cap = cv2.VideoCapture("rtsp://127.0.0.1:8554/chan1/sub/av_stream/0")
    ret, frame = cap.read()
    while ret:
        ret, frame = cap.read()
        cv2.imshow("frame", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    # showRtsp()
    startStream()
    time.sleep(20)
    stopStream()
    # time.sleep(20)
    # startStream()
    # time.sleep(20)
    # stopStream()
    # loadHef()
    # infer()
