import time

import cv2

from utils.RTSCapture import RTSCapture


def gen(camera, modelName, q, rec, dataHandle):
    while True:
        frame = camera.get_frame(modelName, q, rec, dataHandle)
        # 使用generator函数输出视频流， 每次请求输出的content类型是image/jpeg
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


class VideoCamera(object):
    def __init__(self, rtspUrl):
        # 通过opencv获取实时视频流
        # self.video = RTSCapture(rtspUrl)
        self.video = cv2.VideoCapture(rtspUrl)

    def __del__(self):
        self.video.release()

    def get_frame(self, modelName=None, q=None, rec=None, dataHandle=None):
        start = time.time()
        success, image = self.video.read()
        if modelName is not None and q is not None:
            q.put({
                "modelName": modelName,
                "img": image,
                "type": "infer"
            }, timeout=30)
            infer_results = rec.get()
            fps = int(1 / (time.time() - start))
            image = dataHandle.drawPicture(modelName, image, infer_results, fps)
        # 因为opencv读取的图片并非jpeg格式，因此要用motion JPEG模式需要先将图片转码成jpg格式图片
        # print(int(1 / (time.time() - start)))
        bite = image.shape[0] / image.shape[1]
        image = cv2.resize(image, dsize=(1080, int(1080 * bite)), interpolation=cv2.INTER_CUBIC)
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
