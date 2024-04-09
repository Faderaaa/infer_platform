import json
import time
from multiprocessing import Process, Queue, Manager
from flask import Flask, request, Response, render_template
import numpy as np

from middleware.mycelery import make_celery
from utils.DataHandle import DataHandlePool
from utils.ModelPlatformFactory import ModelPlatformFactory
from utils.RTSCapture import RTSCapture
from utils.VideoCamera import VideoCamera, gen
import requests

platformType = "hailo"

app = Flask(__name__, static_url_path='/templates', static_folder='templates')
# 分布式任务管理器，由于管理后台任务
celery = make_celery(app)

# 消息队列，用于同步flask线程与主线程
hmq = Queue()
hrq = Queue()

# 多线程共享Dict，用于流式推理时给计算卡上锁，防止冲突
manager = Manager()
hailoLock = manager.dict()

dataHandle = DataHandlePool()


# 后台长期推理任务（长期推理RTSP的视频流，并且处理其结果）
@celery.task(name="celery.backendInfer")
def backendInfer(modelName, rtsp_url):
    rtscap = RTSCapture.create(rtsp_url)
    rtscap.start_read()
    while rtscap.isStarted():
        ok, frame = rtscap.read_latest_frame()
        if frame is not None:
            img = frame.tobytes()
            params = {
                "modelName": modelName
            }
            files = {
                'file': img
            }
            infer_results = requests.post(url="http://127.0.0.1:5000/device/inferSig", files=files, data=params)
            if infer_results == "err stream is inprocessing":
                continue
            infer_results = infer_results.json()['infer_results']
            infer_results = eval(infer_results)
            res = dataHandle.reportWarningByInferResults(modelName, np.array(infer_results))
            print(res)
            time.sleep(1)
    rtscap.stop_read()
    rtscap.release()


# 实时浏览视频流
@app.route('/video_feed/<modelName>')
def video_feed(modelName="default"):
    if modelName == "default":
        modelName = None
    rstpUrl = request.args.get('rstpUrl')
    if hailoLock["isStreamInfer"]:
        hmq.put(-1, timeout=30)
    return Response(gen(VideoCamera(rstpUrl), modelName, hmq, hrq, dataHandle),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# 登录接口
@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data['username']
    password = data['password']
    if username == "admin" and password == "admin123":
        return json.dumps({
            "code": 20000,
            "message": "登录成功！"
        })
    return json.dumps({
        "code": 50000,
        "message": "用户名或密码错误"
    })


# 开启视频流任务
@app.route('/device/startStream', methods=['POST'])
def startStream():
    modelName = request.form.get("modelName")
    rstpUrl = request.form.get("rstpUrl")
    task = backendInfer.delay(modelName, rstpUrl)
    return task.id


# 手动停止视频流任务
@app.route('/device/stopStream', methods=['GET'])
def stopStream():
    try:
        if hailoLock["isStreamInfer"]:
            hmq.put(-1, timeout=30)
    except Exception:
        return json.dumps({
            "code": 50000,
            "message": "操作失败"
        })
    return json.dumps({
        "code": 20000,
        "message": "操作成功"
    })


# 单张图片推理
@app.route('/device/inferSig', methods=['POST'])
def inferSig():
    upload_file = request.files['file']
    modelName = request.form.get("modelName")
    file_bytes = upload_file.read()
    frame = np.frombuffer(file_bytes, dtype=np.uint8)
    if hailoLock["isStreamInfer"]:
        return {
            "infer_results": "err stream is inprocessing"
        }
    hmq.put({
        "modelName": modelName,
        "img": frame,
        "type": "infer"
    })
    infer_results = hrq.get()
    result = dataHandle.analyzeInferResults(modelName, infer_results).tolist()
    return {
        "infer_results": str(result)
    }


# 首页（登录页面）
@app.route('/', methods=['GET'])
def index():
    # jinja2模板，具体格式保存在index.html文件中
    # TODO 完善完整的项目登录等网页以及其功能
    return render_template('index.html')


@app.route('/home', methods=['GET'])
def homepage():
    return render_template('home.html')


@app.route('/viewInfer', methods=['GET'])
def viewInfer():
    modelName = request.args.get("modelName")
    rstpUrl = request.args.get("rstpUrl")
    content = {
        "modelName": modelName,
        "rstpUrl": rstpUrl
    }
    print(content)
    return render_template('viewInfer.html', **content)


if __name__ == '__main__':
    hailoLock["isStreamInfer"] = False
    # app启动，计算卡管理线程启动
    flas = Process(target=app.run, args=('0.0.0.0',))
    flas.start()
    HailoManage = ModelPlatformFactory(platformType)
    HailoManage(hmq, hrq, hailoLock)
