import json
from multiprocessing import Process, Queue
from flask import Flask, request, render_template
from utils.ModelPlatformFactory import ModelPlatformFactory

modelType = "yolov5sHailo"

app = Flask(__name__, static_url_path='/templates', static_folder='templates')

# 消息队列，用于同步flask线程与主线程
hmq = Queue()


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
    hmq.put({
        "modelName": modelName,
        "rstpUrl": rstpUrl,
        "type": "startStream"
    })
    return json.dumps({
        "code": 20000,
        "message": "操作成功"
    })


# 手动停止视频流任务
@app.route('/device/stopStream', methods=['GET'])
def stopStream():
    hmq.put({
        "modelName": "",
        "rstpUrl": "",
        "type": "stopStream"
    })
    return json.dumps({
        "code": 20000,
        "message": "操作成功"
    })


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
    # app启动，计算卡管理线程启动
    flas = Process(target=app.run, args=('0.0.0.0',))
    flas.start()
    Manage = ModelPlatformFactory(modelType)
    Manage(hmq)
