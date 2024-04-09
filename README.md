# infer_platform
一个模型推理加速平台。目前支持Hailo计算卡、RKNN。
## 运行
由于采用了动态载入的方式，所以当需要切换计算卡或者RKNN时，需要在app.py中切换
platformType即可。若为'hailo'则为'hailo'计算卡，若为'rknn'则为npu计算。