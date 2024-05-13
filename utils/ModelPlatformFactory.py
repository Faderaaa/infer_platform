from importlib import import_module


def ModelPlatformFactory(modelType: str = "hailo"):
    return getattr(import_module("modelsZoo." + modelType), 'Manage')
    # if modelType == 'hailo':
    #     print("platform is hailo")
    #     return getattr(import_module("middleware.HailoEntity"), 'Manage')
    # elif modelType == "rknn":
    #     print("platform is rknn")
    #     return getattr(import_module("middleware.RKEntity"), 'Manage')
    # elif modelType == "onnx":
    #     print("platform is onnx")
    #     return getattr(import_module("middleware.OnnxEntity"), 'Manage')
    # else:
    #     return getattr(import_module("utils.yolov5sOnnx"), 'Manage')
