from importlib import import_module


def ModelPlatformFactory(typeName: str = "hailo"):
    if typeName == 'hailo':
        print("platform is hailo")
        return getattr(import_module("middleware.HailoEntity"), 'HailoManage')
    elif typeName == "rknn":
        print("platform is rknn")
        return getattr(import_module("middleware.RKEntity"), 'HailoManage')
    elif typeName == "onnx":
        print("platform is onnx")
        return getattr(import_module("middleware.OnnxEntity"), 'HailoManage')
    else:
        pass
