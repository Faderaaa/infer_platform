from hailo_sdk_client import ClientRunner, InferenceContext

import json
import os

import matplotlib.patches as patches
import numpy as np
import tensorflow as tf
from IPython.display import SVG
from matplotlib import pyplot as plt
from PIL import Image

IMAGES_TO_VISUALIZE = 5


# def getHar():
#     onnx_model_name = 'yolov8s'
#     onnx_path = '/home/hubu/Documents/hefs/yolov8s.onnx'
#     chosen_hw_arch = 'hailo8'
#     runner = ClientRunner(hw_arch=chosen_hw_arch)
#     runner.translate_onnx_model(onnx_path, onnx_model_name,
#                                           start_node_names=['images'],
#                                           end_node_names=['/model.22/Concat_3'],
#                                           net_input_shapes={'images': [1, 3, 640, 640]})
#
#     return runner


def preDataset():
    # First, we will prepare the calibration set. Resize the images to the correct size and crop them.
    from tensorflow.python.eager.context import eager_mode
    def preproc(image, output_height=640, output_width=640, resize_side=640):
        ''' imagenet-standard: aspect-preserving resize to 256px smaller-side, then central-crop to 224px'''
        with eager_mode():
            h, w = image.shape[0], image.shape[1]
            scale = tf.cond(tf.less(h, w), lambda: resize_side / h, lambda: resize_side / w)
            resized_image = tf.compat.v1.image.resize_bilinear(tf.expand_dims(image, 0),
                                                               [int(h * scale), int(w * scale)])
            cropped_image = tf.compat.v1.image.resize_with_crop_or_pad(resized_image, output_height, output_width)
            return tf.squeeze(cropped_image)

    images_path = '/home/hubu/Documents/data/yolo_data'
    images_list = [img_name for img_name in os.listdir(images_path) if
                   os.path.splitext(img_name)[1] == '.jpg']

    calib_dataset = np.zeros((len(images_list), 640, 640, 3))
    for idx, img_name in enumerate(sorted(images_list)):
        img = np.array(Image.open(os.path.join(images_path, img_name)))
        img_preproc = preproc(img)
        calib_dataset[idx, :, :, :] = img_preproc.numpy()

    np.save('calib_set.npy', calib_dataset)
    return calib_dataset


if __name__ == '__main__':
    onnx_model_name = 'yolov8s'
    onnx_path = '/home/hubu/Documents/hefs/yolov8s.onnx'
    chosen_hw_arch = 'hailo8'
    runner = ClientRunner(hw_arch=chosen_hw_arch)
    runner.translate_onnx_model(onnx_path, onnx_model_name,
                                start_node_names=['images'],
                                end_node_names=['/model.22/Concat_3'],
                                net_input_shapes={'images': [1, 3, 640, 640]})

    calib_dataset = preDataset()

    # model_name = 'yolov8s'
    # hailo_model_har_name = f'{model_name}_hailo_model.har'
    # assert os.path.isfile(hailo_model_har_name), 'Please provide valid path for HAR file'
    # runner = ClientRunner(har=hailo_model_har_name)

    alls = 'normalization1 = normalization([123.675, 116.28, 103.53], [58.395, 57.12, 57.375])\n'

    # Load the model script to ClientRunner so it will be considered on optimization
    runner.load_model_script(alls)

    # Call Optimize to perform the optimization process
    runner.optimize(calib_dataset)

    # Save the result state to a Quantized HAR file
    # quantized_model_har_path = f'{onnx_model_name}_quantized_model.har'
    # runner.save_har(quantized_model_har_path)

    hef = runner.compile()

    file_name = f'{onnx_model_name}_my.hef'
    with open(file_name, 'wb') as f:
        f.write(hef)
