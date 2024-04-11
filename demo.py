import random
import time

import numpy as np
from multiprocessing import Process
from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
                            InputVStreamParams, OutputVStreamParams, InputVStreams, OutputVStreams, FormatType)
import cv2

# The target can be used as a context manager ("with" statement) to ensure it's released on time.
# Here it's avoided for the sake of simplicity
target = VDevice()

names = ["person"
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


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def softmax(x):
    """Compute the softmax of vector x."""
    # 计算指数值
    exp_x = np.exp(x)
    # 计算softmax值
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


def normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data


# Loading compiled HEFs to device: https://github.com/hailo-ai/hailo_model_zoo/blob/master/hailo_models
# /vehicle_detection/README.rst model_name = 'yolov5m_vehicles.hef'
model_name = 'yolov5s.hef'
hef_path = '../hefs/{}'.format(model_name)
print(hef_path)
hef = HEF(hef_path)

# Configure network groups
configure_params = ConfigureParams.create_from_hef(hef=hef, interface=HailoStreamInterface.PCIe)
network_groups = target.configure(hef, configure_params)
network_group = network_groups[0]
network_group_params = network_group.create_params()

# model_name2 = 'lprnet.hef'
# hef_path2 = '../hefs/{}'.format(model_name2)
# print(hef_path2)
# hef2 = HEF(hef_path2)
#
# # Configure network groups
# configure_params2 = ConfigureParams.create_from_hef(hef=hef2, interface=HailoStreamInterface.PCIe)
# network_groups2 = target.configure(hef2, configure_params2)
# network_group2 = network_groups2[0]
# network_group_params2 = network_group2.create_params()

# Create input and output virtual streams params
input_vstreams_params = InputVStreamParams.make(network_group, format_type=FormatType.AUTO)
output_vstreams_params = OutputVStreamParams.make(network_group, format_type=FormatType.AUTO)

# Define dataset params
input_vstream_info = hef.get_input_vstream_infos()[0]
output_vstream_info = []
image_height, image_width, channels = input_vstream_info.shape

print("in:")
print(input_vstream_info.name)
print(input_vstream_info.shape)
print("out:")
print(len(hef.get_output_vstream_infos()))
for i in range(0, len(hef.get_output_vstream_infos())):
    output_vstream = hef.get_output_vstream_infos()[i]
    output_vstream_info.append(output_vstream)
    print(output_vstream.shape)
    print(output_vstream.name)

# 读取图像
image = cv2.imread('/home/hubu/Documents/data/yolo_data/test.jpg')
image = cv2.resize(image, dsize=(640, 640), interpolation=cv2.INTER_CUBIC)
# 将图像转换为ndarray
# print(image.shape)
dataset = np.resize(image, (1, image_height, image_width, channels))

with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
    input_data = {input_vstream_info.name: dataset}
    with network_group.activate(network_group_params):
        infer_results = infer_pipeline.infer(input_data)
        bbox = []
        anchors = [[116, 90, 156, 198, 373, 326],
                   [30, 61, 62, 45, 59, 119],
                   [10, 13, 16, 30, 33, 23]
                   ]
        start = time.time()
        for output_info in output_vstream_info:
            acx = 2
            res = infer_results[output_info.name][0]  # 80， 80， 255
            if res.shape[0] == 40:
                acx = 1
            if res.shape[0] == 20:
                acx = 0
            res_t = res.reshape(-1, 255) / 255.0
            # print("spend time is {}".format(time.time() - start))
            for a in range(0, 3):
                slice = res_t[:, 85 * a:85 * (a + 1)]
                res_ids = np.where(sigmoid(slice[:, 4]) > 0.65)[0]
                # print("spend time is {}".format(time.time() - start))
                for res_id in res_ids:
                    now = slice[res_id]
                    ids = np.argmax(now[5:])
                    chosen_row = int(res_id / res.shape[0])
                    chosen_col = int(res_id % res.shape[0])
                    x, y, w, h = sigmoid(now[:4])
                    x = (x * 2.0 - 0.5 + chosen_col) / res.shape[1]
                    y = (y * 2.0 - 0.5 + chosen_row) / res.shape[1]
                    w = (2.0 * w) * (2.0 * w) * anchors[acx][a * 2] / 640
                    h = (2.0 * h) * (2.0 * h) * anchors[acx][a * 2 + 1] / 640
                    bbox.append((ids, slice[res_id][4], x, y, w, h))

            # for row in range(0, len(res)):
            #     for col in range(0, len(res[row])):
            #         prob_max = 0
            #         for a in range(0, 3):
            #             now = (res[row][col][85 * a:85 * (a + 1)]) / 255.0
            #             # now = res[row][col][85 * a:85 * (a + 1)]
            #             confidence = sigmoid(now[4])
            #             idx = now[5:] * confidence
            #
            #             id = np.argmax(idx)
            #             conf_max = sigmoid(now[5:][id]) * confidence
            #             chosen_cls = id
            #             anchor = a
            #             chosen_row = row
            #             chosen_col = col
            #         if conf_max >= 0.45:
            #             x, y, w, h = sigmoid(now[:4])
            #             x = (x * 2.0 - 0.5 + chosen_col) / res.shape[1]
            #             y = (y * 2.0 - 0.5 + chosen_row) / res.shape[1]
            #             w = (2.0 * w) * (2.0 * w) * anchors[acx][anchor * 2] / 640
            #             h = (2.0 * h) * (2.0 * h) * anchors[acx][anchor * 2 + 1] / 640
            #             bbox.append((chosen_cls, conf_max, x, y, w, h))
        # 绘制图片
        max_bbox = {}
        for box in bbox:
            if box[0] not in max_bbox.keys() or box[1] > max_bbox[box[0]][1]:
                max_bbox[box[0]] = box

        for key in max_bbox:
            box = max_bbox[key]
            print(*box)
            img_h = image.shape[0]
            img_w = image.shape[1]
            x = box[2] * img_w
            y = box[3] * img_h
            w = box[4] * img_w
            h = box[5] * img_h
            # 左上
            pt1 = (int(x - w / 2), int(y - h / 2))
            # 右下
            pt2 = (int(x + w / 2), int(y + h / 2))
            random_int = random.randint(0, 255)
            # cv2.rectangle(image, (int(x + ), y), (w, h), (0, random_int, 0), 4)
            cv2.rectangle(image, pt1, pt2, (0, random_int, 0), 4)
            cv2.putText(image, names[box[0]], pt1, cv2.FONT_ITALIC, 2, (0, random_int, 0), 5)
        # print("spend time is {}".format(time.time() - start))
        cv2.imwrite('/home/hubu/Documents/data/666.jpg', image)


def send(configured_network, num_frames):
    configured_network.wait_for_activation(1000)
    vstreams_params = InputVStreamParams.make(configured_network)
    with InputVStreams(configured_network, vstreams_params) as vstreams:
        vstream_to_buffer = {vstream: np.ndarray([1] + list(vstream.shape), dtype=vstream.dtype) for vstream in
                             vstreams}
        for _ in range(num_frames):
            for vstream, buff in vstream_to_buffer.items():
                vstream.send(buff)


def recv(configured_network, vstreams_params, num_frames):
    configured_network.wait_for_activation(1000)
    with OutputVStreams(configured_network, vstreams_params) as vstreams:
        for _ in range(num_frames):
            for vstream in vstreams:
                data = vstream.recv()
                # print(data)
                # print("*****************")


def recv_all(configured_network, num_frames):
    vstreams_params_groups = OutputVStreamParams.make_groups(configured_network)
    recv_procs = []
    for vstreams_params in vstreams_params_groups:
        proc = Process(target=recv, args=(configured_network, vstreams_params, num_frames))
        proc.start()
        recv_procs.append(proc)
    for proc in recv_procs:
        proc.join()


num_of_frames = 100

send_process = Process(target=send, args=(network_group, num_of_frames))
recv_process = Process(target=recv_all, args=(network_group, num_of_frames))
recv_process.start()
send_process.start()
print('Starting streaming (hef=\'{}\', num_of_frames={})'.format(model_name, num_of_frames))
with network_group.activate(network_group_params):
    send_process.join()
    recv_process.join()
print('Done')

target.release()
