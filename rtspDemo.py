import time
from multiprocessing import Process
import cv2
import threading
import numpy as np

from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
                            InputVStreamParams, OutputVStreamParams, InputVStreams, OutputVStreams, FormatType)


class StreamInfer:
    def __init__(self, model_name, input_stream_type=FormatType.FLOAT32, output_stream_type=FormatType.FLOAT32):
        self.target = VDevice()
        self.model_name = model_name
        hef_path = '../hefs/{}'.format(model_name)
        print("load model....", hef_path)
        self.hef = HEF(hef_path)

        # Configure network groups
        configure_params = ConfigureParams.create_from_hef(hef=self.hef, interface=HailoStreamInterface.PCIe)
        self.network_groups = self.target.configure(self.hef, configure_params)
        self.network_group = self.network_groups[0]
        self.network_group_params = self.network_group.create_params()

        # Create input and output virtual streams params
        self.input_vstreams_params = InputVStreamParams.make(self.network_group, format_type=input_stream_type)
        self.output_vstreams_params = OutputVStreamParams.make(self.network_group, format_type=output_stream_type)

        # Define dataset params
        self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
        self.output_vstream_info = []
        self.image_height, self.image_width, self.channels = self.input_vstream_info.shape
        print("input type: input_vstream_info.shape")

        print("outs type:")
        for i in range(0, len(self.hef.get_output_vstream_infos())):
            output_vstream = self.hef.get_output_vstream_infos()[i]
            self.output_vstream_info.append(output_vstream)
            print(output_vstream.shape)

    def startStreamInfer(self, rtspUrl, func):
        def send(configured_network):
            configured_network.wait_for_activation(1000)
            vstreams_params = InputVStreamParams.make(configured_network)
            with InputVStreams(configured_network, vstreams_params) as vstreams:
                rtscap = RTSCapture.create(rtspUrl)
                rtscap.start_read()
                while rtscap.isStarted():
                    ok, frame = rtscap.read_latest_frame()
                    image = np.resize(frame, (1, self.image_height, self.image_width, self.channels)).astype("float32")
                    vstream_to_buffer = {vstream: image for vstream in vstreams}
                    for vstream, buff in vstream_to_buffer.items():
                        vstream.send(buff)
                rtscap.stop_read()
                rtscap.release()

        def recv(configured_network, vstreams_params):
            configured_network.wait_for_activation(1000)
            with OutputVStreams(configured_network, vstreams_params) as vstreams:
                while True:
                    for vstream in vstreams:
                        data = vstream.recv()
                        func(data)

        def recv_all(configured_network):
            vstreams_params_groups = OutputVStreamParams.make_groups(configured_network)
            recv_procs = []
            for vstreams_params in vstreams_params_groups:
                proc = Process(target=recv, args=(configured_network, vstreams_params))
                proc.start()
                recv_procs.append(proc)
            for proc in recv_procs:
                proc.join()

        send_process = Process(target=send, args=(self.network_group,))
        recv_process = Process(target=recv_all, args=(self.network_group,))
        recv_process.start()
        send_process.start()
        print('Starting streaming (hef=\'{}\')'.format(self.model_name))
        with self.network_group.activate(self.network_group_params):
            send_process.join()
            recv_process.join()
        print('Done')

        self.target.release()


class HailoRT:
    def __init__(self, model_name, input_stream_type=FormatType.FLOAT32, output_stream_type=FormatType.FLOAT32):
        self.target = VDevice()
        self.model_name = model_name
        hef_path = '../hefs/{}'.format(model_name)
        print("load model....", hef_path)
        self.hef = HEF(hef_path)

        # Configure network groups
        configure_params = ConfigureParams.create_from_hef(hef=self.hef, interface=HailoStreamInterface.PCIe)
        self.network_groups = self.target.configure(self.hef, configure_params)
        self.network_group = self.network_groups[0]
        self.network_group_params = self.network_group.create_params()

        # Create input and output virtual streams params
        self.input_vstreams_params = InputVStreamParams.make(self.network_group, format_type=input_stream_type)
        self.output_vstreams_params = OutputVStreamParams.make(self.network_group, format_type=output_stream_type)

        # Define dataset params
        self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
        self.output_vstream_info = []
        self.image_height, self.image_width, self.channels = self.input_vstream_info.shape
        print("input type: input_vstream_info.shape")

        print("outs type:")
        print(len(self.hef.get_output_vstream_infos()))
        for i in range(0, len(self.hef.get_output_vstream_infos())):
            output_vstream = self.hef.get_output_vstream_infos()[i]
            self.output_vstream_info.append(output_vstream)
            print(output_vstream.shape)

    def inferYolo(self, frame):
        image = np.resize(frame, (1, self.image_height, self.image_width, self.channels)).astype("float32")
        # infer_results = self.forward(dataset)
        with InferVStreams(self.network_group, self.input_vstreams_params,
                           self.output_vstreams_params) as infer_pipeline:
            # inhere
            input_data = {self.input_vstream_info.name: image}
            with self.network_group.activate(self.network_group_params):
                infer_results = infer_pipeline.infer(input_data)
                return self.drawImg(infer_results, frame)

    def drawImg(self, infer_results, img):
        bbox = []
        for output_info in self.output_vstream_info:
            res = infer_results[output_info.name][0]
            # print('Stream output shape is {}'.format(res.shape))
            for w in range(0, output_info.shape[0]):
                for h in range(0, output_info.shape[1]):
                    re = res[w][h]
                    b = re[0:6]
                    m = re[6:12]
                    l = re[12:18]
                    num = 0.70
                    if b[-2] > num:
                        print(b)
                        bbox.append(b)
                    if m[-2] > num:
                        print(m)
                        bbox.append(m)
                    if l[-2] > num:
                        print(l)
                        bbox.append(l)
        # 绘制图片
        for box in bbox:
            img_h = img.shape[0]
            img_w = img.shape[1]

            x = box[0] * img_w
            y = box[1] * img_h
            w = box[2] * img_w
            h = box[3] * img_h
            # 左上
            pt1 = (int(x - w / 2), int(y - h / 2))
            # 右下
            pt2 = (int(x + w / 2), int(y + h / 2))

            cv2.rectangle(img, pt1, pt2, (0, 255, 0), 4)

        return img

    def inferYoloStream(self):
        print("infer--------------------------")

        def send(configured_network):
            configured_network.wait_for_activation(1000)
            vstreams_params = InputVStreamParams.make(configured_network)
            with InputVStreams(configured_network, vstreams_params) as vstreams:
                rtscap = RTSCapture.create("rtsp://admin:123456@192.168.31.68:554/ch01.264")
                rtscap.start_read()
                while rtscap.isStarted():
                    ok, frame = rtscap.read_latest_frame()
                    image = np.resize(frame, (1, self.image_height, self.image_width, self.channels)).astype("float32")
                    if cv2.waitKey(100) & 0xFF == ord('q'):
                        break
                    if not ok:
                        continue
                    vstream_to_buffer = {vstream: image for vstream
                                         in
                                         vstreams}
                    for vstream, buff in vstream_to_buffer.items():
                        vstream.send(buff)
                rtscap.stop_read()
                rtscap.release()

        def recv(configured_network, vstreams_params):
            configured_network.wait_for_activation(1000)
            counter = 0
            start_time = time.time()
            with OutputVStreams(configured_network, vstreams_params) as vstreams:
                while True:
                    res = []
                    for vstream in vstreams:
                        data = vstream.recv()
                        res.append(data)
                    # print("------------------")
                    # res = np.array(res)
                    # print(res.shape)

        def recv_all(configured_network):
            vstreams_params_groups = OutputVStreamParams.make_groups(configured_network)
            recv_procs = []
            print(len(vstreams_params_groups))
            for vstreams_params in vstreams_params_groups:
                proc = Process(target=recv, args=(configured_network, vstreams_params))
                proc.start()
                recv_procs.append(proc)
            for proc in recv_procs:
                proc.join()

        send_process = Process(target=send, args=(self.network_group,))
        recv_process = Process(target=recv_all, args=(self.network_group,))
        recv_process.start()
        send_process.start()
        print('Starting streaming (hef=\'{}\')'.format(self.model_name))
        with self.network_group.activate(self.network_group_params):
            send_process.join()
            recv_process.join()
        print('Done')

        self.target.release()


class RTSCapture(cv2.VideoCapture):
    _cur_frame = None
    _reading = False
    schemes = ["rtsp://", "rtmp://"]

    @staticmethod
    def create(url, *schemes):
        rtscap = RTSCapture(url)
        rtscap.frame_receiver = threading.Thread(target=rtscap.recv_frame, daemon=True)
        rtscap.schemes.extend(schemes)
        if isinstance(url, str) and url.startswith(tuple(rtscap.schemes)):
            rtscap._reading = True
        elif isinstance(url, int):
            pass
        return rtscap

    def isStarted(self):
        ok = self.isOpened()
        if ok and self._reading:
            ok = self.frame_receiver.is_alive()
        return ok

    def recv_frame(self):
        while self._reading and self.isOpened():
            ok, frame = self.read()
            if not ok: break
            self._cur_frame = frame
        self._reading = False

    def read2(self):
        frame = self._cur_frame
        self._cur_frame = None
        return frame is not None, frame

    def start_read(self):
        self.frame_receiver.start()
        self.read_latest_frame = self.read2 if self._reading else self.read

    def stop_read(self):
        self._reading = False
        if self.frame_receiver.is_alive(): self.frame_receiver.join()


def haha(data):
    print("haha")
    print(len(data))


if __name__ == '__main__':
    # model = HailoRT("yolov5m_vehicles.hef")
    # # 流式推理
    # model.inferYoloStream()

    model2 = StreamInfer("yolov5s_sigmoid_actived.hef")
    # 流式推理
    model2.startStreamInfer("rtsp://127.0.0.1:8554/chan1/sub/av_stream", haha)

    # # 单张推理
    # rtscap = RTSCapture.create("rtsp://admin:123456@192.168.31.68:554/ch01.264")
    # rtscap.start_read()
    # start_time = time.time()
    # counter = 0
    #
    # while rtscap.isStarted():
    #     ok, frame = rtscap.read_latest_frame()
    #     if cv2.waitKey(100) & 0xFF == ord('q'):
    #         break
    #     if not ok:
    #         continue
    #     # print(frame.shape)
    #     ori_shape = frame.shape
    #     # frame = model.inferYolo(frame)
    #     # frame.resize(ori_shape)
    #     # print(frame.shape)
    #     # cv2.imshow("cam", frame)
    #     counter += 1
    #     if (time.time() - start_time) != 0:  # 实时显示帧数
    #         print("FPS: ", counter / (time.time() - start_time))
    #         counter = 0
    #         start_time = time.time()
    #
    # rtscap.stop_read()
    # rtscap.release()
    # cv2.destroyAllWindows()
