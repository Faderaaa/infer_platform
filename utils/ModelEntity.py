import numpy as np
from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
                            InputVStreamParams, OutputVStreamParams, FormatType,
                            HailoSchedulingAlgorithm, InputVStreams, OutputVStreams)
from multiprocessing import Process, Queue


class ModelEntity:
    """ Hailo计算卡所使用的hef模型文件实体类(若移植则需要重写此功能)
    """

    def __init__(self, hef_path, target, input_stream_type=FormatType.AUTO,
                 output_stream_type=FormatType.AUTO):
        """Constructor for the ModelEntity class.
            Args:
                hef_path (str): 模型文件所在路径
                target (VDevice): 计算卡实例（一个计算卡同一时刻仅允许一个Process使用，且一旦使用会被上锁）
                input_stream_type(FormatType): 模型输入数据类型，默认为FormatType.AUTO
                output_stream_type(FormatType): 模型输出数据类型，默认为FormatType.AUTO
        """
        self.modelName = hef_path
        self.hef = HEF(hef_path)
        self.configure_params = ConfigureParams.create_from_hef(hef=self.hef, interface=HailoStreamInterface.PCIe)
        self.network_groups = target.configure(self.hef, self.configure_params)
        self.network_group = self.network_groups[0]
        self.network_group_params = self.network_group.create_params()

        # Create input and output virtual streams params
        self.input_vstreams_params = InputVStreamParams.make(self.network_group, format_type=input_stream_type)
        self.output_vstreams_params = OutputVStreamParams.make(self.network_group, format_type=output_stream_type)

        # Define dataset params
        self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
        self.output_vstream_info = []
        self.image_height, self.image_width, self.channels = self.input_vstream_info.shape
        self.modelRes = []
        for i in range(0, len(self.hef.get_output_vstream_infos())):
            output_vstream = self.hef.get_output_vstream_infos()[i]
            self.output_vstream_info.append(output_vstream)
            self.modelRes.append({
                "key": output_vstream.name,
                "shape": output_vstream.shape
            })


def inferSigImg(model, frame):
    """ 推理单张图片(若移植则需要重写此功能)
        Args：
            model(ModelEntity): 模型实例
            frame(numpy): 图片
        Returns:
            dict: Output tensors of all output layers. The keys are outputs names and the values
            are output data tensors as :obj:`numpy.ndarray` (or list of :obj:`numpy.ndarray` in case of nms output and tf_nms_format=False).
    """
    image = np.resize(frame, (1, model.image_height, model.image_width, model.channels))
    with InferVStreams(model.network_group, model.input_vstreams_params,
                       model.output_vstreams_params) as infer_pipeline:
        input_data = {model.input_vstream_info.name: image}
        infer_results = infer_pipeline.infer(input_data)
        res = []
        for output_info in model.modelRes:
            res.append(infer_results[output_info["key"]])
        return infer_results


def send(model, sq):
    """ 向计算卡发送数据(若移植则需要重写此功能)
            Args：
                model(ModelEntity): 模型实例
                sq(Queue): 消息队列，用于接受到图片后转发给计算卡
    """
    vstreams_params = InputVStreamParams.make(model.network_group)
    with InputVStreams(model.network_group, vstreams_params) as vstreams:
        while True:
            try:
                frame = sq.get(False)
                # image = frame
                image = np.resize(frame['img'], (1, model.image_height, model.image_width, model.channels))
                vstream_to_buffer = {vstream: image for vstream in vstreams}
                for vstream, buff in vstream_to_buffer.items():
                    vstream.send(buff)
            except:
                continue


def recv(model, vstreams_params, rq, i):
    """ 接收计算卡的数据(若移植则需要重写此功能)
            Args：
                model(ModelEntity): 模型实例
                sq(Queue): 消息队列，用于接受到计算卡推理结果后通知给外面的线程
    """
    with OutputVStreams(model.network_group, vstreams_params) as vstreams:
        while True:
            try:
                for vstream in vstreams:
                    data = vstream.recv()
                rq.put(data)
            except:
                break


def recv_all(model, rq):
    vstreams_params_groups = OutputVStreamParams.make_groups(model.network_group)
    recv_procs = []
    i = 0
    q_list = []
    for vstreams_params in vstreams_params_groups:
        q_x = Queue()
        proc = Process(target=recv, args=(model, vstreams_params, q_x, i))
        proc.start()
        recv_procs.append(proc)
        i = i + 1
    while True:
        res = []
        for q in q_list:
            res.append(q.get())
        rq.put(np.array(res))
    # for proc in recv_procs:
    #     proc.join()


def clear_queue(queue):
    """ 清空队列
            Args：
                queue(Queue): 被清空的消息队列
    """
    while not queue.empty():
        queue.get()


def streamInfor(model, sendQ, recQ):
    """ 流式推理，创建一收一发两个线程，然后通过消息队列进行数据转发。
            Args：
                model(ModelEntity): 模型实例
                sendQ(Queue): 消息队列，用于接受到图片后转发给计算卡
                recQ(Queue): 消息队列，用于接受到计算卡推理结果后通知给外面的线程
    """
    s = Queue()
    r = Queue()
    send_thread = Process(target=send, args=(model, s), name=model.modelName + "send")
    rec_thread = Process(target=recv_all, args=(model, r), name=model.modelName + "rec1")
    # 启动新进程
    rec_thread.start()
    send_thread.start()
    while True:
        try:
            fm = sendQ.get(block=True, timeout=30)
            assert fm != -1
            s.put(fm)
            try:
                res = r.get(False)
                recQ.put(res)
            except:
                pass
        except:
            send_thread.terminate()
            rec_thread.terminate()
            clear_queue(sendQ)
            clear_queue(recQ)
            break
    clear_queue(recQ)
    print("out stream")


def HailoManage(sendQ, recQ, streamLock):
    """ Hailo计算卡管理线程(若移植则需要重写此功能)
        Args：
            q(Queue): 消息队列，用于和此线程进行数据交互
    """
    # 获得计算卡实例
    target = VDevice()
    lastName = ""
    # 载入一个默认模型
    model = ModelEntity("/home/hubu/Documents/hefs/lprnet.hef", target)
    while True:
        res = sendQ.get()  # 阻塞等待其他线程传来的数据
        if lastName != res['modelName']:
            model = ModelEntity("/home/hubu/Documents/hefs/" + res["modelName"], target)  # 实例化模型
            lastName = res["modelName"]
        if "stream" == res['type']:
            streamLock["isStreamInfer"] = True
            with model.network_group.activate(model.network_group_params):
                streamInfor(model, sendQ, recQ)
                lastName = ""  # 清空一下模型名称，方便下次推理前重新实例化（否则计算卡驱动会报错出bug，同一个模型不可推理完流式后再单张推理）
            streamLock["isStreamInfer"] = False
        else:
            with model.network_group.activate(model.network_group_params):
                infer = inferSigImg(model, res["img"])  # 推理
                recQ.put(infer)  # 传回数据


if __name__ == '__main__':
    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
    target = VDevice()
    print("6")
