import subprocess

import numpy


class RTSPPush:
    def __init__(self, width, height, pushUrl):
        rtsp_p = "rtsp://127.0.0.1:8554/res/" + pushUrl
        command = ['ffmpeg',
                   '-y', '-an',
                   '-re',
                   '-f', 'rawvideo',
                   '-pix_fmt', 'bgr24',
                   '-s', str(width) + "x" + str(height),
                   '-i', '-',
                   '-c:v', 'libx264',
                   '-g', '1',
                   '-maxrate:v', '6M',
                   '-minrate:v', '2M',
                   '-bufsize:v', '4M',
                   '-pix_fmt', 'yuv420p',
                   '-preset', 'ultrafast',
                   '-tune', 'zerolatency',
                   '-f', 'rtsp',
                   rtsp_p]
        self.pipe = subprocess.Popen(command
                                     , shell=False
                                     , stdin=subprocess.PIPE
                                     )

    def pushData(self, data: numpy):
        self.pipe.stdin.write(data.tobytes())

    def relace(self):
        self.pipe.terminate()

