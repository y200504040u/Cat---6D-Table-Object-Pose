# coding: utf-8

from pyorbbecsdk import *
import json
import cv2
import numpy as np
from utils import frame_to_bgr_image
import argparse
from run_obj_pose import GetObjPose
 

list_color = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255), (0,153,255), (153,0,255)]

class cGetObjPose:
    def __init__(self, opts):
        self.opts = opts
        self.debug = self.opts.debug
        path_trt1 = "./data/checkpoints/seg.trt"
        path_trt2 = "./data/checkpoints/vec.trt"
        self.cObjPose = GetObjPose(path_trt1, path_trt2)

        self.pipeline = Pipeline()
        config = Config()
        
        align_mode = 'HW'
        enable_sync = True
        try:
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            # color_profile = profile_list.get_default_video_stream_profile()
            color_profile: VideoStreamProfile = profile_list.get_video_stream_profile(640, 480, OBFormat.RGB, 30)
            config.enable_stream(color_profile)
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            assert profile_list is not None
            # depth_profile = profile_list.get_default_video_stream_profile()
            depth_profile = profile_list.get_video_stream_profile(640, 0, OBFormat.Y16, 30)
            assert depth_profile is not None
            print("color profile : {}x{}@{}_{}".format(color_profile.get_width(),
                                                    color_profile.get_height(),
                                                    color_profile.get_fps(),
                                                    color_profile.get_format()))
            print("depth profile : {}x{}@{}_{}".format(depth_profile.get_width(),
                                                    depth_profile.get_height(),
                                                    depth_profile.get_fps(),
                                                    depth_profile.get_format()))
            config.enable_stream(depth_profile)
        except Exception as e:
            print(e)
            return
        if align_mode == 'HW':
            config.set_align_mode(OBAlignMode.HW_MODE)
        elif align_mode == 'SW':
            config.set_align_mode(OBAlignMode.SW_MODE)
        else:
            config.set_align_mode(OBAlignMode.DISABLE)
        if enable_sync:
            try:
                self.pipeline.enable_frame_sync()
            except Exception as e:
                print(e)
        try:
            self.pipeline.start(config)
        except Exception as e:
            print(e)
            return
        
    def Run(self, flagUseSocket=True):
        if not flagUseSocket:
            while True:
                frames: FrameSet = self.pipeline.wait_for_frames(100)
                if frames is None:
                    continue
                color_frame = frames.get_color_frame()
                if color_frame is None:
                    continue
                # covert to RGB format
                img_bgr = frame_to_bgr_image(color_frame)
                if img_bgr is None:
                    print("failed to convert frame to image")
                    continue
                depth_frame = frames.get_depth_frame()
                if depth_frame is None:
                    continue

                width = depth_frame.get_width()
                height = depth_frame.get_height()

                depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                depth_data = depth_data.reshape((height, width))
                # dep4show = (depth_data * 0.0625).astype(np.uint8)

                dep16 = np.asanyarray(depth_data, np.uint16)

                start_time_est = time.time()
                out_put = self.cObjPose.Run(img_bgr, dep16)
                end_time_est = time.time()
                time_est = end_time_est - start_time_est
                print("Est time / frame: {:.2f} ms".format(1000 * time_est))

                result = {'type': 'rgbd', 'cls_pose': out_put}
                result = json.dumps(result, ensure_ascii=False)

        else:
            # ------------- socket --------------
            import socket
            import sys
            import time

            # 获取本地ip
            # serversocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # serversocket.connect(('8.8.8.8', 80))
            # host_ip = serversocket.getsockname()[0]
            # print("host_ip=", host_ip)
            # 创建 socket 对象, 绑定端口号, 设置最大连接数
            serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            serversocket.bind(('127.0.0.1', self.opts.ip_port))
            serversocket.listen(5)  # 超过后排队
            # ------------- end -----------------

            while True:
                result = {'type': 'rgbd', 'cls_pose': [], 'errorcode': '0', 'time': 0}
                
                try:
                    # 建立客户端连接
                    print("Rgbd is waiting for client...")
                    clientsocket = None
                    clientsocket, addr = serversocket.accept()
                    print("连接地址: %s" % str(addr))

                    get_frame = True
                    frames: FrameSet = self.pipeline.wait_for_frames(100)
                    if frames is None:
                        get_frame = False
                        print("Getting rgbd frames failed !!!")
                    else:
                        color_frame = frames.get_color_frame()
                        if color_frame is None:
                            get_frame = False
                            print("Getting rgb frames failed !!!")
                        else:
                            # covert to RGB format
                            img_bgr = frame_to_bgr_image(color_frame)
                            if img_bgr is None:
                                get_frame = False
                                print("Covert to RGB format failed !!!")
                            else:
                                depth_frame = frames.get_depth_frame()
                                if depth_frame is None:
                                    get_frame = False
                                    print("Getting depth frames failed !!!")

                    result['time'] = time.time()
                    if get_frame:
                        width = depth_frame.get_width()
                        height = depth_frame.get_height()

                        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                        depth_data = depth_data.reshape((height, width))
                        # dep4show = (depth_data * 0.0625).astype(np.uint8)

                        dep16 = np.asanyarray(depth_data, np.uint16)

                        start_time_est = time.time()
                        out_put = self.cObjPose.Run(img_bgr, dep16)
                        end_time_est = time.time()
                        time_est = end_time_est - start_time_est
                        print("Est time / frame: {:.2f} ms".format(1000 * time_est))

                        result['time'] = time.time()
                        result['cls_pose'] = out_put
                        result = json.dumps(result, ensure_ascii=False)
                        print("=" * 20)
                        print(result)
                    else:
                        result['errorcode'] = "ReadCamFailed"
                        print("Getting rgbd frames failed !!!")
                    
                    if self.debug:
                        img_bgr, img_desk = self.cObjPose.draw(img_bgr, dep16)
                        cv2.imshow("img_desk", img_desk)
                        cv2.imshow("img_bgr", img_bgr)
                        key = cv2.waitKey(1)
                        if key == ord('q') or key == 27:
                            break

                    # send result
                    try:
                        # result = json.dumps(result, ensure_ascii=False)
                        clientsocket.sendall(result.encode())
                        # clientsocket.send(msg.encode('utf-8'))
                    except:
                        clientsocket.close()
                        print("send wrong")
                except:
                    if clientsocket:
                        clientsocket.close()
                    print("server is killed !!!")
                    break


                
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='rgbd')

    parser.add_argument('--debug',  type=bool, default=True)
    # ip port for socket
    parser.add_argument('--ip_port', type=int, default=9998, help='ip port')

    args = parser.parse_args()
    myEbCr = cGetObjPose(args)
    myEbCr.Run(flagUseSocket=True)
    
