#!/usr/bin/env python
#coding: utf-8

import rospy
from   sensor_msgs      import point_cloud2
from   sensor_msgs.msg  import PointCloud2, PointField
from   std_msgs.msg     import Header


def talker():
    # 导入 socket、sys 模块
    import socket
    import sys
    import json
    import time

    # 获取本地ip
    # s1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # s1.connect(('8.8.8.8', 80))
    # host_ip = s1.getsockname()[0]
    # print("host_ip=", host_ip, ", type(ip)=", type(host_ip))

    #话题
    pub = rospy.Publisher('aimo/rgbd_obj_cls_pose', PointCloud2, queue_size=1)
    rospy.init_node('rgbd_node', anonymous=True)

    rate = rospy.Rate(2)  # 1hz
    while not rospy.is_shutdown():
        print("-------")
        t0=time.time()
        
        # 创建 socket 对象
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('127.0.0.1', 9998)) # 连接服务，指定主机和端口
        msg = s.recv(4096) # 接收小于 1024 字节的数据
        print("msg=\n",msg)
        s.close()
        msg_dict = eval(msg)
        print(msg.decode('utf-8'))

        #fill point cloud
        fields =[
            PointField(name='cls', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='x', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='a', offset=16, datatype=PointField.FLOAT32, count=1),
            PointField(name='b', offset=20, datatype=PointField.FLOAT32, count=1),
            PointField(name='c', offset=24, datatype=PointField.FLOAT32, count=1),
            PointField(name='d', offset=28, datatype=PointField.FLOAT32, count=1)
        ]
        point_data=msg_dict['cls_pose']
        header = Header()
        header.frame_id = "stere_frame_id"
        header.stamp = rospy.Time(msg_dict['time'])
        pc2 = point_cloud2.create_cloud(header, fields, point_data)

        #publish
        pub.publish(pc2)
        rate.sleep()
        print("t1-t0=",time.time()-t0)


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
