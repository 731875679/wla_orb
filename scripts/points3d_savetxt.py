import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2
import struct

# 存储点云数据
points_data = []

def callback(msg):
    global points_data

    # 原始数据长度
    point_step = msg.point_step

    for i in range(msg.width):
        # 计算偏移量
        offset = i * point_step

        # 提取x, y, z
        x = struct.unpack_from('f', msg.data, offset)[0]
        y = struct.unpack_from('f', msg.data, offset + 4)[0]
        z = struct.unpack_from('f', msg.data, offset + 8)[0]

        # 提取rgb，作为32位整数
        rgb_int = struct.unpack_from('I', msg.data, offset + 16)[0]

        # 提取RGB分量
        r = (rgb_int >> 16) & 0xFF
        g = (rgb_int >> 8) & 0xFF
        b = rgb_int & 0xFF

        # 检查RGB值是否在合理范围内
        if 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255:
            point_id = len(points_data)  # 使用当前列表长度作为点的ID
            points_data.append((point_id, x, y, z, r, g, b))

    # 每次更新时保存到文件
    save_to_file()

def save_to_file():
    with open('point_cloud_data.txt', 'w') as f:
        for point in points_data:
            f.write(f'{point[0]},{point[1]},{point[2]},{point[3]},{point[4]},{point[5]},{point[6]}\n')

def listener():
    rospy.init_node('point_cloud_listener', anonymous=True)
    rospy.Subscriber('/slam/keyframe_point3d', PointCloud2, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()