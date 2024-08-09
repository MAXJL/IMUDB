import rosbag
import pandas as pd
import numpy as np

def bag_to_csv(bag_file, output_csv):
    bag = rosbag.Bag(bag_file)
    imu_data = []
    timestamps = []

    for topic, msg, t in bag.read_messages(topics=['/imu0']):
        timestamps.append(msg.header.stamp.to_nsec())
        imu_data.append([
            msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
            msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z
        ])

    bag.close()

    # 转换为DataFrame
    df_timestamps = pd.DataFrame(timestamps, columns=["#timestamp [ns]"])
    df_imu_data = pd.DataFrame(imu_data, columns=[
        "w_RS_S_x [rad s^-1]", "w_RS_S_y [rad s^-1]", "w_RS_S_z [rad s^-1]",
        "a_RS_S_x [m s^-2]", "a_RS_S_y [m s^-2]", "a_RS_S_z [m s^-2]"
    ])
    
    # 合并时间戳和IMU数据
    df = pd.concat([df_timestamps, df_imu_data], axis=1)
    
    # 保存到CSV文件
    df.to_csv(output_csv, index=False)
    print(f"Saved data to {output_csv}")

if __name__ == "__main__":
    bag_file = '/home/jia/MA/IMUDB-main/EuRoC/bags/V1_01_easy.bag.new.imudb2.bag'  # 替换为您的bag文件路径
    output_csv = '/home/jia/MA/IMUDB-main/result/V101/new_imu_data.csv'  # 替换为您希望保存的csv文件路径
    bag_to_csv(bag_file, output_csv)
