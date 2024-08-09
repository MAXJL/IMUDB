import pandas as pd
import torch
import os
import sys
import numpy as np
from datetime import datetime
from torch import nn, optim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'models')))

from models.imudb import Model

class IMUData:
    def __init__(self):
        self.timestamps = []
        self.imu_values = []

def count_data_lines(filename):
    with open(filename, 'r') as file:
        return sum(1 for _ in file) - 1

def read_imu_csv_euroc(filename, start_line, num_lines):
    data = pd.read_csv(filename, skiprows=range(1, start_line), nrows=num_lines)
    imu_data = IMUData()
    imu_data.timestamps = data.iloc[:, 0].tolist()
    imu_data.imu_values = data.iloc[:, 1:].values.flatten().tolist()
    print(f"Read {filename} with {len(imu_data.imu_values)} values.")
    return imu_data

def save_to_csv(filename, timestamps, outputs, save_all=False):
    # Check if the file exists and if it is empty
    file_exists = os.path.exists(filename) and os.path.getsize(filename) > 0
    
    # Ensure outputs is detached, on CPU, and numpy formatted
    outputs = outputs.detach().cpu().numpy().squeeze()

    if save_all:
        df_timestamps = pd.DataFrame(timestamps, columns=["#timestamp [ns]"])
        df_outputs = pd.DataFrame(outputs, columns=["w_RS_S_x [rad s^-1]", "w_RS_S_y [rad s^-1]", "w_RS_S_z [rad s^-1]", "a_RS_S_x [m s^-2]", "a_RS_S_y [m s^-2]", "a_RS_S_z [m s^-2]"])
    else:
        df_timestamps = pd.DataFrame([timestamps[-1]], columns=["#timestamp [ns]"])
        df_outputs = pd.DataFrame([outputs[-1]], columns=["w_RS_S_x [rad s^-1]", "w_RS_S_y [rad s^-1]", "w_RS_S_z [rad s^-1]", "a_RS_S_x [m s^-2]", "a_RS_S_y [m s^-2]", "a_RS_S_z [m s^-2]"])
    
    # Concatenate along the columns
    df = pd.concat([df_timestamps, df_outputs], axis=1)
    
    # Determine whether to write header: if the file doesn't exist or is not empty
    header = not file_exists  # Write header if file does not exist or is empty
    df.to_csv(filename, mode='a', header=header, index=False)
    
    print(f"Saved data to {filename} with {'header' if header else 'no header'}")


def main():
    model_path = "/home/jia/MA/IMUDB-main/checkpoints_jiab760gubuntu2004/euroc_imudb/dev_imudb_2024-08-07T08:46:40.284296-07:00/euroc_imudb-epoch=2382-val_denoise_loss=0.000011.ckpt"
    output_filename = "/home/jia/MA/IMUDB-main/result/MH02/new_imu_data.csv"
    filename = "/home/jia/datasets/euroc/MH02/imu0/gt_backup/data.csv"
    
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")  # 添加这一行来打印是否使用了 CUDA

    
    # 定义配置
    config = {
        'model': {
            'hyper_params': {
                'starting_learning_rate': 0.001,
                'denoise_loss_weights': 10,
                'nsp_loss_weights': 1,
                'mlm_loss_weights': 1,
                'T_0': 5000,
                'T_mult': 10,
                'eta_min': 1e-8,
                'feature_num': 6,
                'hidden': 72,
                'hidden_ff': 144,
                'n_layers': 4,
                'n_heads': 4,
                'seq_len': 120,
                'emb_norm': True
            }
        }
    }


    model = Model.load_from_checkpoint(checkpoint_path = model_path, config=config)
    model.to(device)
    model.eval()

    # Get the limu_bert_mlm model获得去噪模型
    limu_bert_mlm = model.limu_bert_mlm

    total_lines = count_data_lines(filename)
    window_size = 30
    data_buffer = np.zeros((window_size, 6))
    timestamps_buffer = []

    frist_full_window = True

    for line_index in range(1, total_lines + 1):
        imu_data = read_imu_csv_euroc(filename, line_index, 1)
        new_data_point = np.array(imu_data.imu_values).reshape(-1, 6)
        new_timestamp = imu_data.timestamps

        data_buffer = np.roll(data_buffer, -1, axis=0)
        data_buffer[-1, :] = new_data_point
        timestamps_buffer.append(new_timestamp)

        if line_index >= window_size:
            input_data = torch.tensor(data_buffer, dtype=torch.float32).view(1, window_size, 6).to(device)
            start_time = datetime.now()
            #打印input的shape
            # print(f"Input shape: {input_data.shape}") 
            output = limu_bert_mlm(input_data)
            print(output)
            #打印output的shape
            # print(f"Output shape: {output.shape}")

            duration = (datetime.now() - start_time).total_seconds() * 1000
            print(f"Inference time: {duration} ms")

            denoise_output = output
           

            if frist_full_window:
                save_to_csv(output_filename, timestamps_buffer[-window_size:], denoise_output, save_all=True)
                frist_full_window = False
            else:
                save_to_csv(output_filename, timestamps_buffer[-window_size:], denoise_output, save_all=False)



if __name__ == "__main__":
    main()
