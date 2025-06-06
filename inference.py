import os
import torch
import torch.nn as nn
from models.convlstm import ConvLSTM
from utils import load_model, denormalize
import numpy as np
import xarray as xr
from config import *

def load_and_preprocess(file_path, mean, std):
    """
    加载并预处理单个NetCDF文件。
    """
    ds = xr.open_dataset(file_path)
    value = ds[PREDICTION_VAR].values
    ds.close()

    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    elif value.ndim == 2:
        pass
    else:
        raise ValueError(f"Unexpected data shape: {value.shape} in file {file_path}")

    value = (value - mean) / std
    return value

def predict():
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    # 加载均值和标准差
    mean = np.load(os.path.join(DATA_PROCESSED_DIR, 'mean.npy')).item()
    std = np.load(os.path.join(DATA_PROCESSED_DIR, 'std.npy')).item()

    # 获取输入文件列表
    input_files = [
        'era5_2023_06_05_12.nc',
        'era5_2023_06_05_18.nc',
        'era5_2023_06_06_00.nc',
        'era5_2023_06_06_06.nc',
        'fuxi_2023_06_06_12.nc'
    ]
    input_files = [os.path.join(DATA_RAW_DIR, f) for f in input_files]

    # 加载输入序列
    input_sequence = []
    for file_path in input_files:
        data = load_and_preprocess(file_path, mean, std)
        input_sequence.append(data)
    input_sequence = np.stack(input_sequence, axis=0)
    input_sequence = input_sequence[:, np.newaxis, :, :]  
    input_tensor = torch.from_numpy(input_sequence).float().unsqueeze(0)  

    # 定义并加载模型
    model = ConvLSTM(**MODEL_CONFIG)
    model = nn.DataParallel(model)
    model = model.to(DEVICE)
    model_path = os.path.join(OUTPUT_DIR, f'best_weight_{PREDICTION_VAR}.pth')
    model = load_model(model, model_path, DEVICE)
    model.eval()

    # 进行预测
    with torch.no_grad():
        input_tensor = input_tensor.to(DEVICE)
        prediction = model(input_tensor)  
        prediction = prediction.cpu().numpy()[0] 
        #print(prediction.shape)
        prediction = prediction.squeeze()
        print(f"压缩后的形状: {prediction.shape}") 
         

    # 反标准化
    prediction = denormalize(prediction, mean, std)

    # 保存预测结果为NetCDF文件
    ds_pred = xr.Dataset(
        data_vars={
            f'{PREDICTION_VAR}_pred': (['time', 'lat', 'lon'], prediction)
        },
        coords={
            'time': np.arange(1, OUTPUT_STEPS + 1),
            'lat': LATITUDE,
            'lon': LONGITUDE
        }
    )

    # 保存预测结果
    output_filename = f"prediction_{PREDICTION_VAR}_2023_06_06_06.nc"
    ds_pred.to_netcdf(os.path.join(PREDICTIONS_DIR, output_filename))
    ds_pred.close()

    print(f"预测结果已保存到: {os.path.join(PREDICTIONS_DIR, output_filename)}")

if __name__ == "__main__":
    predict() 