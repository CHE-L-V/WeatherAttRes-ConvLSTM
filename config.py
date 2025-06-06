import torch
import numpy as np

# 数据路径配置
DATA_RAW_DIR = 'E:/WeatherAttRes-ConvLSTM/data/raw_t2m'
DATA_PROCESSED_DIR = 'E:/WeatherAttRes-ConvLSTM/data/processed_t2m'
OUTPUT_DIR = 'E:/WeatherAttRes-ConvLSTM/weights/models_t2m'
PREDICTIONS_DIR = 'E:/WeatherAttRes-ConvLSTM/outputs/predictions'

# 模型配置
BATCH_SIZE = 1
OUTPUT_STEPS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据维度配置
LAT_START = 90.0
LAT_END = -90.0
LAT_STEP = -0.25
LON_START = 0.0
LON_END = 360.0
LON_STEP = 0.25

LATITUDE = np.arange(LAT_START, LAT_END + LAT_STEP, LAT_STEP)
LONGITUDE = np.arange(0.0, 360.0, LON_STEP)

# 模型参数
MODEL_CONFIG = {
    'input_dim': 1,
    'hidden_dim': [32, 64, 64],
    'kernel_size': (3, 3),
    'num_layers': 3,
    'batch_first': True,
    'bias': True,
    'return_all_layers': False,
    'output_steps': OUTPUT_STEPS
}

# 预测变量配置
PREDICTION_VAR = 't2m'  # 预测变量名称 