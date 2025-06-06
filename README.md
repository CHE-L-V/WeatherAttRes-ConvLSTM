# WeatherAttRes-ConvLSTM

基于ConvLSTM的天气要素时间降尺度预测模型。该模型利用当前时刻及前3个时刻的数据，结合后6小时的fuxi输出数据，进行高时间分辨率的天气要素预测。

## 模型特点

- 输入：当前时刻及前3个时刻的天气数据（共4个时刻）
- 辅助输入：后6小时的fuxi模型输出数据
- 输出：未来5个时刻的高时间分辨率预测结果
- 时间分辨率：6小时
- 空间分辨率：0.25° x 0.25°

## 项目结构

```
WeatherAttRes-ConvLSTM/
├── data/
│   ├── raw_msl/                # 原始数据目录（ERA5数据）
│   └── processed_msl/          # 处理后的数据目录
├── weights/
│   └── models_msl/            # 模型权重文件目录
├── outputs/
│   └── predictions/           # 预测结果输出目录
├── models/
│   └── convlstm.py           # ConvLSTM模型定义
├── config.py                  # 配置文件
├── inference.py              # 推理脚本
├── utils.py                  # 工具函数
└── requirements.txt          # 项目依赖
```

## 环境要求

- Python 3.11
- CUDA (如果使用GPU)

## 安装步骤

1. 克隆仓库：
```bash
git clone [repository_url]
cd WeatherAttRes-ConvLSTM
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 配置参数：
   - 在 `config.py` 中设置相关参数，包括：
     - 数据路径
     - 模型参数
     - 预测变量

2. 准备数据：
   - 将原始ERA5数据文件放在 `data/raw_msl/` 目录下
   - 确保 `data/processed_msl/` 目录下有 `mean.npy` 和 `std.npy` 文件
   - 准备fuxi模型输出的未来数据

3. 运行推理：
```bash
python inference.py
```

4. 查看结果：
   - 预测结果将保存在 `outputs/predictions/` 目录下
   - 输出文件格式为 NetCDF，文件名格式为 `prediction_[变量名]_[时间].nc`
   - 预测结果包含未来5个时刻的数据

## 数据说明

### 输入数据
- ERA5数据：当前时刻及前3个时刻的天气数据6小时分辨率
- Fuxi数据：后6小时的模型输出数据

### 输出数据
- 预测结果：未来5个时刻的高时间分辨率天气要素数据
- 时间分辨率：1小时
- 空间分辨率：0.25° x 0.25°

## 配置说明

在 `config.py` 中可以修改以下参数：

- `DATA_RAW_DIR`: 原始数据目录
- `DATA_PROCESSED_DIR`: 处理后数据目录
- `OUTPUT_DIR`: 模型权重文件目录
- `PREDICTIONS_DIR`: 预测结果输出目录
- `PREDICTION_VAR`: 预测变量名称
- `OUTPUT_STEPS`: 预测步长（默认为5）
- `MODEL_CONFIG`: 模型配置参数

## 注意事项

1. 确保输入数据文件按时间顺序排列
2. 确保模型权重文件存在于指定目录
3. 确保有足够的磁盘空间存储预测结果
4. 输入数据必须包含当前时刻及前3个时刻的数据
5. 必须提供后6小时的fuxi模型输出数据

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 联系方式

[添加联系方式] 