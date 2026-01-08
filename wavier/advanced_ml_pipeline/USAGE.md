# 使用指南 (Usage Guide)

## 快速开始 (Quick Start)

### 1. 环境设置 (Environment Setup)

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 训练模型 (Train Model)

```bash
# 基础训练
python src/main.py

# 使用自定义配置
python src/main.py model.hidden_dim=1024 training.epochs=50

# 超参数搜索
python src/main.py -m \
    model.hidden_dim=256,512,1024 \
    training.learning_rate=1e-4,3e-4
```

### 3. 启动API服务 (Start API Server)

```bash
# 开发模式
uvicorn src.inference.api:app --reload

# 生产模式
uvicorn src.inference.api:app --host 0.0.0.0 --port 8000 --workers 4
```

### 4. Docker部署 (Docker Deployment)

```bash
# 构建镜像
docker build -t ml-pipeline:latest .

# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f
```

## 核心功能 (Core Features)

### 数据处理 (Data Processing)

项目使用Polars进行高性能数据处理：

```python
from src.data.loaders import CSVDataLoader, DataSplitter

# 加载数据
loader = CSVDataLoader(
    file_path="data/raw/dataset.csv",
    target_column="target",
    cache_dir="data/processed"
)
df = loader.load()

# 分割数据
splitter = DataSplitter(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
train_df, val_df, test_df = splitter.split(df, "target")
```

### 特征工程 (Feature Engineering)

使用sklearn兼容的自定义transformer：

```python
from src.features.engineering import (
    MissingValueImputer,
    OutlierClipper,
    FeatureInteractionGenerator
)

# 缺失值填充
imputer = MissingValueImputer(strategy="median", add_indicator=True)
df = imputer.fit_transform(train_df)

# 异常值裁剪
clipper = OutlierClipper(method="iqr", iqr_multiplier=1.5)
df = clipper.fit_transform(df)

# 特征交互
interactor = FeatureInteractionGenerator(
    columns=["feature_1", "feature_2"],
    degree=2
)
df = interactor.transform(df)
```

### 模型训练 (Model Training)

支持现代深度学习技术：

```python
from src.models.architectures import TransformerClassifier
from src.models.trainer import Trainer, WarmupCosineScheduler

# 创建模型
model = TransformerClassifier(
    input_dim=20,
    hidden_dim=512,
    num_layers=6,
    num_heads=8,
    dropout=0.1
)

# 配置训练器
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    mixed_precision=True,  # 混合精度训练
    gradient_clip=1.0,
    accumulation_steps=2
)

# 训练
history = trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100,
    scheduler=scheduler
)
```

### API调用 (API Usage)

```bash
# 健康检查
curl http://localhost:8000/health

# 单个预测
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [1.0, 2.0, 3.0, ...]
  }'

# 批量预测
curl -X POST http://localhost:8000/batch_predict \
  -H "Content-Type: application/json" \
  -d '{
    "batch": [
      [1.0, 2.0, ...],
      [3.0, 4.0, ...]
    ]
  }'
```

## 高级功能 (Advanced Features)

### 1. 混合精度训练 (Mixed Precision Training)

自动使用PyTorch的AMP进行加速：
- 减少内存使用约50%
- 训练速度提升2-3倍
- 保持模型精度

### 2. 梯度累积 (Gradient Accumulation)

处理大batch size时的内存限制：
```yaml
training:
  batch_size: 32
  gradient_accumulation_steps: 4  # 等效batch_size=128
```

### 3. 学习率调度 (Learning Rate Scheduling)

Warmup + Cosine Annealing策略：
- 前期线性warmup避免震荡
- 后期cosine decay收敛更稳定

### 4. Early Stopping

自动在验证集性能不提升时停止训练：
```python
early_stopping = EarlyStopping(patience=10, min_delta=0.001)
```

### 5. 模型编译 (Model Compilation)

使用PyTorch 2.0的torch.compile加速：
```yaml
training:
  compile_model: true  # 加速30-100%
```

## 测试 (Testing)

```bash
# 运行所有测试
pytest tests/ -v

# 代码覆盖率
pytest tests/ --cov=src --cov-report=html

# 只运行单元测试
pytest tests/ -m "not integration"

# 属性测试
pytest tests/test_pipeline.py::TestPropertyBased -v
```

## 监控 (Monitoring)

### MLflow实验追踪

```bash
# 启动MLflow UI
mlflow ui --host 0.0.0.0 --port 5000

# 访问 http://localhost:5000
```

### Prometheus + Grafana

```bash
# 启动监控栈
docker-compose up prometheus grafana

# Grafana: http://localhost:3000
# Prometheus: http://localhost:9090
```

## 性能优化建议 (Performance Tips)

1. **数据加载**: 使用Polars代替Pandas (3-10x faster)
2. **模型优化**: 启用torch.compile (PyTorch 2.0+)
3. **推理加速**: 考虑ONNX导出或TorchScript
4. **批处理**: 使用批量预测endpoint处理大量请求
5. **缓存**: 启用数据缓存避免重复处理

## 常见问题 (FAQ)

**Q: CUDA内存不足怎么办?**
A: 
- 减小batch_size
- 启用gradient_accumulation_steps
- 减少模型hidden_dim

**Q: 如何使用自己的数据?**
A: 
- 继承DataLoader类实现load_raw()和transform()
- 或直接使用CSVDataLoader

**Q: 如何部署到生产环境?**
A: 
- 使用docker-compose快速部署
- 或构建Docker镜像部署到K8s

## 参考资料 (References)

- [PyTorch Documentation](https://pytorch.org/docs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Hydra Configuration](https://hydra.cc/)
- [Polars Guide](https://pola-rs.github.io/polars/)
