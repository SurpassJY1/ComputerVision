# NEU-DET 基线：YOLOv8n

本仓库提供 **Ultralytics YOLOv8n** 在 NEU-DET（YOLO 格式）上的**最小可复现**训练与验证脚本，便于后续做轻量化与 TensorRT 实验时在相同协议下对比。

## 环境（Windows 4080 训练机）

1. 安装 **Python 3.10+**，建议使用 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)。
2. 按 [PyTorch 官网](https://pytorch.org/get-started/locally/) 安装 **带 CUDA 的 PyTorch**（与显卡驱动匹配）。
3. 安装依赖：

```bash
pip install -r requirements.txt
```

## 数据集（NEU-DET → YOLO 格式）

1. 将 NEU-DET 转为 YOLO 目录结构：

```text
YOUR_DATASET_ROOT/
  images/
    train/
    val/
  labels/
    train/
    val/
```

2. 复制配置并修改路径：

```bash
copy data\neu_det_template.yaml data\neu_det.yaml
```

编辑 `data/neu_det.yaml`，把 `path:` 设为 **`YOUR_DATASET_ROOT` 的绝对路径**（Windows 可用正斜杠，如 `D:/data/NEU-DET-YOLO`）。

3. **类别顺序**需与标注文件中的 `class id` 一致。模板中默认 6 类（0–5）；若你的转换脚本顺序不同，请同步修改 `names` 与 `nc`。

## 训练（基线 YOLOv8n）

```bash
python train_baseline.py --data data/neu_det.yaml --epochs 150 --imgsz 640 --batch 16 --device 0
```

- 默认模型：`yolov8n.pt`（COCO 预训练权重）
- 结果目录：`runs/detect/neu_det_yolov8n_baseline/`

## 训练（轻量化改进：Ghost-YOLOv8n）

Backbone 下采样 `Conv` 替换为 `GhostConv`，减少参数量与 FLOPs，同时保持检测精度。

```bash
python train_ghost.py --data data/neu_det.yaml --epochs 150 --imgsz 640 --batch 16 --device 0
```

- 模型配置：`models/yolov8n-ghost.yaml`（从零训练，无预训练权重）
- 结果目录：`runs/detect/neu_det_yolov8n_ghost/`

> **注意（公平对比）**：Ghost 模型从零开始训练，而 baseline 使用 COCO 预训练权重。
> 论文中建议在实验设置章节明确说明，并补充一组 baseline 也从零训练的对照实验（可选）。

## 验证

```bash
# 验证基线
python val_baseline.py --weights runs/detect/neu_det_yolov8n_baseline/weights/best.pt --data data/neu_det.yaml

# 验证 Ghost 改进模型
python val_baseline.py --weights runs/detect/neu_det_yolov8n_ghost/weights/best.pt --data data/neu_det.yaml
```

## 论文实验对比表格（参考格式）

| 方法 | mAP@0.5 | mAP@0.5:0.95 | Params(M) | FLOPs(G) | FPS(PyTorch) | FPS(TRT-FP16) |
|------|---------|--------------|-----------|----------|--------------|---------------|
| YOLOv8n (Baseline) | 83.0% | 51.2% | 3.0 | 8.1 | — | — |
| Ghost-YOLOv8n (Ours) | ? | ? | ?↓ | ?↓ | ?↑ | ?↑↑ |

## 消融实验（论文必须）

- 对比 Baseline 与 Ghost 改进：证明 GhostConv 带来 Params/FLOPs 下降
- 可选：替换位置消融（只替换 1/2/3/4 个 backbone 层）

## 导出 ONNX / TensorRT（在 NVIDIA 机器上）

```bash
# 导出基线
python export_tensorrt.py --weights runs/detect/neu_det_yolov8n_baseline/weights/best.pt --imgsz 640 --format engine --half

# 导出 Ghost 改进模型
python export_tensorrt.py --weights runs/detect/neu_det_yolov8n_ghost/weights/best.pt --imgsz 640 --format engine --half
```

具体 TensorRT 版本与 `export` 参数以 [Ultralytics Export](https://docs.ultralytics.com/modes/export/) 为准。

## Mac 上写代码、Windows 上训练

- 用 **Git** 同步本仓库；数据集与 `.pt` 权重不要提交，在 4080 本机存放。
- 训练与 TensorRT **必须在 Windows 4080（CUDA）** 上执行。

## 论文中建议写明的设置

| 设置项 | 值 |
|--------|-----|
| Baseline 模型 | `yolov8n.pt` |
| Ghost 改进模型 | `models/yolov8n-ghost.yaml` |
| 数据集 | NEU-DET（nc=6） |
| 输入尺寸 | 640 |
| Epochs | 150（含 early stopping patience=50） |
| Batch | 16 |
| Seed | 42 |
| 硬件 | RTX 4080 Laptop |
| 指标 | mAP@0.5、mAP@0.5:0.95、Params、FLOPs、FPS |
