# 🎯 Fashion-MNIST CNN Comparative Study

 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/) 



本项目使用 **5种经典CNN模型**（LeNet、AlexNet、VGGNet16、GoogLeNet、ResNet18）对Fashion-MNIST数据集进行图像分类，包含完整的训练、测试和可视化流程，可用于学习或基准测试。**让深度学习更生动！** 🚀

## 🚩 功能特性
- **📦 多模型支持**：LeNet、AlexNet、VGGNet16、GoogLeNet、ResNet18
- **⚡ 高效完整的训练流程**：支持模型选择、超参数配置、模型保存
- **📊 可视化工具**：数据集示例、训练损失/准确率曲线
- **🎨 轻量级代码**：基于PyTorch实现，模块化设计，易于扩展

## 📂 文件结构
```
├── dataset/                  # 数据集目录（自动下载或手动放置）
├── model/                    # 模型定义
├── output_model/             # 训练好的模型权重（.pth文件）
├── images/                   # 结果可视化图片（如损失/准确率曲线）
├── plot.py                   # 绘制数据集示例
├── train.py                  # 训练脚本（支持模型选择）
├── test.py                   # 测试脚本
└── README.md
```
## 🚀 快速开始

### 环境依赖
```bash
git clone https://github.com/tang2399/Fashion-MNIST-CNN-Comparative-Study.git
cd Fashion-MNIST-CNN-Comparative-Study pip install -r requirements.txt
```

### 使用步骤
1. **数据集准备**  
   运行 `plot.py` 查看数据集示例并下载数据集：  
   
   ```bash
   python plot.py # 👉 生成示例图片到 images/
   ```
   
2. **训练模型**  
   运行 `train.py` 并输入模型序号：  
   
   ```bash
   python train.py # 🚂 开始训练吧！
   ```
   - 训练完成后，模型权重保存至 `output_model/` ✅
   - 训练过程的可视化结果保存至 `images/` 📈
   
3. **测试模型**  
   运行 `test.py` 并指定模型序号：  
   
   ```bash
   python test.py # 🧪 测试模型性能
   ```

## 📊 实验结果

不同模型在测试集上的准确率对比 📝（epochs=,lr=0.001）：  
| 模型      | 测试准确率 | 训练时间（3060，6G） |
| --------- | ---------- | -------------------- |
| LeNet     |            | ⏳                    |
| AlexNet   |            |                      |
| VGGNet16  |            |                      |
| GoogLeNet |            |                      |
| ResNet18  |            |                      |



## 🤝 贡献与许可

- 欢迎提交Issue或PR！详细问题请联系 [2399340647@qq.com] 📧
- 项目基于 [MIT License](LICENSE) 开源 💡