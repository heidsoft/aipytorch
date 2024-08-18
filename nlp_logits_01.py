import torch
import torch.nn as nn
import torch.optim as optim

# 假设我们有 3 个类别的 logits
logits = torch.tensor([[2.0, 1.0, 0.1]], dtype=torch.float32)
labels = torch.tensor([0])  # 真实标签为第 0 类

# 定义损失函数
loss_fn = nn.CrossEntropyLoss()

# 计算损失
loss = loss_fn(logits, labels)
print(f"Loss: {loss.item()}")
