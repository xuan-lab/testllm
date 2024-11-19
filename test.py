import torch
import torch.nn as nn
import torch.optim as optim

# 设置随机种子以确保结果可重复
torch.manual_seed(42)

# 定义简单的多层感知机模型
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        # 输入层到隐藏层，10个输入神经元，5个隐藏神经元
        self.fc1 = nn.Linear(10, 5)  
        # 隐藏层到输出层，5个隐藏神经元，1个输出神经元
        self.fc2 = nn.Linear(5, 1)    
        self.activation = nn.ReLU()    # 使用ReLU激活函数

    def forward(self, x):
        # 前向传播过程
        # 输入层到隐藏层的前向传播
        x = self.fc1(x)                
        print(f'  Layer 1 Output (Raw): \n{x.detach().numpy()}')  # 输出隐藏层的中间结果（未激活值）
        
        # 通过激活函数
        x = self.activation(x)         
        print(f'  Layer 1 Output (Activated): \n{x.detach().numpy()}')  # 输出激活后的结果
        
        # 隐藏层到输出层的前向传播
        x = self.fc2(x)                
        return x

# 创建模型、损失函数和优化器
model = SimpleMLP()
criterion = nn.MSELoss()  # 使用均方误差损失函数
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam优化器，学习率为0.01

# 生成输入数据和目标数据
inputs = torch.randn(32, 10)  # 32个样本，每个样本10个特征
targets = torch.randn(32, 1)   # 32个目标值

# 训练循环
num_epochs = 10  # 为了演示，减少训练周期数
for epoch in range(num_epochs):
    # 清零梯度
    optimizer.zero_grad()

    # 前向传播
    outputs = model(inputs)  # 模型的输出
    loss = criterion(outputs, targets)  # 计算损失

    # 反向传播
    loss.backward()  # 计算梯度

    # 打印训练信息
    print(f'\n--- Epoch [{epoch + 1}/{num_epochs}] ---')
    print(f'  Loss: {loss.item():.4f}')  # 当前损失值

    # 输出模型的权重和偏置，强调每层的参数
    print('  Layer Parameters:')
    print(f'    Layer 1 Weights: \n{model.fc1.weight.data.numpy()}')  # 输出第一层权重
    print(f'    Layer 1 Bias: \n{model.fc1.bias.data.numpy()}')  # 输出第一层偏置
    print(f'    Layer 2 Weights: \n{model.fc2.weight.data.numpy()}')  # 输出第二层权重
    print(f'    Layer 2 Bias: \n{model.fc2.bias.data.numpy()}')  # 输出第二层偏置

    # 输出当前模型的输出与目标
    print('  Current Outputs vs Targets (first 5 samples):')
    for i in range(min(5, len(outputs))):  # 只输出前5个样本的预测和目标
        print(f'    Output [{i}]: {outputs[i].item():.4f}, Target [{i}]: {targets[i].item():.4f}')

    # 输出梯度
    print('  Gradients (after backward pass):')
    print(f'    Layer 1 Weights Gradients: \n{model.fc1.weight.grad.data.numpy()}')  # 输出第一层权重的梯度
    print(f'    Layer 1 Bias Gradients: \n{model.fc1.bias.grad.data.numpy()}')  # 输出第一层偏置的梯度
    print(f'    Layer 2 Weights Gradients: \n{model.fc2.weight.grad.data.numpy()}')  # 输出第二层权重的梯度
    print(f'    Layer 2 Bias Gradients: \n{model.fc2.bias.data.numpy()}')  # 输出第二层偏置的梯度

    # 更新参数
    optimizer.step()  # 使用优化器更新参数

    print('------------------------------')  # 分隔符
