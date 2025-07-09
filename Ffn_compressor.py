import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

class FFNCompressor(nn.Module):
    """
    基于前馈网络的自动编码器，用于压缩和重建输入数据
    input_dim: 输入特征维度
    hidden_dims: 中间隐藏层维度列表
    latent_dim: 压缩后的潜在表示维度
    """
    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int):
        super().__init__()
        # 构建编码器
        encoder_layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h))
            encoder_layers.append(nn.ReLU())
            prev_dim = h
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # 构建解码器
        decoder_layers = []
        prev_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h))
            decoder_layers.append(nn.ReLU())
            prev_dim = h
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z


if __name__ == "__main__":
    # 假设已有一个形状为 (2048, 8192) 的矩阵
    matrix = np.random.randint(-8, 8, size=(2048, 8192))
    # 转为 float 类型的 torch.Tensor
    data = torch.from_numpy(matrix).float()

    # 构建数据加载器
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 实例化模型（可根据需求调整隐藏层和潜在维度）
    input_dim = 8192
    hidden_dims = [1024, 512]
    latent_dim = 256
    model = FFNCompressor(input_dim, hidden_dims, latent_dim)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 训练循环
    epochs = 20
    for epoch in range(epochs):
        total_loss = 0.0
        for batch, in loader:
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.6f}")

    # 训练完成后，获取压缩后的表示
    with torch.no_grad():
        compressed = model.encoder(data)
    print("Compressed shape:", compressed.shape)  # (2048, latent_dim)
