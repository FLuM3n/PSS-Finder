import torch
import torch.nn as nn

class GlobalLocalNetwork(nn.Module):
    def __init__(self, global_dim, local_dim, num_classes=53):
        """
        :param global_dim: Dimension of global features.
        :param local_dim: Dimension of local features.
        :param num_classes: Number of classes for classification task, default is 53.
        """
        super(GlobalLocalNetwork, self).__init__()

        # 全局特征处理
        self.global_fc = nn.Linear(global_dim, 576)

        # 局部特征处理
        self.cnn1 = nn.Sequential(
            nn.Conv1d(local_dim, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.cnn2 = nn.Sequential(
            nn.Conv1d(local_dim, 128, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.cnn3 = nn.Sequential(
            nn.Conv1d(local_dim, 64, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # 全连接层
        self.fc1 = nn.Sequential(
            nn.Linear(576 + 256 + 128 + 64, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(256, num_classes)
        )

    def forward(self, global_features, local_features, mask=None):
        """
        前向传播。
        :param global_features: 形状为 (batch_size, global_dim) 的全局特征。
        :param local_features: 形状为 (batch_size, length, local_dim) 的局部特征。
        :param mask: 形状为 (batch_size, length) 的掩码，1 表示有效 token，0 表示填充 token。
        :return: 形状为 (batch_size, num_classes) 的输出。
        """
        # 全局特征处理
        global_features = self.global_fc(global_features)  # 形状: (batch_size, 576)

        # 局部特征处理
        local_features = local_features.permute(0, 2, 1)  # 调整形状为 (batch_size, local_dim, length)

        # 如果提供了 mask，将填充 token 的特征置为零
        if mask is not None:
            mask = mask.unsqueeze(1)  # 调整形状为 (batch_size, 1, length)
            local_features = local_features * mask  # 将填充 token 的特征置为零

        # 通过 CNN 层提取特征
        local_features1 = self.cnn1(local_features).squeeze(-1)  # 形状: (batch_size, 256)
        local_features2 = self.cnn2(local_features).squeeze(-1)  # 形状: (batch_size, 128)
        local_features3 = self.cnn3(local_features).squeeze(-1)  # 形状: (batch_size, 64)

        # 特征拼接
        combined_features = torch.cat([global_features, local_features1, local_features2, local_features3], dim=1)  # 形状: (batch_size, 576 + 256 + 128 + 64)

        # 全连接层
        x = self.fc1(combined_features)  # 形状: (batch_size, 512)
        x = self.fc2(x)  # 形状: (batch_size, 256)
        output = self.fc3(x)  # 形状: (batch_size, num_classes)

        return output