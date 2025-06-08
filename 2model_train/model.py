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

        # global
        self.global_fc = nn.Linear(global_dim, 576)

        # local
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

        # fc
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
        
        global_features = self.global_fc(global_features)  

        local_features = local_features.permute(0, 2, 1)  

        if mask is not None:
            mask = mask.unsqueeze(1)  
            local_features = local_features * mask  

        # 3d-CNN-layer
        local_features1 = self.cnn1(local_features).squeeze(-1)  
        local_features2 = self.cnn2(local_features).squeeze(-1)  
        local_features3 = self.cnn3(local_features).squeeze(-1) 

        combined_features = torch.cat([global_features, local_features1, local_features2, local_features3], dim=1)  # 形状: (batch_size, 576 + 256 + 128 + 64)

        x = self.fc1(combined_features)  # 形状: (batch_size, 512)
        x = self.fc2(x)  # 形状: (batch_size, 256)
        output = self.fc3(x)  # 形状: (batch_size, num_classes)

        return output
