import torch.nn as nn

class ActionProjector(nn.Module):
    def __init__(self, in_dim, out_dim=1024):
        super(ActionProjector, self).__init__()
        self.global_1d_pool = nn.AdaptiveAvgPool1d(1)
        self.mlps = nn.ModuleList([
            # nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        )

    def forward(self, x):
        x = self.global_1d_pool(x.permute(1, 0)).permute(1, 0)
        for mlp in self.mlps:
            x = mlp(x)
        return x


class FiLM(nn.Module):
    def __init__(self, feature_dim, condition_dim):
        super(FiLM, self).__init__()
        self.scale_fc = nn.Linear(condition_dim, feature_dim)
        self.shift_fc = nn.Linear(condition_dim, feature_dim)

        nn.init.zeros_(self.scale_fc.weight)
        nn.init.zeros_(self.scale_fc.bias)
        nn.init.zeros_(self.shift_fc.weight)
        nn.init.zeros_(self.shift_fc.bias)

    def forward(self, x, condition):
        # 计算缩放和偏移参数
        scale = self.scale_fc(condition)
        shift = self.shift_fc(condition)

        # 应用 FiLM 调制
        return x * (1 + scale) + shift
