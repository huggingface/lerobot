import torch.nn as nn


def mlp(
        input_dim,
        hidden_dim,
        output_dim,
        hidden_depth,
        output_mod=None,
        batchnorm=False,
        activation=nn.ReLU,
):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), activation(inplace=True)] if not batchnorm else [
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            activation(inplace=True)
        ]

        for _ in range(hidden_depth - 1):
            mods += (
                [nn.Linear(hidden_dim, hidden_dim), activation(inplace=True)]
                if not batchnorm
                else [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    activation(inplace=True),
                ]
            )
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


class MLP(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            output_dim,
            hidden_depth,
            output_mod=None,
            batchnorm=False,
            activation=nn.ReLU,
    ):
        super().__init__()
        self.trunk = mlp(
            input_dim,
            hidden_dim,
            output_dim,
            hidden_depth,
            output_mod,
            batchnorm=batchnorm,
            activation=activation,
        )
        self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)
