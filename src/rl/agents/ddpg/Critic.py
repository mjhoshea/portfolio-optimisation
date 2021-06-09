import torch
import torch.nn as nn


class Critic(nn.Module):

    def __init__(self, input_dim, wight_init=3e-3):
        super(Critic, self).__init__()

        out = nn.Linear(128, 1)
        out.weight.data.uniform_(-wight_init, wight_init)
        out.bias.data.uniform_(-wight_init, wight_init)

        layers = [
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            out
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=-1)
        return self.model(x)
