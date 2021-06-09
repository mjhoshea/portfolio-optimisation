import torch.nn as nn


class DeterministicActor(nn.Module):

    def __init__(self, input_dim, output_dim, weight_init=3e-3):
        super(DeterministicActor, self).__init__()

        out = nn.Linear(128, output_dim)
        out.weight.data.uniform_(-weight_init, weight_init)
        out.bias.data.uniform_(-weight_init, weight_init)

        layers = [
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            out,
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, state):
        return self.model(state)