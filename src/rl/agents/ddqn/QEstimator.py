import torch.nn as nn


class QEstimator(nn.Module):

    def __init__(self, n_states, n_actions, n_hidden=64):
        super(QEstimator, self).__init__()
        layers = [
            nn.Linear(n_states, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden // 2),
            nn.ReLU(),
            nn.Linear(n_hidden // 2, n_actions)
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, s):
        qs = self.model(s)
        return qs
