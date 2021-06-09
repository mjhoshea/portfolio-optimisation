import torch.nn as nn


class DuelingQEstimator(nn.Module):

    def __init__(self, n_states, n_actions, n_hidden=64):
        super(DuelingQEstimator, self).__init__()

        adv_layers = [
            nn.Linear(n_states, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden // 2),
            nn.ReLU(),
            nn.Linear(n_hidden // 2, n_actions)
        ]

        val_layers = [
            nn.Linear(n_states, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden // 2),
            nn.ReLU(),
            nn.Linear(n_hidden // 2, 1)
        ]

        self.adv_model = nn.Sequential(*adv_layers)
        self.val_model = nn.Sequential(*val_layers)

    def forward(self, s):

        adv = self.adv_model(s)

        val = self.val_model(s)

        return val + adv - adv.mean()