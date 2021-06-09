import random

import torch
import torch.nn as nn

from BaseTrader import BaseTrader
from DuelingQEstimator import DuelingQEstimator
from MemoryBuffer import MemoryBuffer
from QEstimator import QEstimator
from Transitions import Transition


class QSingleAssetTrader(BaseTrader):
    TARGET_UPDATE_STRATEGY = ['SOFT', 'HARD']

    def __init__(self,
                 n_states,
                 n_actions,
                 args):
        super().__init__(args)

        assert args.target_update in self.TARGET_UPDATE_STRATEGY

        # basic runtime params
        self.target_update = args.target_update
        self.τ = args.τ
        self.γ = args.γ
        self.evaluate = args.evaluate
        self.batch_size = args.batch_size
        self.n_actions = n_actions
        self.update_cadence = args.update_cadence
        self.decay_start_time = args.decay_start_time
        self.best_weights = []
        self.burn_in_period = args.burn_in_period
        self.q_network_hidden_size = args.q_network_hidden_size
        self.start_ts = args.start_ts

        # for function approximation
        self.memory = MemoryBuffer(args.buffer_size)
        if args.dueling:
            self.online_q_estimator = DuelingQEstimator(n_states, n_actions, n_hidden=self.q_network_hidden_size)
            self.target_q_estimator = DuelingQEstimator(n_states, n_actions, n_hidden=self.q_network_hidden_size)
        else:
            self.online_q_estimator = QEstimator(n_states, n_actions, n_hidden=self.q_network_hidden_size)
            self.target_q_estimator = QEstimator(n_states, n_actions, n_hidden=self.q_network_hidden_size)

        self.optimiser = torch.optim.Adam(self.online_q_estimator.parameters(), lr=args.lr)
        self.criterion = nn.SmoothL1Loss()
        self._hard_target_update()

    def act(self, state):
        if random.random() < self.ϵ and not self.evaluate:
            return random.randint(0, self.n_actions - 1)
        else:
            with torch.no_grad():
                q_values = self.online_q_estimator.forward(torch.Tensor([state]))
                greedy_action = torch.argmax(q_values).cpu().item()
                return greedy_action

    def step(self, state, action, reward, next_state, is_done):
        self.timestep += 1
        self._append_memory(state, action, reward, next_state, is_done)

        if self.timestep < self.start_ts or len(self.memory) < self.batch_size:
            return

        self.decay_epsilon()

        transitions = self.memory.sample(self.batch_size)
        self._update_q_approximator(transitions)

    def _soft_target_update(self):
        for target_param, online_param in zip(self.target_q_estimator.parameters(),
                                              self.online_q_estimator.parameters()):
            new_target_param = self.τ * online_param + (1 - self.τ) * target_param
            target_param.data.copy_(new_target_param)

    def _hard_target_update(self):
        self.target_q_estimator.load_state_dict(self.online_q_estimator.state_dict())

    def _update_q_approximator(self, transitions):
        batch = Transition(*zip(*transitions))

        state = torch.stack(batch.state)
        action = torch.cat(batch.action)
        reward = torch.cat(batch.reward)
        next_state = torch.stack(batch.next_state)
        not_done = 1 - torch.tensor(batch.is_done, dtype=torch.uint8)


        current_q = self.online_q_estimator.forward(state.unsqueeze(1)).gather(1, action.unsqueeze(1))
        online_next_actions = self.online_q_estimator.forward(next_state.unsqueeze(1)).max(1).indices.detach()
        target_q = self.target_q_estimator.forward(next_state.unsqueeze(1)).detach().gather(
            1, online_next_actions.unsqueeze(1)).squeeze(1)

        td_target = reward + self.γ * target_q * not_done

        loss = self.criterion(current_q, td_target.unsqueeze(1))

        self.optimiser.zero_grad()
        loss.backward()

        # torch.nn.utils.clip_grad.clip_grad_norm_(self.online_q_estimator.parameters(), 10)

        for param in self.online_q_estimator.parameters():
            param.grad.data.clamp_(-1e1, 1e1)

        self.optimiser.step()

        if self.target_update == 'HARD' and not self.timestep % self.update_cadence:
            self._hard_target_update()
        else:
            self._soft_target_update()

    def _append_memory(self, state, action, reward, next_state, is_done):
        state_t = torch.tensor(state, dtype=torch.float, device='cpu')
        action_t = torch.tensor([action], dtype=torch.long, device='cpu')
        reward_t = torch.tensor([reward], dtype=torch.float, device='cpu')
        next_state_t = torch.tensor(next_state, dtype=torch.float, device='cpu')
        is_done_t = torch.tensor([is_done], dtype=torch.bool, device='cpu')
        transition = Transition(state_t, action_t, reward_t, next_state_t, is_done_t)

        self.memory.add(transition)

    # def reset_exploration(self):
    #     self.ϵ = self.ϵ_reset
