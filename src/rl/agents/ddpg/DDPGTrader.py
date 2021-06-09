import torch
import torch.nn.functional as F
import numpy as np

from BaseTrader import BaseTrader
from Critic import Critic
from DeterministicActor import DeterministicActor
from MemoryBuffer import MemoryBuffer
from OUNoise import OUNoise
from Transitions import Transition


class DDPGTrader(BaseTrader):

    def __init__(self, state_dim, action_dim, memory_size, batch_size, ou_θ, ou_σ, n_assets, γ: float = 0.99,
                 τ: float = 5e-3, burn_in_steps: int = 1e4):
        """Initialize."""

        super().__init__(n_assets)
        self.memory = MemoryBuffer(memory_size)
        self.batch_size = batch_size
        self.γ = γ
        self.τ = τ
        self.burn_in_steps = burn_in_steps
        self.t = 0

        # noise
        self.noise = OUNoise(action_dim, θ=ou_θ, σ=ou_σ)

        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # networks
        self.actor = DeterministicActor(state_dim, action_dim).to(self.device)
        self.actor_target = DeterministicActor(state_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim + action_dim).to(self.device)
        self.critic_target = Critic(state_dim + action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizer
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        self.is_test = False

    def act(self, state: np.ndarray):

        if self.t < self.burn_in_steps and not self.is_test:
            selected_action = np.random.uniform(low=-1, high=1)
        else:
            selected_action = self.actor(
                torch.FloatTensor(state).to(self.device)
            ).detach().cpu().numpy()

        # add noise for exploration during training
        if not self.is_test:
            noise = self.noise.sample()
            selected_action = np.clip(selected_action + noise, -1.0, 1.0)

        self.transition = [state, selected_action]

        return selected_action

    def step(self, state, action, reward, next_state, is_done):
        self.t += 1
        self._append_memory(state, action, reward, next_state, is_done)
        if len(self.memory) >= self.batch_size and self.t > self.burn_in_steps:
            transitions = self.memory.sample(self.batch_size)
            actor_loss, critic_loss = self._update_model(transitions)
            return actor_loss, critic_loss
        return 0, 0

    def _update_model(self, transitions):
        batch = Transition(*zip(*transitions))

        state = torch.stack(batch.state)
        action = torch.tensor(batch.action).unsqueeze(1)
        reward = torch.cat(batch.reward).unsqueeze(1)
        next_state = torch.stack(batch.next_state)
        done = torch.tensor(batch.is_done, dtype=torch.uint8).unsqueeze(1)

        masks = 1 - done
        next_action = self.actor_target(next_state)
        next_value = self.critic_target(next_state, next_action)
        curr_return = reward + self.γ * next_value * masks

        # train critic
        values = self.critic(state, action)
        critic_loss = F.mse_loss(values, curr_return)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # train actor
        actor_loss = -self.critic(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # target update
        self._target_soft_update()

        return actor_loss.data, critic_loss.data

    def _target_soft_update(self):

        for t_param, l_param in zip(
                self.actor_target.parameters(), self.actor.parameters()
        ):
            t_param.data.copy_(self.τ * l_param.data + (1.0 - self.τ) * t_param.data)

        for t_param, l_param in zip(
                self.critic_target.parameters(), self.critic.parameters()
        ):
            t_param.data.copy_(self.τ * l_param.data + (1.0 - self.τ) * t_param.data)

    def _append_memory(self, state, action, reward, next_state, is_done):
        state_t = torch.tensor(state, dtype=torch.float, device=self.device)
        action_t = torch.tensor([action], dtype=torch.float, device=self.device)
        reward_t = torch.tensor([reward], dtype=torch.float, device=self.device)
        next_state_t = torch.tensor(next_state, dtype=torch.float, device=self.device)
        is_done_t = torch.tensor([is_done], dtype=torch.bool, device=self.device)
        transition = Transition(state_t, action_t, reward_t, next_state_t, is_done_t)

        self.memory.add(transition)