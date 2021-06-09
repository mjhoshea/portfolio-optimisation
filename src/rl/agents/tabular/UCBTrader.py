import numpy as np

class UCBTrader(object):

  def __init__(self, name, number_of_arms, bonus_multiplier):
    self._number_of_arms = number_of_arms
    self.c = bonus_multiplier
    self.name = name
    self.Q = None
    self.N = None
    self.t = None
    self.reset()

  def step(self, previous_action, reward):
    self._update_stats(previous_action, reward)
    self.t += 1
    return self._choose_next_action()

  def reset(self):
    self.Q = np.zeros(self._number_of_arms)
    self.N = np.zeros(self._number_of_arms)
    self.t = 1

  def _update_stats(self, prev_a, reward):
    if not prev_a == None:
        self.N[prev_a] += 1
        self.Q[prev_a] += (reward - self.Q[prev_a])/self.N[prev_a]

  def _choose_next_action(self):
    δ = self.c*np.sqrt(np.log(self.t)/self.N)
    aw = self.Q + δ
    aw[np.isnan(aw)] = np.finfo(np.float64).max
    return np.random.choice(np.flatnonzero(aw == aw.max()))