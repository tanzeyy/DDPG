import numpy as np


class ReplayMemory(object):
    def __init__(self, max_size, obs_size, act_size, reward_size):
        self.max_size = int(max_size)

        self.obs = np.zeros((max_size, ) + obs_size, dtype='float32')
        self.act = np.zeros((max_size, ) + act_size, dtype='float32')
        self.reward = np.zeros((max_size, ) + reward_size, dtype='float32')
        self.next_obs = np.zeros((max_size, ) + obs_size, dtype='float32')
        self.terminal = np.zeros((max_size, 1), dtype='bool')

        self._cur_size = 0
        self._cur_pos = 0

    def sample_batch(self, batch_size=32):
        batch_idx = np.random.randint(self._cur_size, size=batch_size)
        obs = self.obs[batch_idx]
        act = self.act[batch_idx]
        reward = self.reward[batch_idx]
        next_obs = self.next_obs[batch_idx]
        terminal = self.terminal[batch_idx]

        return obs, act, reward, next_obs, terminal

    def append(self, obs, act, reward, next_obs, terminal):
        if self._cur_size < self.max_size:
            self._cur_size += 1

        self.obs[self._cur_pos] = obs
        self.act[self._cur_pos] = act
        self.reward[self._cur_pos] = reward
        self.next_obs[self._cur_pos] = next_obs
        self.terminal[self._cur_pos] = terminal

        self._cur_pos = (self._cur_pos + 1) % self.max_size

    def size(self):
        return self._cur_size
