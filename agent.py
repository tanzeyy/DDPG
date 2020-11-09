from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DDPGAgent(object):
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.target_model = deepcopy(self.model).to(device)
        self.policy_opt = optim.Adam(params=self.model.policy.parameters(),
                                     lr=config.policy_lr)
        self.value_opt = optim.Adam(params=self.model.value.parameters(),
                                    lr=config.value_lr)

        self.gamma = config.gamma
        self.polyak = config.polyak
        self.noise = config.noise

        self.device = device

    def predict(self, obs):
        obs = torch.from_numpy(obs).to(self.device)
        out = self.model.policy(obs)
        return np.squeeze(out.cpu().detach().numpy())

    def learn(self, obs, act, reward, next_obs, terminal):
        obs = torch.from_numpy(obs).to(self.device)
        act = torch.from_numpy(act).to(self.device)
        reward = torch.from_numpy(reward).to(self.device)
        next_obs = torch.from_numpy(next_obs).to(self.device)
        terminal = torch.from_numpy(terminal).to(self.device)

        value_loss = self._learn_value(obs, act, reward, next_obs, terminal)
        policy_loss = self._learn_policy(obs)
        self._update_target()
        return value_loss.cpu().detach().numpy(), \
            policy_loss.cpu().detach().numpy()

    def _learn_value(self, obs, act, reward, next_obs, terminal):
        # Compute target
        with torch.no_grad():
            next_act = self.target_model.policy(next_obs)
            next_Q = self.target_model.value(next_obs, next_act)
            target_Q = reward + self.gamma * (1.0 - terminal.float()) * next_Q

        # Minimize MSBE
        Q = self.model.value(obs, act)
        loss = F.mse_loss(Q, target_Q)
        self.value_opt.zero_grad()
        loss.backward()
        self.value_opt.step()
        return loss

    def _learn_policy(self, obs):
        act = self.model.policy(obs)
        Q = self.model.value(obs, act)
        loss = torch.mean(-1.0 * Q)
        self.policy_opt.zero_grad()
        loss.backward()
        self.policy_opt.step()
        return loss

    def _update_target(self):
        # Polyak updating
        for target_param, param in zip(self.target_model.value.parameters(),
                                       self.model.value.parameters()):
            target_param.data.copy_(self.polyak * target_param.data +
                                    (1.0 - self.polyak) * param.data)

        for target_param, param in zip(self.target_model.policy.parameters(),
                                       self.model.policy.parameters()):
            target_param.data.copy_(self.polyak * target_param.data +
                                    (1.0 - self.polyak) * param.data)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.target_model = deepcopy(self.model)
