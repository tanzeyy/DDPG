import argparse
import time

import gym
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from agent import DDPGAgent
from config import Config
from model import Model
from replay_memory import ReplayMemory


class EnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, act):
        act = np.expand_dims(act, 0)
        next_obs, reward, done, info = self.env.step(act)
        return next_obs.astype('float32'), reward.astype('float32'), done, info

    def reset(self):
        obs = self.env.reset()
        return obs.astype('float32')


SEED = 864


def run(config):
    torch.manual_seed(SEED)
    writer = SummaryWriter('./log')

    env = gym.make('Pendulum-v0')
    env = EnvWrapper(env)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    max_action = env.action_space.high[0]

    # Build model
    model = Model(obs_dim[0], act_dim[0], max_action)

    # Init agent
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent = DDPGAgent(model, config, device)

    # Init RPM
    rpm = ReplayMemory(int(1e6), obs_dim, act_dim, (1, ))

    # Run episodes
    pbar = tqdm(range(3000))
    for episode in pbar:
        value_loss, policy_loss = torch.tensor([0]), torch.tensor([0])
        obs, done = env.reset(), False
        episode_reward = []
        while not done:
            act = agent.predict(obs)
            # Add noise
            act += np.random.normal(config.noise.loc,
                                    config.noise.scale,
                                    size=act.shape)

            act = act.clip(-1., 1.)
            next_obs, reward, done, _ = env.step(act)
            episode_reward.append(reward)
            rpm.append(obs, act, reward, next_obs, done)
            obs = next_obs

        # Update
        policy_losses, value_losses = [], []
        for _ in range(len(episode_reward)):
            batch = rpm.sample_batch(128)
            value_loss, policy_loss = agent.learn(*batch)
            value_losses.append(value_loss)
            policy_losses.append(policy_loss)

        # Log metrics
        writer.add_scalar('episode_length', len(episode_reward), episode)
        writer.add_scalar('episode_reward', sum(episode_reward), episode)
        writer.add_scalar('value_loss', np.mean(value_losses), episode)
        writer.add_scalar('policy_loss', np.mean(policy_losses), episode)


if __name__ == "__main__":
    config = Config({
        "gamma": 0.99,
        "polyak": 0.995,
        "noise": {
            "loc": 0,
            "scale": 0.1
        },
        "policy_lr": 0.0001,
        "value_lr": 0.001
    })
    run(config)
