import gym
import numpy as np
from env import RegEnv
import pandas as pd
from Agents import QLearning
import matplotlib.pyplot as plt

lr = 0.1
env = RegEnv(lr=lr)
decay = True
alpha = 0.3
gamma = 0.7
epsilon = 0.15
episodes = 50000
disc_factor = 0.15
discount = 1-1/(episodes*disc_factor)
def adaptive(self, episode):
    self.epsilon = max(0.01, min(1.0, self.epsilon*discount))

OBSERVATION_DIMS = env.observation_space.shape[0]
LOWER = env.observation_space.low
HIGHER = env.observation_space.high

RESOLUTIONS = list(((HIGHER-LOWER)/lr).astype('int'))
ALL_POSSIBLE_STATES = np.array(np.meshgrid(
    *[range(res) for res in RESOLUTIONS])).T.reshape(-1, OBSERVATION_DIMS)
STATE_SPACE = {tuple(j): i for i, j in enumerate(ALL_POSSIBLE_STATES)}

def discretize(state):
    for i in range(OBSERVATION_DIMS):
        state[i] = np.digitize(state[i],
                               np.linspace(LOWER[i],
                                           HIGHER[i],
                                           RESOLUTIONS[i]-1))
    return STATE_SPACE[tuple(state.astype(int))]

if decay == True:
    epsilon = 1
    adaptive = adaptive
else:
    adaptive = None
    disc_factor = 0

model = QLearning(alpha, gamma, epsilon, adaptive=adaptive, discretize=discretize, double=True, verbose=True, save=f'./weights_{episodes}_{alpha}_{gamma}_{decay}_{disc_factor}.npy')

stats = model.train(env, episodes)

checks = np.array(stats['checkpoints']).astype(int)
rewards = np.array(stats['rewards'])
smooth = pd.DataFrame(rewards).rolling(40).mean()
plt.plot(range(len(rewards)), rewards, alpha=0.5)
plt.plot(range(len(smooth)), smooth)
plt.scatter(checks, smooth.iloc[checks], c='r', marker='.')
plt.show()