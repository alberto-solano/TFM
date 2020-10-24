import gym
import numpy as np
from env import RegEnv
import pandas as pd
from Agents import DQN
import matplotlib.pyplot as plt

lr = 0.1
env = RegEnv(lr=lr)
alpha = 0.001
gamma = 0.99
episodes = 30000
batch_size = 128
epsilon = 0.3
decay = True
disc_factor = 0.35
discount = 1-1/(episodes*disc_factor)
def adaptive(self, episode):
    self.epsilon = max(0.01, min(1.0, self.epsilon*discount))

if decay == True:
    epsilon = 1
    adaptive = adaptive
else:
    adaptive = None
    disc_factor = 0

model = DQN(env, alpha, gamma, epsilon, adaptive = adaptive, save=f'./weights_{episodes}_{alpha}_{gamma}_{batch_size}_{decay}_{disc_factor}.pt')

stats = model.train(env, episodes, batch_size=batch_size)

checks = np.array(stats['checkpoints']).astype(int)
rewards = np.array(stats['rewards'])
smooth = pd.DataFrame(rewards).rolling(40).mean()
plt.plot(range(len(rewards)), rewards, alpha=0.5)
plt.plot(range(len(smooth)), smooth)
plt.scatter(checks, smooth.iloc[checks], c='r', marker='.')
plt.show()