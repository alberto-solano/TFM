import numpy as np
from env_linear_simple import RegEnv
import pandas as pd
from agents import DQN
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
# Adaptive modifica el epsilon creando un decaimiento exponencial del mismo
# desde 1 hasta un mínimo de 0.01, para ver el decaimiento se puede
# representar: plt.plot(range(episodes),discount**np.array(range(episodes)))
# De manera que si el discount factor es mayor, el decaimiento es más suave.
# De esta manera se incentiva la exploración frente a la explotación
# en los primeros episodios.


if decay is True:
    epsilon = 1
    adaptive = adaptive
else:
    adaptive = None
    disc_factor = 0
# Si especifico decay = False, entonces no se utiliza el decaimiento para
# épsilon, y este parámetro queda fijo con el valor que se ponga arriba.

model = DQN(env, alpha, gamma, epsilon, adaptive=adaptive,
            save=f'./weights_{episodes}_{alpha}_{gamma}_\
            {batch_size}_{decay}_{disc_factor}.pt')

stats = model.train(env, episodes, batch_size=batch_size)
checks = np.array(stats['checkpoints']).astype(int)
rewards = np.array(stats['rewards'])
smooth = pd.DataFrame(rewards).rolling(40).mean()
# Plot de las curvas de entrenamiento.
plt.plot(range(len(rewards)), rewards, alpha=0.5)
plt.plot(range(len(smooth)), smooth)
plt.scatter(checks, smooth.iloc[checks], c='r', marker='.')
plt.show()
