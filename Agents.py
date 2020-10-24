import numpy as np
import pandas as pd
from collections import defaultdict, namedtuple
import itertools
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


# ONE-STEP, TABULAR, MODEL-FREE, TD METHODS


class QLearning:
    """
    Q-learning off-policy Temporal Difference Control.

    Parameters
    ----------
    alpha : float
        Temporal difference learning rate. (between (0,1])

    gamma : float
        Discount factor. (between [0,1])

    epsilon : float
        Probability that the action is chosen randomly by the epsilon-greedy
        policy. (between [0,1])

    policy : function
         Policy function to perform during learning. (default = epsilon_greedy)

    adaptive : function
        Function to perform adaptive hyperparameters. (default = None)

    discretize: function
        Discretization function to discretize continuous states.
        (default = None)

    double = bool
        If true perform Double Q Learning to avoid maximization bias.
        It will double the memory requirements. (default = False)

    verbose: bool
        Show training information. (default = True)

    save: str
        The path where the action-state value matrix will be saved.
        (default = None)

    Attributes
    ----------
    Q: ndarray
        The action-state value matrix. (n_states x n_actions)

    n_actions: int
        Number of actions that can be performed in the environment.
    """

    def __init__(self, alpha, gamma, epsilon, policy=None, adaptive=None,
                 discretize=None, double=False, verbose=True, save=None):
        self.alpha = alpha
        self.gamma = gamma
        self.adaptive = adaptive
        self.epsilon = epsilon
        self.Q = None
        if double:
            self.Q2 = None
        self.policy = policy if policy is not None else self.epsilon_greedy
        self.n_actions = None
        self.discretize = discretize if discretize is not None else lambda x: x
        self.double = double
        self.verbose = verbose
        self.save = save

    def epsilon_greedy(self, state):
        """
        Epsilon-greedy policy.

        Chose a random action with probability gamma
        or chose the best action with probability (1 - gamma).

        Parameters
        ----------
        state: int
            Current state of the environment.

        Returns
        -------
        policy_action: int
            Action chosen by the policy.
        """

        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.n_actions)
        else:
            if self.double:
                Q = self.Q[state] + self.Q2[state]
            else:
                Q = self.Q[state]
            return np.argmax(Q)

    def train(self, env, episodes):
        """
        Train the agent.

        Parameters
        ----------
        env: class
            The environment in which the training has to be made.

        episodes: int
            Number of training episodes.

        Returns
        -------
        stats: dict
            Record of rewards in each episode. If save is enabled,
            the dictionary has the rewards and the checkpoints.

        """

        self.n_actions = env.action_space.n

        self.Q = defaultdict(lambda: np.zeros(self.n_actions))
        if self.double:
            self.Q2 = defaultdict(lambda: np.zeros(self.n_actions))

        if self.save:
            stats = {'checkpoints': [], 'rewards': []}
        else:
            stats = {'rewards': []}

        max_returns = -9999999999999999

        for episode in range(episodes):
            if self.adaptive is not None:
                self.adaptive(self, episode)

            state = self.discretize(env.reset())

            returns = 0

            for t in itertools.count():

                action = self.policy(state)
                state_, reward, done, info = env.step(action)
                state_ = self.discretize(state_)

                if not self.double:
                    self.Q[state][action] += self.alpha * \
                        (reward + self.gamma * np.max(self.Q[state_]) -
                            self.Q[state][action])

                else:
                    if np.random.randint(0, 2):
                        self.Q[state][action] += self.alpha *\
                            (reward + self.gamma *
                                self.Q2[state_][np.argmax(self.Q[state_])] -
                             self.Q[state][action])
                    else:
                        self.Q2[state][action] += self.alpha *\
                            (reward + self.gamma *
                                self.Q[state_][np.argmax(self.Q2[state_])] -
                             self.Q2[state][action])

                returns += reward

                if done:
                    break

                state = state_

            stats['rewards'].append(returns)

            if self.verbose is True:
                print('Episode:', episode, "; Returns:", returns)

            if self.save is not None:
                if len(stats['rewards']) >= 100 and self.epsilon < 0.3:
                    mean_returns = np.mean(stats['rewards'][-100:])
                    if mean_returns >= max_returns:
                        if self.double:
                            np.save(self.save, [dict(self.Q), dict(self.Q2)])
                        else:
                            np.save(self.save, dict(self.Q))
                        print("Saved state at episode:", episode,
                              "with mean returns:", mean_returns)
                        stats['checkpoints'].append(episode)
                        max_returns = mean_returns

        return stats

    def load(self, path):
        """
        Load a saved action-state value matrix.

        Parameters
        ----------
        path: str
            The path where is the .npy file with the action-state value matrix
            saved.

        """
        self.Q = defaultdict(lambda: np.zeros(self.n_actions))
        if self.double:
            self.Q2 = defaultdict(lambda: np.zeros(self.n_actions))
            Q1, Q2 = np.load(path, allow_pickle='TRUE')
            for i, j in Q2.items():
                self.Q2[i] = j
        else:
            Q1 = np.load(path, allow_pickle='TRUE').item()
        for i, j in Q1.items():
            self.Q[i] = j

    def predict(self, state):
        """
        Predict which is the best action to choose.

        Parameters
        ----------
        state: int
            Current state of the environment.

        Returns
        -------
        action: int
            Action chosen by the policy.
        """

        state = self.discretize(state)
        if self.double:
            return np.argmax(self.Q[state] + self.Q2[state])
        else:
            return np.argmax(self.Q[state])


class ExpectedSARSA:
    """
    Expected SARSA off-policy Temporal Difference Control.

    Parameters
    ----------
    alpha : float
        Temporal difference learning rate. (between (0,1])

    gamma : float
        Discount factor. (between [0,1])

    epsilon : float
        Probability that the action is chosen randomly by the
        epsilon-greedy policy. (between [0,1])

    policy : function
         Policy function to perform during learning. (default = epsilon_greedy)

    adaptive : function
        Function to perform adaptive hyperparameters. (default = None)

    discretize: function
        Discretization function to discretize continuous states.
        (default = None)

    double = bool
        If true perform Double Expected SARSA to avoid maximization bias.
        It will double the memory requirements. (default = False)

    verbose: bool
        Show training information. (default = True)

    Attributes
    ----------
    Q: ndarray
        The action-state value matrix. (n_states x n_actions)

    n_actions: int
        Number of actions that can be performed in the environment.
    """

    def __init__(self, alpha, gamma, epsilon, policy=None, adaptive=None,
                 discretize=None, double=False, verbose=True):
        self.alpha = alpha
        self.gamma = gamma
        self.adaptive = adaptive
        self.epsilon = epsilon
        self.Q = None
        if double:
            self.Q2 = None
        self.policy = policy if policy is not None else self.epsilon_greedy
        self.n_actions = None
        self.discretize = discretize if discretize is not None else lambda x: x
        self.double = double
        self.verbose = verbose

    def epsilon_greedy(self, state):
        """
        Epsilon-greedy policy.

        Chose a random action with probability gamma
        or chose the best action with probability (1 - gamma).

        Parameters
        ----------
        state: int
            Current state of the environment.

        Returns
        -------
        policy_action: int
            Action chosen by the policy.
        """

        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.n_actions)
        else:
            if self.double:
                Q = self.Q[state] + self.Q2[state]
            else:
                Q = self.Q[state]
            return np.argmax(Q)

    def train(self, env, episodes):
        """
        Train the agent.

        Parameters
        ----------
        env: class
            The environment in which the training has to be made.

        episodes: int
            Number of training episodes.

        Returns
        -------
        stats: list
            Record of rewards in each episode.

        """

        self.n_actions = env.action_space.n

        self.Q = defaultdict(lambda: np.zeros(self.n_actions))
        if self.double:
            self.Q2 = defaultdict(lambda: np.zeros(self.n_actions))

        stats = []

        for episode in range(episodes):
            if self.adaptive is not None:
                self.adaptive(self, episode)

            state = self.discretize(env.reset())

            returns = 0
            for t in itertools.count():

                action = self.policy(state)
                state_, reward, done, info = env.step(action)
                state_ = self.discretize(state_)

                if not self.double:
                    expectation = np.sum((np.exp(self.Q[state_]) /
                                         np.sum(np.exp(self.Q[state_]))) *
                                         self.Q[state_])
                    self.Q[state][action] += self.alpha * \
                        (reward + self.gamma * expectation -
                         self.Q[state][action])

                else:

                    Q = (self.Q[state_] + self.Q2[state_]) / 2

                    if np.random.randint(0, 2):
                        expectation = np.sum((np.exp(Q) /
                                             np.sum(np.exp(Q))) *
                                             self.Q2[state_])
                        self.Q[state][action] += self.alpha * (
                                reward + self.gamma * expectation -
                                self.Q[state][action])

                    else:
                        expectation = np.sum((np.exp(Q) /
                                             np.sum(np.exp(Q))) *
                                             self.Q[state_])
                        self.Q2[state][action] += self.alpha * (
                                reward + self.gamma * expectation -
                                self.Q2[state][action])

                returns += reward

                if done:
                    break

                state = state_

            if self.verbose is True:
                print('Episode:', episode, "; Returns:", returns)

            stats.append(returns)

        return stats

    def predict(self, state):
        """
        Predict which is the best action to choose.

        Parameters
        ----------
        state: int
            Current state of the environment.

        Returns
        -------
        action: int
            Action chosen by the policy.
        """

        state = self.discretize(state)
        if self.double:
            return np.argmax(self.Q[state] + self.Q2[state])
        else:
            return np.argmax(self.Q[state])


# APPROXIMATE, ONE-STEP, MODEL-FREE, TD METHODS


class DQN:
    """

    """

    def __init__(self, env, alpha, gamma, epsilon, capacity=10000, policy=None,
                 adaptive=None, double=False, qnetwork=None,
                 loss=F.smooth_l1_loss, optimizer=None,
                 verbose=True, save=None):
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")
        self.alpha = alpha
        self.gamma = gamma
        self.adaptive = adaptive
        self.double = double
        self.epsilon = epsilon
        self.capacity = capacity
        self.n_actions = env.action_space.n
        self.state_dims = env.observation_space.shape[0]
        if qnetwork is None:
            self.Q_net = self.DQNetwork(self.state_dims,
                                        self.n_actions).to(self.device)
            self.target_net = self.DQNetwork(self.state_dims,
                                             self.n_actions).to(self.device)
        else:
            self.Q_net = qnetwork.to(self.device)
            self.target_net = qnetwork.to(self.device)
        self.target_net.load_state_dict(self.Q_net.state_dict())
        self.target_net.eval()
        if optimizer is None:
            self.optimizer = torch.optim.RMSprop(self.Q_net.parameters(),
                                                 lr=self.alpha)
        else:
            self.optimizer = optimizer
        self.policy = policy if policy is not None else self.epsilon_greedy
        self.verbose = verbose
        self.Transition = namedtuple('Transition', ('state', 'action',
                                                    'next_state', 'reward'))
        self.loss = loss
        self.steps_done = 0
        self.save = save

    class DQNetwork(nn.Module):
        def __init__(self, state_dims, n_actions):
            super().__init__()
            self.l1 = nn.Linear(state_dims, 64)
            self.l2 = nn.Linear(64, 64)
            self.l3 = nn.Linear(64, n_actions)

        def forward(self, x):
            x = F.relu(self.l1(x))
            x = F.relu(self.l2(x))
            x = self.l3(x)
            return x

    class ReplayMemory:
        def __init__(self, capacity):
            self.Transition = namedtuple('Transition',
                                         ('state', 'action',
                                          'next_state', 'reward'))
            self.capacity = capacity
            self.memory = []
            self.position = 0

        def push(self, *args):
            """Saves a transition."""
            if len(self.memory) < self.capacity:
                self.memory.append(None)
            self.memory[self.position] = self.Transition(*args)
            self.position = (self.position + 1) % self.capacity

        def sample(self, batch_size):
            return random.sample(self.memory, k=batch_size)

        def __len__(self):
            return len(self.memory)

    def epsilon_greedy(self, state):
        sample = np.random.random()
        if sample > self.epsilon:
            self.Q_net.eval()
            with torch.no_grad():
                return self.Q_net(state).max(1)[1].view(1, 1)
            self.Q_net.train()
        else:
            return torch.tensor([[np.random.randint(self.n_actions)]],
                                device=self.device, dtype=torch.long)

    def train(self, env, episodes, batch_size, target_update=4):

        memory = self.ReplayMemory(self.capacity)

        self.steps_done = 0

        def optimize_model():
            if len(memory) < batch_size:
                return

            transitions = memory.sample(batch_size)
            batch = self.Transition(*zip(*transitions))

            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                    batch.next_state)),
                                          device=self.device,
                                          dtype=torch.bool)
            non_final_next_states = \
                torch.cat([s for s in batch.next_state if s is not None])

            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            state_action_values = \
                self.Q_net(state_batch).gather(1, action_batch)

            next_state_values = torch.zeros(batch_size, device=self.device)

            if self.double:
                with torch.no_grad():
                    action_ = self.Q_net(non_final_next_states).max(1)[1]
                next_state_values[non_final_mask] = \
                    self.target_net(non_final_next_states).detach()\
                    .gather(1, action_.unsqueeze(1)).squeeze(1)
            else:
                next_state_values[non_final_mask] = \
                    self.target_net(non_final_next_states).max(1)[0].detach()

            expected_state_action_values = \
                (next_state_values * self.gamma) + reward_batch

            loss = self.loss(state_action_values,
                             expected_state_action_values.unsqueeze(1))

            self.optimizer.zero_grad()
            loss.backward()
            for param in self.Q_net.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

        if self.save:
            stats = {'checkpoints': [], 'rewards': []}
        else:
            stats = {'rewards': []}

        max_returns = -9999999999999999

        for i_episode in range(episodes):

            if self.adaptive is not None:
                self.adaptive(self, i_episode)

            returns = 0
            state = env.reset()
            state = torch.from_numpy(state).float()\
                .unsqueeze(0).to(self.device)
            for t in itertools.count():
                action = self.policy(state)
                next_state, reward, done, _ = env.step(action.item())
                next_state = torch.from_numpy(next_state).float()\
                    .unsqueeze(0).to(self.device)
                reward = torch.tensor([reward], device=self.device).float()
                returns += reward
                if done:
                    next_state = None

                memory.push(state, action, next_state, reward)

                state = next_state

                optimize_model()
                if done:
                    break
            stats['rewards'].append(returns.item())

            if i_episode % target_update == 0:
                self.target_net.load_state_dict(self.Q_net.state_dict())

            if self.verbose:
                print("Episode:", i_episode, "Returns:", returns.item())

            if self.save is not None:
                if len(stats['rewards']) >= 40:
                    mean_returns = np.mean(stats['rewards'][-40:])
                    if mean_returns >= max_returns:
                        max_returns = mean_returns
                        torch.save(self.Q_net.state_dict(), self.save)
                        print("Saved state at episode:", i_episode,
                              "with mean returns:", mean_returns)
                        stats['checkpoints'].append(i_episode)
        return stats

    def predict(self, state):
        with torch.no_grad():
            self.Q_net.eval()
            state = torch.from_numpy(state).float()\
                .unsqueeze(0).to(self.device)
            return self.Q_net(state).max(1)[1].view(1, 1).item()


class TWAP:
    def __init__(self):
        self.H = 0
        self.V = 0
        self.ratio = 0
        self.index = []
        self.actions = []

    def train(self, env):
        self.H = env.H - pd.to_timedelta(1, unit='s')
        self.V = env.V
        self.ratio = self.V/self.H.delta
        self.time_step = env.time_step.delta
        self.actions = env.posible_actions

    def predict(self, state):
        volume = self.ratio * self.time_step
        action = np.where(self.actions >= volume)[0]
        if action.size == 0:
            action = len(self.actions) - 1
        else:
            action = action[0]
        return action


class POV:
    def __init__(self, percent):
        self.percent = percent
        self.actions = []

    def train(self, env):
        self.actions = env.posible_actions

    def predict(self, env):
        state = env.orderbook.iloc[env.istep][env.SizeCols]
        volume = np.sum(state) * self.percent
        action = np.where(self.actions >= volume)[0]
        if action.size == 0:
            action = len(self.actions) - 1
        else:
            action = action[0]
        return action
