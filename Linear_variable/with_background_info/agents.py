import numpy as np
from collections import namedtuple
import itertools
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN:
    """
    Deep-Q-Network for RL tasks based on gym environments.

    Parameters
    ----------
    alpha : float
        Network optimizer learning rate. (between (0,1])

    gamma : float
        Discount factor. (between [0,1])

    epsilon : float
        Probability that the action is chosen randomly by the epsilon-greedy
        policy. (between [0,1])

    capacity : int
        Capacity for the experience replay on DQN.

    policy : function
        Policy function to perform during learning. (default = epsilon_greedy)

    adaptive : function
        Function to perform adaptive hyperparameters. (default = None)

    double : bool
        If true perform Double DQN to avoid overestimation bias.
        It will double the memory requirements. (default = False)

    qnetwork : class
        Saved network architecture. (default = None)

    loss : pytorch loss function
        Loss function. (default: l1_loss)

    optimizer : pytorch optimizer
        Optimizer for the backpropagation.  (default = RMSprop)

    verbose: bool
        Show training information. (default = True)

    save: str
        The path where the action-state value matrix will be saved.
        (default = None)

    Attributes
    ----------

    n_actions : int
        Number of actions that can be performed in the environment.

    state_dims : int
        Dimension of the environment.

    Transition : named tuple
        Stores the info ('state', 'action', 'next_state', 'reward')
        in memory for the experience replay.
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
        self.save = save

    class DQNetwork(nn.Module):
        """
        Defines the network used.

        Parameters
        ----------
        state_dims : int
            Dimension of the environment.

        n_actions : int
            Number of actions which can be performed.

        """
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
        """
        Defines the memory usage for the experience replay.

        Parameters
        ----------
        capacity : int
            Memory max size.

        position : int
            Index for the current position in memory.
        """
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
            # Takes k=batch_size samples of the memory.
            return random.sample(self.memory, k=batch_size)

        def __len__(self):
            return len(self.memory)

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
            torch.long with the action chosen by the policy.
        """
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
        """
        Train the agent.

        Parameters
        ----------
        env: class
            The environment in which the training has to be made.

        episodes: int
            Number of training episodes.

        batch_size : int
            Number of tuples used for training before weights update.

        target_update : int
            Number of episodes where the target net for
            estimating the Q values remains freezed before updating.

        Returns
        -------
        stats: dict
            Record of rewards in each episode. If save is enabled,
            the dictionary has the rewards and the checkpoints.

        """
        memory = self.ReplayMemory(self.capacity)

        def optimize_model():
            """
            Performs the backpropagation.

            """
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
        """
        Predict which is the best action to choose.

        Parameters
        ----------
        state: np.array
            Current state of the environment.

        Returns
        -------
        action: int
            Action chosen by the policy.
        """
        with torch.no_grad():
            self.Q_net.eval()
            state = torch.from_numpy(state).float()\
                .unsqueeze(0).to(self.device)
            return self.Q_net(state).max(1)[1].view(1, 1).item()
