import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


class RegEnv(gym.Env):
    """Environment de ajuste con RL a una recta fija para todos los episodios.

    Parámetros
    ----------
    N : int, optional (default = 10)
        Número de puntos a ajustar.

    noise : float, optional (default = 0.3)
        Cantidad de ruido normal que se le añade a los puntos a ajustar,
        los cuales siguen una cierta recta teórica dada por los
        parámetros m y n.

    lr : float, optional (default = 0.1)
        Determina el paso en cada step, lo que se mueven los parámetros
        de la recta.

    Atributos
    ---------
    x, y: float
        Puntos generados con la función _generate_random_problem.

    observation_space : class
        Devuelve el grid o box en función de los parámetros
        teóricos de la recta que siguen los puntos.

    action_space : class
        Devuelve el tipo de espacio utilizado para especificar las
        acciones del environment, se define por convenio en gym.

    actions : np.array
        Array con las acciones disponibles para cada step, en este
        caso incrementar o decrementar individualmente la pendiente
        o la ordenada o también simultáneamente ambas.
    """
    def __init__(self, N: int = 10, noise: float = 0.3, lr: float = 0.1):
        self.N = N
        self.noise = noise
        self.lr = lr
        # Genero los puntos
        self.x, self.y = self._generate_random_problem()
        # Establezco los límites del espacio, en este caso elijo que los
        # valores posibles para la pendiente y la ordenada estén como mucho
        # entre [-5,5].
        # compruebo los límites para el reward al dar un step
        # rmse = []
        #for i in range(1000):
        #    x,y = _generate_random_problem()
        #    actions = np.array([(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1),
        #                                 (-1, -1), (1, -1), (-1, 1)])*0.1
        #    A = np.random.uniform(-5, 5)
        #    B = np.random.uniform(-5, 5)
        #    y_pred1 = A*x + B
        #    action = actions[
        #            np.random.randint(actions.shape[0])]
        #    A = A + action[0]
        #    B = B + action[1]
        #    y_pred2 = A*x + B
        #    metric1 = mean_squared_error(y, y_pred1)
        #    metric2 = mean_squared_error(y, y_pred2)
        #    rmse.append(metric1-metric2)
        self.observation_space = spaces.Box(
            low=np.array([-5, -5, -5, 5, -100], dtype=np.float32),
            high=np.array([5, 5, 5, 5, 100], dtype=np.float32))
        self.action_space = spaces.Discrete(4)
        self.actions = np.array([(1, 0), (0, 1), (-1, 0), (0, -1)])*lr

    def _generate_random_problem(self):
        """Función que se llama y que genera los puntos que tratará de
        ajustarse la recta.
        """
        self.m, self.n = np.random.uniform(-5, 5, 2)
        x = np.linspace(1, self.N, self.N) + \
            np.random.normal(0, self.noise, self.N)
        y = self.m*np.linspace(1, self.N, self.N) + \
            self.n + np.random.normal(0, self.noise, self.N)
        return x, y

    def step(self, action):
        """Función que define el nuevo estado dada una acción de las posibles

        Parámetros
        ----------
        action : np.array
            Acción realizada.

        Devuelve
        --------
        state : np.array
            Nuevo estado de la recta con la forma (A,B).

        reward : float
            Recompensa (RMSE) obtenida en el nuevo estado.

        done : bool
            Indicador de si se ha terminado el episodio o no.

        info : empty
        """
        action = self.actions[action]
        # Me aseguro que si se llega al borde del 'box' no sea posible
        # salirse con una acción de las posibles, si se da ese caso, se repite
        # el valor que se tomaría en el borde. Almaceno el estado anterior
        # y el reward que obtuvimos antes de hacer el step.

        y_prev = self.A * self.x + self.B
        reward_prev = -mean_squared_error(self.y, y_prev)
        state_prev = np.array([self.A, self.B])

        self.A = np.clip(action[0] + self.A, self.observation_space.low[0],
                         self.observation_space.high[0])
        self.B = np.clip(action[1] + self.B, self.observation_space.low[1],
                         self.observation_space.high[1])
        new_state = np.array([self.A, self.B])

        y_ = self.A * self.x + self.B
        reward = -mean_squared_error(self.y, y_)

        state = np.hstack((new_state, state_prev, reward-reward_prev))
        self.t += 1
        if self.t >= 100 or reward >= -1:
            # El -1 se basa en una estadística para el MSE a partir del cual
            # se considera que la recta está sobre los puntos y por tanto se ha
            # ajustado, esto es altamente dependiente del lr (el paso) en cada
            # step y es válido para lr = 0.1 si no hay que cambiarlo.
            done = True
        else:
            done = False
        info = {}
        return state, reward, done, info

    def reset(self):
        """Función para resetear el estado a valores aleatorios

        Devuelve
        --------
        state : np.array
            Nuevo estado inicializado para la recta con la forma (A,B).
        """
        self.t = 0
        self.A = np.random.uniform(-5, 5)
        self.B = np.random.uniform(-5, 5)
        state_prev = np.array([self.A, self.B])
        action = self.actions[
            np.random.randint(self.actions.shape[0])]

        self.A = np.clip(action[0] + self.A, self.observation_space.low[0],
                         self.observation_space.high[0])
        self.B = np.clip(action[1] + self.B, self.observation_space.low[1],
                         self.observation_space.high[1])

        new_state = np.array([self.A, self.B])

        state = np.hstack((new_state, state_prev, 0))

        self.x, self.y = self._generate_random_problem()
        return state

    def render(self):
        """Función para plotear el estado actual (recta) junto a los puntos
        fijos

        Devuelve
        --------
        fig : class
            Figura de matplotlib con el gráfico.
        """
        fig = plt.Figure()
        ax = fig.add_subplot()
        ax.scatter(self.x, self.y, c='k')
        x = np.linspace(min(self.x), max(self.x), 2)
        y = self.A * x + self.B
        ax.plot(x, y, c='r')
        ax.set_ylim(np.min(self.y)-2, np.max(self.y)+2)
        ax.set_xlim(np.min(self.x)-2, np.max(self.x)+2)
        return fig
