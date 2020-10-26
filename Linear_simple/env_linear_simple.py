import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt


class RegEnv(gym.Env):
    """Environment de ajuste con RL a una recta fija para todos los episodios.

    Parámetros
    ----------
    m : float, optional (default = 1)
        Pendiente teórica que siguen los puntos.

    n : float, optional (default = 1)
        Pendiente teórica que siguen los puntos.

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
    def __init__(self, m: float = 1, n: float = 0, N: int = 10,
                 noise: float = 0.3, lr: float = 0.1):
        self.m = m
        self.n = n
        self.N = N
        self.noise = noise
        self.lr = lr
        # Genero los puntos
        self.x, self.y = self._generate_random_problem()
        # Establezco los límites del espacio, en este caso elijo que los
        # valores posibles para la pendiente y la ordenada estén como mucho
        # 5 unidades por encima o por debajo del valor que marcaría la
        # distribución de puntos generados.
        self.observation_space = spaces.Box(
            low=np.array([m-5, n-5], dtype=np.float32),
            high=np.array([m+5, n+5], dtype=np.float32))
        self.action_space = spaces.Discrete(8)
        self.actions = np.array([(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1),
                                 (-1, -1), (1, -1), (-1, 1)])*lr

    def _generate_random_problem(self):
        """Función que se llama al comienzo y que genera los puntos fijos a los que
        se ajustará la recta.
        """
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
        # el valor que se tomaría en el borde.
        self.A = np.clip(action[0] + self.A, self.observation_space.low[0],
                         self.observation_space.high[0])
        self.B = np.clip(action[1] + self.B, self.observation_space.low[1],
                         self.observation_space.high[1])
        state = np.array([self.A, self.B])
        y_ = self.A * self.x + self.B
        reward = -np.sum(((y_ - self.y)**2).mean())
        self.t += 1
        if self.t >= 200 or reward >= -0.4:
            # El -0.4 se basa en una estadística para el RMSE a partir del cual
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
        self.A = np.random.uniform(self.m-5, self.m+5)
        self.B = np.random.uniform(self.n-5, self.n+5)
        state = np.array([self.A, self.B])
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
