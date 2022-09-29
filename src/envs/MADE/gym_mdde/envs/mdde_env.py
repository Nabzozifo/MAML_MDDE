# Let's code our multi-agent environment.
import numpy as np
import random
import gym
from gym.spaces import Discrete, MultiDiscrete
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from gym.spaces import Discrete, MultiDiscrete, Dict


def finish_mission(u, xu, yu, x, y, t, T):
    if t >= T-1:
        if x[u] == xu and y == yu:
            return 1
        else:
            return 0
    else:
        return 0


def computeDistance(x1, x2, y1, y2, z1, z2):
    return np.sqrt((x1 - x2)**2+(y1 - y2)**2+(z1 - z2)**2)


def computeRate(B, noise, beta0, P, fading, x1, x2, y1, y2, z1, z2, alpha):
    return B * np.log2(1 + ((beta0 * P * fading**2) / ((computeDistance(x1, x2, y1, y2, z1, z2)**alpha) * noise**2)))


def objective2cont(a, u, t, xu, yu, zu, x, y, Tui, U, I, lambda1, lambda2, lambda3, BIoT, noise, beta0, PIot, fadingIot, Xi, Yi, H, T):
    if t == 0:
        return 0
    else:
        obj1 = sum(a[:, u, t] * computeRate(BIoT[:], noise, beta0, PIot[:,
                   u, t], fadingIot[:, u, t], Xi[:], xu, Yi[:], yu, 0, zu, alpha=2))
        obj2 = sum((Tui[:, u, t-1] + 1) * (1 - a[:, u, t-1]))
        # (1 / np.exp(np.sqrt(computeDistance(x[u], xu, y, yu, 0, zu))))
        obj3 = finish_mission(u, xu, yu, x, y, t, T)
    return ((lambda1 / I * U) * obj1) + ((-lambda2 / I * U) * obj2) + ((lambda3 / I * U) * obj3)


env_config = {

    'num_drones': 4,
    'num_IoT_devices': 500,
    'max_episode_steps': 200,
    'lambda1': 0.3,
    'lambda2': 0.4,
    'lambda3': 0.3,
    'beta0': 1,
    'deltat': 1,  # time step length
    'V': 15,  # vitesse du drone
    'dmin': 15,
    'xumin': 0,
    'xumax': 30,
    'yumin': 1,
    'yumax': 30,
    'alpha': 2,
    'noises': 1e-15,
    'Ri': 120000,
    'zumin': 80,
    'zumax': 100,
}


class MultiAgentDrone(gym.Env):
    def __init__(self, lambda1=0.3, lambda2=0.3, lambda3=0.4, task={}, env_config=env_config):
        self.task = task

        self.width = env_config.get('xumax')
        self.heigh = env_config.get('yumax')

        # objective function parameters
        self.lambda1 = env_config.get('lambda1')
        self.lambda2 = env_config.get('lambda2')
        self.lambda3 = env_config.get('lambda3')

        # defining number of RL agent
        self.n_agents = env_config.get('num_drones')
        self.I = env_config.get('num_IoT_devices')  # number of Iot devices

        # task parameters
        self.Ri = self.task.get('Ri', env_config.get('Ri'))
        self.max_episode_steps = self.task.get(
            'max_episode_steps', env_config.get('max_episode_steps'))
        self.T = self.max_episode_steps
        # rate parameters
        self.alpha = env_config.get('alpha')
        self.beta0 = env_config.get('beta0')
        self.deltat = env_config.get('deltat')
        self.noises = env_config.get('noises')
        # Puissance transmit of IoTs
        self.PIot = np.random.rand(self.I, self.n_agents, self.T)
        self.fadingIot = np.random.rand(
            self.I, self.n_agents, self.T)  # fading for IoTs devices
        self.BIoT = np.random.randint(
            1500, 1700, size=self.I)  # IoTs bandwidth

        self.y = self.heigh  # CDS position y
        self.x = np.random.randint(
            0, self.width, size=self.n_agents)  # CDS position x

        self.initial_y = 0  # initial drone position y
        self.initial_x = np.random.randint(
            0, self.width, size=self.n_agents)  # initial drone position x
        while len(set(self.initial_x)) != self.n_agents:
            self.initial_x = np.random.randint(
                0, self.width, size=self.n_agents)
        self.H = np.random.randint(low=env_config.get(
            'zumin'), high=env_config.get('zumax'), size=self.n_agents)

        self.Xi = np.random.randint(
            low=0, high=self.width, size=self.I)  # X coordinate IOTs
        self.Yi = np.random.randint(
            low=1, high=self.heigh, size=self.I)  # Y coordinate IOTs
        self.Zi = np.zeros((self.I))

        # initialize AoU and binary association matrix
        self.a = np.zeros((self.I, self.n_agents, self.T))
        self.AoU = np.zeros((self.I, self.n_agents, self.T))
        self.All_AoU = {}
        self.Rate = np.zeros((self.I, self.n_agents, self.T))
        self.num_served_Iot = {}
        self.data_collected = {}

        # initialize visited_square for renderieng
        self.visited_square = {}

        # initialize visited_square for rendering
        self._agent_ids = set([i for i in range(self.n_agents)])

        # Defining observation space and action space

        self.observation_space = gym.spaces.Tuple(
            tuple([gym.spaces.Box(low=0, high=self.width,
                  shape=(2,), dtype=np.int32)] * self.n_agents)
        )
        self.action_space = gym.spaces.Tuple(
            tuple(self.n_agents * [Discrete(4)]))

    def reset(self):
        self.timestep = 1
        self.a = np.zeros((self.I, self.n_agents, self.T))
        self.AoU = np.zeros((self.I, self.n_agents, self.T))
        self.Rate = np.zeros((self.I, self.n_agents, self.T))
        self.collison = []
        self.obs = [0 for _ in self._agent_ids]
        self.visited_square = {}
        self.All_AoU = {}
        self.num_served_Iot = {}
        self.data_collected = {}
        for agent in self._agent_ids:
            self.obs[agent] = np.array([self.initial_x[agent], self.initial_y])
            self.visited_square[agent] = []
        self.obs = tuple(self.obs)
        return self.obs

    def step(
        self, action_list
    ):
        rewards = [0 for _ in self._agent_ids]
        dones = [False for _ in self._agent_ids]
        infos = {}
        self.obs = list(self.obs)

        for agent in self._agent_ids:
            if action_list[agent] == 0:
                if self.obs[agent][0]+1 <= self.width:
                    self.obs[agent][0] += 1
            elif action_list[agent] == 1:
                if self.obs[agent][0]-1 >= env_config.get('xumin'):
                    self.obs[agent][0] -= 1
            elif action_list[agent] == 2:
                if self.obs[agent][1]+1 <= self.heigh:
                    self.obs[agent][1] += 1
            else:
                if self.obs[agent][1]-1 >= env_config.get('yumin'):
                    self.obs[agent][1] -= 1

        for i in range(self.I):
            for agent in self._agent_ids:
                if computeRate(self.BIoT[i], self.noises, self.beta0, self.PIot[i, agent, self.timestep], self.fadingIot[i, agent, self.timestep], self.Xi[i],
                               self.obs[agent][0], self.Yi[i], self.obs[agent][1], 0, self.H[agent], alpha=2) >= self.Ri and [self.obs[agent][0], self.obs[agent][1]] == [self.Xi[i], self.Yi[i]]:
                    self.a[i, agent, self.timestep] = 1
                    self.AoU[i, agent, self.timestep] = 0
                else:
                    self.AoU[i, agent, self.timestep] = 1

        for agent in self._agent_ids:
            for agent1 in self._agent_ids:
                if agent != agent1:
                    if self.obs[agent][0] != self.obs[agent1][0] or self.obs[agent][1] != self.obs[agent1][1]:
                        rewards[agent] = objective2cont(
                            self.a, agent, self.timestep, self.obs[agent][
                                0], self.obs[agent][1], self.H[agent], self.x, self.y,
                            self.AoU, self.n_agents, self.I, self.lambda1, self.lambda2, self.lambda3, self.BIoT, self.noises, self.beta0, self.PIot,
                            self.fadingIot, self.Xi, self.Yi, self.H, self.T)
                    else:
                        rewards[agent] = - objective2cont(
                            self.a, agent, self.timestep, self.obs[agent][
                                0], self.obs[agent][1], self.H[agent], self.x, self.y,
                            self.AoU, self.n_agents, self.I, self.lambda1, self.lambda2, self.lambda3, self.BIoT, self.noises, self.beta0, self.PIot,
                            self.fadingIot, self.Xi, self.Yi, self.H, self.T)

                        self.collison.append([True, agent, agent1])

            self.visited_square[agent].append(
                [self.obs[agent][0], self.obs[agent][1]])

        for agent in self._agent_ids:
            if self.timestep >= self.T-1:
                dones[agent] = True
            else:
                dones[agent] = False

        self.obs = tuple(self.obs)
        self.timestep += 1
        return self.obs, rewards, dones, infos

    def render(self):
        fig, ax = plt.subplots(figsize=(15, 15))
        ax.scatter(self.Xi, self.Yi, marker='o', c='g', s=30)

        marker = ["v", "^", "<", ">"]
        color = ['b', 'r', 'c', 'y']
        for agent in self._agent_ids:
            # initial position
            ax.scatter(self.initial_x[agent], self.initial_y,
                       marker="X", c=color[agent], s=50)
            # Target position
            ax.scatter(self.x[agent], self.y, marker="*", c=color[agent], s=50)
            # Current position
            ax.scatter(self.obs[agent][0], self.obs[agent][1],
                       marker=marker[agent], c=color[agent], s=50)
            # Visited square
            for square in self.visited_square[agent]:
                ax.scatter(square[0], square[1],
                           marker=marker[agent], c=color[agent], s=50)

        # for collision in self.collison:
        #     if collision[0]==True:
        #         print('Collision between '+ str(self.collison[0]) + " and " +str(self.collison[1]))

        ax.set_yticks(np.arange(self.width+1))
        ax.set_yticks(np.arange(self.width+2)-0.5, minor=True)

        ax.set_xticks(np.arange(self.width+1))
        ax.set_xticks(np.arange(self.width+2)-0.5, minor=True)

        ax.grid(True, which="minor")
        ax.set_aspect("equal")
        plt.show()
        return

    def sample_tasks(self, num_tasks):
        Ts = np.random.randint(100, 300, size=(num_tasks,))
        Ris = np.random.randint(80000, 120000, size=(num_tasks,))

        tasks = [{'max_episode_steps': T, 'Ri': Ri}
                 for T, Ri in zip(Ts, Ris)]
        # tasks = [{'T': T, 'Ri': Ri, 'xumax': xumax, 'yumax': yumax, 'lambda1': lambda1, 'lambda2': lambda2, 'lambda3': lambda3}
        #          for T,Ri,xumax,yumax,lambda1,lambda2,lambda3 in zip(Ts,Ris,xumaxs,yumaxs,lambda1s,lambda2s,lambda3s)]
        return tasks

    def reset_task(self, task):
        self.task = task
        self.Ri = self.task['Ri']
        self.max_episode_steps = self.task['max_episode_steps']
        self.T = self.max_episode_steps
        # Puissance transmit of IoTs
        self.PIot = np.random.rand(
            self.I, self.n_agents, self.max_episode_steps)
        self.fadingIot = np.random.rand(
            self.I, self.n_agents, self.max_episode_steps)  # fading for IoTs devices
        self.a = np.zeros((self.I, self.n_agents, self.max_episode_steps))
        self.AoU = np.zeros((self.I, self.n_agents, self.max_episode_steps))
        self.Rate = np.zeros((self.I, self.n_agents, self.max_episode_steps))

    def set_task(self, task):
        self.task = task
        self.Ri = self.task['Ri']
        self.max_episode_steps = self.task['max_episode_steps']
        self.T = self.max_episode_steps
        # Puissance transmit of IoTs
        self.PIot = np.random.rand(
            self.I, self.n_agents, self.max_episode_steps)
        self.fadingIot = np.random.rand(
            self.I, self.n_agents, self.max_episode_steps)  # fading for IoTs devices
        self.a = np.zeros((self.I, self.n_agents, self.max_episode_steps))
        self.AoU = np.zeros((self.I, self.n_agents, self.max_episode_steps))
        self.Rate = np.zeros((self.I, self.n_agents, self.max_episode_steps))
