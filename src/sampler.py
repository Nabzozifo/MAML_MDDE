import numpy as np


class Sampler():
    def __init__(self, num_tasks):
        self.num_tasks = num_tasks

    def sample_tasks(self):
        Ts = np.random.randint(100, 300, size=(self.num_tasks,))
        Ris = np.random.randint(80000, 120000, size=(self.num_tasks,))

        tasks = [{'max_episode_steps': T, 'Ri': Ri}
                 for T, Ri in zip(Ts, Ris)]
        return tasks


# # import numpy as np


# class Sampler():
#     def __init__(self, num_tasks=10):
#         self.num_tasks = num_tasks

#     def sample_tasks(self):
#         # np.random.randint(50, 150, size=(self.num_tasks,))
#         Ts = [i for i in range(100, 350, 5)]
#         # np.random.randint(80000, 120000, size=(self.num_tasks,))
#         Ris = [i for i in range(75500, 150000, 1500)]

#         tasks = [{'max_episode_steps': T, 'Ri': Ri}
#                  for T, Ri in zip(Ts, Ris)]
#         return tasks
