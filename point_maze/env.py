import point_maze.toy_maze as toy_maze
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


class Env(toy_maze.Env, gym.Env):
    def __init__(self, render_mode=None, **kwargs):
        self.render_mode = render_mode
        super().__init__(**kwargs)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=[2]
        )
        segments = np.array(list(self.maze._walls)).reshape(-1, 4)
        xs, ys = segments[:, :2], segments[:, 2:]
        self.observation_space = gym.spaces.Box(
            low=np.array([xs.min(), ys.min()]),
            high=np.array([xs.max(), ys.max()]),
        )

    def reset(self, seed=None, options=None):
        if options:
            super().reset(**options)
        else:
            super().reset()
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), self._get_info()

    def step(self, action):
        action *= self.action_range
        super().step(action)
        reward = self._get_reward()
        terminated = self.step_limit_reached()
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, terminated, False, self._get_info()

    def step_limit_reached(self):
        return self._state['n'] >= self.n

    def _get_obs(self):
        return self.state.numpy()  # agent

    def _get_reward(self):
        return 0  # see toy_maze.Env.reward for alternatives

    def _get_info(self):
        return {}

    def render(self):
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        self.maze.plot(ax)
        agent = self.state.numpy()
        ax.plot([agent[0]], [agent[1]], 'o')
        return fig, ax
