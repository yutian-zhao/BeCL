from dm_env import StepType, specs
from dmc import ExtendedTimeStep
import gymnasium as gym
import numpy as np
import point_maze

def space_to_spec(space, name):
    if not isinstance(space, gym.spaces.Box):
        raise Exception(f"Spaces {type(space)} except for Box currently are not supported, please add on your own if needed.")
    return specs.BoundedArray(shape=space.shape, maximum=space.high, minimum=space.low, name=name, dtype=space.dtype)

class DMCWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation_spec(self):
        return space_to_spec(self.observation_space, "observation")
    
    def action_spec(self):
        return space_to_spec(self.action_space, "action")
    
    def reset(self, seed=None, options=None):
        if options:
            obs, _ = super().reset(options=options)
        else:
            obs, _ = super().reset()

        return ExtendedTimeStep(observation=obs,
                                step_type=StepType.FIRST,
                                action=np.zeros(self.action_space.shape, dtype=self.action_space.dtype),
                                reward=0.0,
                                discount=1.0)
    
    def step(self, action):
        obs, reward, terminated, _, _ = super().step(action)
        return ExtendedTimeStep(observation=obs,
                                step_type=StepType.LAST if terminated else StepType.MID,
                                action=action,
                                reward=reward,
                                discount=1.0)
    
    def __getattr__(self, name):
        if name != 'env':
            return getattr(self.unwrapped, name)

if __name__ == '__main__':
    env = DMCWrapper(gym.make('point_maze', n=4, maze_type='square_a'))

    # print(env.action_spec())
    # print(env.observation_space.dtype)

    timestep = env.reset()
    print(timestep)
    episode_over = False
    while not episode_over:
        action = env.action_space.sample()  # agent policy that uses the observation and info
        timestep = env.step(action)
        print(timestep)

        episode_over = timestep.last()