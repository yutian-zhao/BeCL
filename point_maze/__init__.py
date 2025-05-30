from gymnasium.envs.registration import register
from .env import Env
__all__ = ['Env']

register(
    id="point_maze",
    entry_point="point_maze:Env", # gym_examples.envs:
)