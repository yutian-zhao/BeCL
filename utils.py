import math
import random
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection 

NUM_TRAJECTORIES = 20
TRAJECTORY_KWARGS = dict(alpha=0.2, linewidth=2)

SAVEFIG_KWARGS = dict(bbox_inches='tight', transparent=True)

ENV_LIMS = dict(
    square_a=dict(xlim=(-0.55, 4.55), ylim=(-4.55, 0.55), x=(-0.5, 4.5), y=(-4.5, 0.5)),
    square_bottleneck=dict(xlim=(-0.55, 9.55), ylim=(-0.55, 9.55), x=(-0.5, 9.5), y=(-0.5, 9.5)),
    square_corridor=dict(xlim=(-5.55, 5.55), ylim=(-0.55, 0.55), x=(-5.5, 5.5), y=(-0.5, 0.5)),
    square_corridor2=dict(xlim=(-5.55, 5.55), ylim=(-0.55, 0.55), x=(-5.5, 5.5), y=(-0.5, 0.5)),
    square_tree=dict(xlim=(-6.55, 6.55), ylim=(-6.55, 0.55), x=(-6.5, 6.5), y=(-6.5, 0.5))
)

class eval_mode:
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False
    
# skill_kwargs = dict(figsize=(5,5), reset_dict=dict(state=torch.tensor([0., -0.5])))
# ax = plot_all_skills(None, cmap, agent=self.agent_model.agent, **skill_kwargs)  # NOTE: sample 20 trajs for each skill
# plt.savefig(os.path.join(self.exp_dir, f'epoch_{self.curr_epoch}.png'))

def choose_cmap(skill_n):
    if skill_n <= 10:
        cmap = plt.get_cmap('tab10')
    elif 10 < skill_n <= 20:
        cmap = plt.get_cmap('tab20')
    else:
        cmap = plt.get_cmap('plasma', skill_n)
    return cmap

def plot_all_skills(rollouts_dict, env, ax=None, save_path=None, figsize=(5, 5), **kwargs):
    cmap = choose_cmap(len(rollouts_dict.keys()))

    if ax is None:
        return_ax = True
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        return_ax = False

    env.maze.plot(ax)

    for skill_id, rollouts in rollouts_dict.items():
        lc = LineCollection(rollouts, label="Skill #{}".format(skill_id), color=cmap(skill_id), zorder=10, 
        **TRAJECTORY_KWARGS)
        ax.add_collection(lc)

    ax.plot(rollouts[0][0][0], rollouts[0][0][1], marker='o', markersize=8, color='black', zorder=11)

    config_subplot(ax, maze_type=env.maze_type, **kwargs)

    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
    elif return_ax:
        return ax
    
def config_subplot(ax, maze_type=None, title=None, extra_lim=0., fontsize=14):
    if maze_type is not None:
        env_config = ENV_LIMS[maze_type]
        ax.set_xlim(env_config["xlim"][0] - extra_lim, env_config["xlim"][1] + extra_lim)
        ax.set_ylim(env_config["ylim"][0] - extra_lim, env_config["ylim"][1] + extra_lim)

    if title is not None:
        ax.set_title(title, fontsize=fontsize)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    for p in ["left", "right", "top", "bottom"]:
        ax.spines[p].set_visible(False)

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def chain(*iterables):
    for it in iterables:
        yield from it


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def hard_update_params(net, target_net):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(param.data)


def to_torch(xs, device):
    return tuple(torch.as_tensor(x, device=device) for x in xs)


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def grad_norm(params, norm_type=2.0):
    params = [p for p in params if p.grad is not None]
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type) for p in params]),
        norm_type)
    return total_norm.item()


def param_norm(params, norm_type=2.0):
    total_norm = torch.norm(
        torch.stack([torch.norm(p.detach(), norm_type) for p in params]),
        norm_type)
    return total_norm.item()


class Until:
    def __init__(self, until, action_repeat=1):
        self._until = until
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._until is None:
            return True
        until = self._until // self._action_repeat
        return step < until


class Every:
    def __init__(self, every, action_repeat=1):
        self._every = every
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._every is None:
            return False
        every = self._every // self._action_repeat
        if step % every == 0:
            return True
        return False


class Timer:
    def __init__(self):
        self._start_time = time.time()
        self._last_time = time.time()

    def reset(self):
        elapsed_time = time.time() - self._last_time
        self._last_time = time.time()
        total_time = time.time() - self._start_time
        return elapsed_time, total_time

    def total_time(self):
        return time.time() - self._start_time


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        # NOTE: a custom differentiable clamp
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip) # NOTE: clipped normal distribution, don't want the noise added to the mean larger than 0.3 -> NO, it's adding noise to mean
        x = self.loc + eps
        return self._clamp(x) # NOTE: clip to valid range
    

class ScaledBeta(pyd.Beta):
    def __init__(self, c1, c0, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(c1, c0, validate_args=False)
        self.c1 = c1 
        self.c0 = c0 
        self.low = low
        self.high = high
        self.eps = eps
        self.scale = scale

    def _clamp(self, x):
        # NOTE: a custom differentiable clamp
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    @property
    def mean(self): # NOTE: actually return the mode, but named as mean for consistency
        return (self.c1 - 1) / (self.c0 + self.c1 - 2)

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.c1.dtype,
                               device=self.c1.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)

        x = (self.high-self.low) * self.mean + self.low + eps # NOTE: this transform should base on high and low
        return self._clamp(x)

    def log_prob(self, action):
        action_logit = action/2 + 0.5
        log_probs = super().log_prob(action_logit) - torch.log(torch.tensor(2.0, device=self.c1.device))
        return log_probs
    
    def entropy(self):
        return super().entropy() - torch.log(torch.tensor(2.0, device=self.c1.device))



class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r'step_linear\((.+),(.+),(.+),(.+),(.+)\)', schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        x = x.float()
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class RMS(object):
    """running mean and std """
    def __init__(self, device, epsilon=1e-4, shape=(1,)):
        self.M = torch.zeros(shape).to(device)
        self.S = torch.ones(shape).to(device)
        self.n = epsilon

    def __call__(self, x):
        bs = x.size(0)
        delta = torch.mean(x, dim=0) - self.M
        new_M = self.M + delta * bs / (self.n + bs)
        new_S = (self.S * self.n + torch.var(x, dim=0) * bs +
                 torch.square(delta) * self.n * bs /
                 (self.n + bs)) / (self.n + bs)

        self.M = new_M
        self.S = new_S
        self.n += bs

        return self.M, self.S


class PBE(object):
    """particle-based entropy based on knn normalized by running mean """
    def __init__(self, rms, knn_clip, knn_k, knn_avg, knn_rms, device):
        self.rms = rms
        self.knn_rms = knn_rms
        self.knn_k = knn_k
        self.knn_avg = knn_avg
        self.knn_clip = knn_clip
        self.device = device

    def __call__(self, rep):
        source = target = rep
        b1, b2 = source.size(0), target.size(0)
        # (b1, 1, c) - (1, b2, c) -> (b1, 1, c) - (1, b2, c) -> (b1, b2, c) -> (b1, b2)
        sim_matrix = torch.norm(source[:, None, :].view(b1, 1, -1) -
                                target[None, :, :].view(1, b2, -1),
                                dim=-1,
                                p=2)
        reward, _ = sim_matrix.topk(self.knn_k,
                                    dim=1,
                                    largest=False,
                                    sorted=True)  # (b1, k)
        if not self.knn_avg:  # only keep k-th nearest neighbor
            reward = reward[:, -1]
            reward = reward.reshape(-1, 1)  # (b1, 1)
            reward /= self.rms(reward)[0] if self.knn_rms else 1.0
            reward = torch.maximum(
                reward - self.knn_clip,
                torch.zeros_like(reward).to(self.device)
            ) if self.knn_clip >= 0.0 else reward  # (b1, 1)
        else:  # average over all k nearest neighbors
            reward = reward.reshape(-1, 1)  # (b1 * k, 1)
            reward /= self.rms(reward)[0] if self.knn_rms else 1.0
            reward = torch.maximum(
                reward - self.knn_clip,
                torch.zeros_like(reward).to(
                    self.device)) if self.knn_clip >= 0.0 else reward
            reward = reward.reshape((b1, self.knn_k))  # (b1, k)
            reward = reward.mean(dim=1, keepdim=True)  # (b1, 1)
        reward = torch.log(reward + 1.0)
        return reward
