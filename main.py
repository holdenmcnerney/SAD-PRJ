# AEM 525 Project Holden McNerney

from quaternionpointing import QuaternionPointingEnv

import numpy as np
from typing import Optional
from collections import defaultdict

import torch
from torch import nn
import tqdm
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule

from torchrl.data import Bounded, Composite, Unbounded
from torchrl.envs import EnvBase

from torchrl.envs.utils import check_env_specs, step_mdp
from torch.linalg import norm

def simple_rollout(env, steps=100):
    # preallocate:
    data = TensorDict({}, [steps])
    # reset
    _data = env.reset()
    for i in range(steps):
        _data["action"] = env.action_spec.rand()
        _data = env.step(_data)
        data[i] = _data
        _data = step_mdp(_data, keep_other=True)
    return data

def plot(logs):
    import matplotlib
    from matplotlib import pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(logs["return"])
    plt.title("returns")
    plt.xlabel("iteration")
    plt.subplot(1, 2, 2)
    plt.plot(logs["last_reward"])
    plt.title("last reward")
    plt.xlabel("iteration")
    plt.show()

def training(env):
    torch.manual_seed(0)
    env.set_seed(0)

    net = nn.Sequential(
        nn.LazyLinear(64),
        nn.Tanh(),
        nn.LazyLinear(64),
        nn.Tanh(),
        nn.LazyLinear(64),
        nn.Tanh(),
        nn.LazyLinear(1),
    )
    policy = TensorDictModule(
        net,
        in_keys=["observation"],
        out_keys=["action"],
    )
    optim = torch.optim.Adam(policy.parameters(), lr=2e-3)
    batch_size = 32
    pbar = tqdm.tqdm(range(20_000 // batch_size))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 20_000)
    logs = defaultdict(list)

    for _ in pbar:
        init_td = env.reset(env.gen_params(batch_size=[batch_size]))
        rollout = env.rollout(100, policy, tensordict=init_td, auto_reset=False)
        traj_return = rollout["next", "reward"].mean()
        (-traj_return).backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optim.step()
        optim.zero_grad()
        pbar.set_description(
            f"reward: {traj_return: 4.4f}, "
            f"last reward: {rollout[..., -1]['next', 'reward'].mean(): 4.4f}, gradient norm: {gn: 4.4}"
        )
        logs["return"].append(traj_return.item())
        logs["last_reward"].append(rollout[..., -1]["next", "reward"].mean().item())
        scheduler.step()

    plot(logs)

def main():

    env = QuaternionPointingEnv()
    check_env_specs(env)

    # batch_size = 10  # number of environments to be executed in batch
    # td = env.reset(env.gen_params(batch_size=[batch_size]))
    # print("reset (batch size of 10)", td)

    return 0

if __name__=='__main__':
    main()