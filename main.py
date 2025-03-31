# AEM 525 Project Holden McNerney

from quaternionpointing_env import QuaternionPointingEnv
from quaternionpointing_ppo_training import QuaternionPointingTraining

from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import multiprocessing
import tqdm
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule

from torchrl.envs import (
    CatTensors,
    TransformedEnv,
    Compose,
    StepCounter,
)

from torchrl.envs.utils import check_env_specs, step_mdp

def simple_rollout(env, steps=100):
    # preallocate:
    data = TensorDict({}, [steps])
    # reset
    _data = env.reset()
    for i in range(steps):
        _data["action"] = env.action_spec.rand()
        # print(_data["action"])
        _data = env.step(_data)
        data[i] = _data
        _data = step_mdp(_data, keep_other=True)
    return data

def plot(logs):
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.plot(logs["reward"])
    plt.title("training rewards (average)")
    plt.subplot(2, 2, 2)
    plt.plot(logs["step_count"])
    plt.title("Max step count (training)")
    plt.subplot(2, 2, 3)
    plt.plot(logs["eval reward (sum)"])
    plt.title("Return (test)")
    plt.subplot(2, 2, 4)
    plt.plot(logs["eval step_count"])
    plt.title("Max step count (test)")
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
        nn.LazyLinear(3),
    )
    policy = TensorDictModule(
        net,
        in_keys=["observation"],
        out_keys=["action"],
    )
    optim = torch.optim.Adam(policy.parameters(), lr=2e-3)
    # batch_size = 32
    pbar = tqdm.tqdm(range(200))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 200)
    logs = defaultdict(list)

    for _ in pbar:
        init_td = env.reset(env.gen_params())
        rollout = env.rollout(100, policy, tensordict=init_td, auto_reset=False)
        traj_return = rollout["next", "reward"].mean()
        (-traj_return).backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optim.step()
        optim.zero_grad()
        pbar.set_description(
            f"reward: {traj_return: 4.4f}, "
            f"last reward: {rollout[..., -1]['next', 'reward'].mean(): 4.4f}, \
              gradient norm: {gn: 4.4}"
        )
        logs["return"].append(traj_return.item())
        logs["last_reward"].append(rollout[..., -1]["next", "reward"].mean().item())
        scheduler.step()

    plot(logs)
    return rollout

def main():

    is_fork = multiprocessing.get_start_method() == "fork"
    device = (
        torch.device(0)
        if torch.cuda.is_available() and not is_fork
        else torch.device("cpu")
    )
    env = QuaternionPointingEnv(device=device)
    env = TransformedEnv(
        env,
        Compose(
            StepCounter(),
            CatTensors(
                in_keys=["q", "q_dot", "omega", "q_d"], dim=1, 
                out_key="observation", del_keys=False, sort=False
            ),
        ),
    )
    check_env_specs(env)

    logs = QuaternionPointingTraining(env)

    plot(logs)

    return 0

if __name__=='__main__':
    main()