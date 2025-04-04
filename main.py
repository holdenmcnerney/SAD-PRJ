# AEM 525 Project Holden McNerney

from quaternionpointing_env import QuaternionPointingEnv
from quaternionpointing_ppo_training import QuaternionPointingTraining

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import multiprocessing
from tensordict import TensorDict

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

    q = np.squeeze(np.array(logs["q"][0].cpu()), axis=(1,))
    q_d = np.squeeze(np.array(logs["q_d"][0].cpu()), axis=(1,))

    plt.plot(q[:,0] - q_d[:,0], label='q0')
    plt.plot(q[:,1] - q_d[:,1], label='q1')
    plt.plot(q[:,2] - q_d[:,2], label='q2')
    plt.plot(q[:,3] - q_d[:,3], label='q3')
    plt.legend()
    plt.show()
    # plot(logs)

    return 0

if __name__=='__main__':
    main()