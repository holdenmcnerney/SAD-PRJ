# AEM 525 Project Holden McNerney

import torch
import numpy as np
from typing import Optional
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Bounded, Composite, Unbounded
from torchrl.envs import EnvBase
from torch.linalg import norm

THETA_LIM = 10*np.pi/180

def _step(tensordict):
    ''''
    Step Function
    '''
    q, q_dot, omega = tensordict["q"], tensordict["q_dot"], tensordict["omega"]
    q_d = tensordict["q_d"]
    a_1, a_2 = tensordict["params", "a_1"], tensordict["params", "a_2"]
    Ixx = tensordict["params", "Ixx"]
    Iyy = tensordict["params", "Iyy"]
    Izz = tensordict["params", "Izz"]
    dt = tensordict["params", "dt"]
    M = tensordict["action"].squeeze(-1)
    M = M.clamp(-tensordict["params", "max_moment"], 
                 tensordict["params", "max_moment"])
    costs = torch.square(norm(q - q_d)) + a_1*torch.square(norm(q_dot)) \
                                        + a_2*torch.square(norm(M))
    
    I = torch.tensor(([Ixx, 0, 0], [0, Iyy, 0], [0, 0, Izz]), 
                     dtype=torch.float32)
    old_omega_1, old_omega_2, old_omega_3 = omega[0], \
                                            omega[1], \
                                            omega[2] 
    omega_skew = torch.tensor(([0, -old_omega_3, old_omega_2], 
                               [old_omega_3, 0, -old_omega_1], 
                               [-old_omega_2, old_omega_1, 0]), 
                               dtype=torch.float32)

    omega_dot_inner = torch.transpose(M, 0, 1) \
                      - torch.mm(torch.mm(omega_skew, I), omega)
    omega_dot = torch.mm(torch.transpose(I, 0, 1), omega_dot_inner)
    new_omega = omega + omega_dot*dt
    new_omega_1, new_omega_2, new_omega_3 = new_omega[0], \
                                            new_omega[1], \
                                            new_omega[2] 
    Omega_mat = torch.tensor(([0, new_omega_3, -new_omega_2, new_omega_1], 
                              [-new_omega_3, 0, new_omega_1, new_omega_2], 
                              [new_omega_2, -new_omega_1, 0, new_omega_3], 
                              [-new_omega_1, -new_omega_2, -new_omega_1, 0]), 
                              dtype=torch.float32)
    new_q_dot = 1/2*torch.mm(Omega_mat, q)
    new_q = q + new_q_dot*dt
    new_q = torch.nn.functional.normalize(new_q, dim=0)

    reward = -costs.view(*tensordict.shape, 1)
    done = torch.zeros_like(reward, dtype=torch.bool)
    out = TensorDict(
        {
            "q": new_q,
            "q_dot": new_q_dot,
            "omega": new_omega,
            "q_d": q_d,
            "params": tensordict["params"],
            "reward": reward,
            "done": done,
        },
        tensordict.shape,
    )
    return out

def _reset(self, tensordict):
    if tensordict is None or tensordict.is_empty():
        tensordict = self.gen_params(batch_size=self.batch_size)

    # Randomly start the simulation values. Starting with small angles
    q = torch.tensor(([0], [0], [0], [1]), dtype=torch.float32)
    q_dot = torch.tensor(([0], [0], [0], [0]), dtype=torch.float32)
    omega = torch.tensor(([0], [0], [0]), dtype=torch.float32)

    rot_axis = torch.rand(3, generator=self.rng, device=self.device)
    rot_axis = torch.nn.functional.normalize(rot_axis, dim=0)
    theta = torch.rand(1, generator=self.rng, device=self.device)
    # THETA_LIM is set to 10 degrees to start
    theta = theta * THETA_LIM
    q_d_1_3 = torch.tensor(([rot_axis[0]*torch.sin(theta/2)], 
                            [rot_axis[1]*torch.sin(theta/2)], 
                            [rot_axis[2]*torch.sin(theta/2)]), 
                            dtype=torch.float32)
    q_d_4 = torch.tensor(([torch.cos(theta/2)]), dtype=torch.float32).view(1, 1)
    q_d = torch.cat((q_d_1_3, q_d_4), 0) 

    out = TensorDict(
        {
            "q": q,
            "q_dot": q_dot,
            "omega": omega,
            "q_d": q_d,
            "params": tensordict["params"],
        },
        batch_size=tensordict.shape,
    )
    return out

def _make_spec(self, td_params):
    self.observation_spec = Composite(
        q=Bounded(
            low=-td_params["params", "max_q"],
            high=td_params["params", "max_q"],
            shape=(4,1),
            dtype=torch.float32,
        ),
        q_dot=Bounded(
            low=-td_params["params", "max_q_dot"],
            high=td_params["params", "max_q_dot"],
            shape=(4,1),
            dtype=torch.float32,
        ),
        omega=Bounded(
            low=-td_params["params", "max_omega"],
            high=td_params["params", "max_omega"],
            shape=(3,1),
            dtype=torch.float32,
        ),
        q_d=Bounded(
            low=-td_params["params", "max_q"],
            high=td_params["params", "max_q"],
            shape=(4,1),
            dtype=torch.float32,
        ),
        params = make_composite_from_td(td_params["params"]),
        shape=(),
    )
    self.state_spec = self.observation_spec.clone()
    self.action_spec = Bounded(
        low=-td_params["params", "max_moment"],
        high=td_params["params", "max_moment"],
        shape=(1,3),
        dtype=torch.float32,
    )
    self.reward_spec = Unbounded(shape=(*td_params.shape, 1))

def make_composite_from_td(td):
    composite = Composite(
        {
            key: make_composite_from_td(tensor)
            if isinstance(tensor, TensorDictBase)
            else Unbounded(
                dtype=tensor.dtype, device=tensor.device, shape=tensor.shape
            )
            for key, tensor in td.items()
        },
        shape=td.shape,
    )
    return composite

def _set_seed(self, seed: Optional[int]):
    rng = torch.manual_seed(seed)
    self.rng = rng

def gen_params(batch_size=None) -> TensorDictBase:
    if batch_size is None:
        batch_size = []
    td = TensorDict(
        {
            "params": TensorDict(
                {
                    "max_q": 1,
                    "max_q_dot": 1,
                    "max_omega": 1,
                    "max_moment": 1,
                    "a_1": 0.01,
                    "a_2": 0.001,
                    "Ixx": 1,
                    "Iyy": 1,
                    "Izz": 1,
                    "dt": 0.01,
                },
                [],
            )
        },
        [],
    )
    if batch_size:
        td = td.expand(batch_size).contiguous()
    return td

class QuaternionPointingEnv(EnvBase):
    metadata = {
        "description": "Quaternion Pointing Env for Spacecraft Deep RL"
    }
    _batch_locked = False

    def __init__(self, td_params=None, seed=None, device="cpu"):
        if td_params is None:
            td_params = self.gen_params()
        
        super().__init__(device=device, batch_size=[])
        self._make_spec(td_params)
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)
    
    gen_params = staticmethod(gen_params)
    _make_spec = _make_spec

    _reset = _reset
    _step = staticmethod(_step)
    _set_seed = _set_seed