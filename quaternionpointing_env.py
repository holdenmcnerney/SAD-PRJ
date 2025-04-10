# AEM 525 Project Holden McNerney

import torch
import numpy as np
from typing import Optional
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Bounded, Composite, Unbounded
from torchrl.envs import EnvBase
from torch.linalg import norm

from torch import multiprocessing

THETA_LIM = 20*np.pi/180

def _step(tensordict):
    ''''
    Step Function
    '''
    q, q_dot, omega = tensordict["q"], tensordict["q_dot"], tensordict["omega"]
    q_d = tensordict["q_d"]
    a_1, a_2, a_3 = tensordict["params", "a_1"], tensordict["params", "a_2"], \
                    tensordict["params", "a_2"]
    Ixx = tensordict["params", "Ixx"]
    Iyy = tensordict["params", "Iyy"]
    Izz = tensordict["params", "Izz"]
    dt = tensordict["params", "dt"]
    M = tensordict["action"]
    M = M.clamp(-tensordict["params", "max_moment"], 
                 tensordict["params", "max_moment"])
    q = torch.transpose(q, 0, 1)
    omega = torch.transpose(omega, 0, 1)
    M = torch.transpose(M, 0, 1)
    q_d = torch.transpose(q_d, 0, 1)
    costs = 10*norm(q - q_d)**2 + a_1*norm(q_dot)**2 \
                                + a_2*norm(M)**2 \
                                + a_3*norm(omega)**2
    
    I = torch.tensor(([Ixx, 0, 0], [0, Iyy, 0], [0, 0, Izz]), 
                     dtype=torch.float32, device='cuda:0')
    old_omega_1, old_omega_2, old_omega_3 = omega[0], \
                                            omega[1], \
                                            omega[2] 
    omega_skew = torch.tensor(([0, -old_omega_3, old_omega_2], 
                               [old_omega_3, 0, -old_omega_1], 
                               [-old_omega_2, old_omega_1, 0]), 
                               dtype=torch.float32, device='cuda:0')

    omega_dot_inner = M - torch.mm(torch.mm(omega_skew, I), omega)
    omega_dot = torch.mm(torch.transpose(I, 0, 1), omega_dot_inner)
    new_omega = omega + omega_dot*dt
    new_omega_1, new_omega_2, new_omega_3 = new_omega[0], \
                                            new_omega[1], \
                                            new_omega[2] 
    Omega_mat = torch.tensor(([0, new_omega_3, -new_omega_2, new_omega_1], 
                              [-new_omega_3, 0, new_omega_1, new_omega_2], 
                              [new_omega_2, -new_omega_1, 0, new_omega_3], 
                              [-new_omega_1, -new_omega_2, -new_omega_1, 0]), 
                              dtype=torch.float32, device='cuda:0')
    new_q_dot = 1/2*torch.mm(Omega_mat, q)
    new_q = q + new_q_dot*dt
    new_q = torch.nn.functional.normalize(new_q, dim=0)

    if 2*np.arccos(sum(q.cpu()*q_d.cpu())) < np.pi/180: 
        reward = -costs.view(*tensordict.shape, 1) + 50
    elif 2*np.arccos(sum(q.cpu()*q_d.cpu())) < 5*np.pi/180: 
        reward = -costs.view(*tensordict.shape, 1) + 5
    else:
        reward = -costs.view(*tensordict.shape, 1) - 5
    # done = torch.zeros_like(reward, dtype=torch.bool)
    done = torch.tensor([True], device='cuda:0') if norm(omega) > 2 else torch.tensor([False], device='cuda:0') 

    new_q = torch.transpose(new_q, 0, 1)
    new_q_dot = torch.transpose(new_q_dot, 0, 1)
    new_omega = torch.transpose(new_omega, 0, 1)
    q_d = torch.transpose(q_d, 0, 1)
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
    if tensordict is None or tensordict.is_empty() or bool(tensordict["_reset"]) is True:
        tensordict = self.gen_params(batch_size=self.batch_size)

    # Randomly start the simulation values. Starting with small angles
    q = torch.tensor(([[0, 0, 0, 1]]), dtype=torch.float32, device=self.device)
    q_dot = torch.tensor(([[0, 0, 0, 0]]), dtype=torch.float32, device=self.device)
    omega = torch.tensor(([[0, 0, 0]]), dtype=torch.float32, device=self.device)

    rot_axis = torch.rand(3, generator=torch.Generator(device='cuda'), device=self.device)
    rot_axis = torch.nn.functional.normalize(rot_axis, dim=0)
    theta = torch.rand(1, generator=torch.Generator(device='cuda'), device=self.device)
    # THETA_LIM is set to 10 degrees to start
    theta = theta * THETA_LIM
    q_d = torch.tensor(([[rot_axis[0]*torch.sin(theta/2), 
                          rot_axis[1]*torch.sin(theta/2), 
                          rot_axis[2]*torch.sin(theta/2),
                          torch.cos(theta/2)]]), dtype=torch.float32, device=self.device)

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
            shape=torch.Size([1, 4]),
            dtype=torch.float32,
            device=self.device,
        ),
        q_dot=Bounded(
            low=-td_params["params", "max_q_dot"],
            high=td_params["params", "max_q_dot"],
            shape=torch.Size([1, 4]),
            dtype=torch.float32,
            device=self.device,
        ),
        omega=Bounded(
            low=-td_params["params", "max_omega"],
            high=td_params["params", "max_omega"],
            shape=torch.Size([1, 3]),
            dtype=torch.float32,
            device=self.device,
        ),
        q_d=Bounded(
            low=-td_params["params", "max_q"],
            high=td_params["params", "max_q"],
            shape=torch.Size([1, 4]),
            dtype=torch.float32,
            device=self.device,
        ),
        params = make_composite_from_td(td_params["params"]),
        shape=(),
    )
    self.state_spec = self.observation_spec.clone()
    self.action_spec = Bounded(
        low=-td_params["params", "max_moment"],
        high=td_params["params", "max_moment"],
        shape=torch.Size([1, 3]),
        dtype=torch.float32,
        device=self.device,
    )
    self.reward_spec = Unbounded(shape=(*td_params.shape, 1), device=self.device)

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
    is_fork = multiprocessing.get_start_method() == "fork"
    device = (
        torch.device(0)
        if torch.cuda.is_available() and not is_fork
        else torch.device("cpu")
    )
    if batch_size is None:
        batch_size = []
    td = TensorDict(
        {
            "params": TensorDict(
                {
                    "max_q": 1,
                    "max_q_dot": 2,
                    "max_omega": 2,
                    "max_moment": 0.5,
                    "a_1": 1,
                    "a_2": 1,
                    "a_3": 1,
                    "Ixx": 30,
                    "Iyy": 30,
                    "Izz": 10,
                    "dt": 0.01,
                },
                [],
            ),
        },
        device=device,
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