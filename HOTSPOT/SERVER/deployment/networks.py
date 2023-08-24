import sys
import time
from collections import deque
from typing import Dict, List, Tuple

import numpy as np

import torch

sys.path.append("..")
from sac import SAC_Agent
from models import LinearVAE, SimpleNetwork


class Translator:
    def __init__(self):
        pass

    def __call__(self, x: np.ndarray):
        pass


class VAENetwork:
    def __init__(
        self,
        model_params: Dict[str, float],
        saved_model: str = None,
        live_train: bool = True,
    ):

        self.model_params = model_params
        self.model = LinearVAE(**model_params)
        self.live_train = live_train

        self.training_data: List[torch.Tensor] = []
        self.batch = 0

        self.lr = 0.0001
        self.losses: deque = deque(maxlen=16)

        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = torch.nn.MSELoss(reduction="mean")

    def __call__(self, x: np.ndarray):
        X = torch.from_numpy(x).type(torch.float)
        # self.batch.append(X.detach())
        self.batch += 1
        self.training_data.append(X.detach())

        if self.batch >= 8 and self.live_train:
            self.train()

        with torch.no_grad():
            Z = self.model.encode(X).detach().numpy()

        return Z

    def train(self):
        self.model.train()

        batch_losses = []

        for batch in torch.split(torch.stack(self.training_data), 8):
            self.optimiser.zero_grad()

            X = torch.stack(self.training_data)
            Z = self.model.encode(X)
            Y = self.model.decode(Z)

            loss = self.criterion(X, Y)

            loss.backward()
            self.optimiser.step()
            batch_losses.append(loss.item())

        self.losses.append(np.mean(batch_losses))

        print('*' * 50)
        print("* VAE Losses")
        print(f"* {loss.item()}")
        print('*' * 50)

        self.batch = 0

        self.training_data = self.training_data[-256:]


class LinearNetwork:
    def __init__(
        self,
        model_params: Dict[str, float],
        saved_model: str = None,
    ):

        self.model_params = model_params
        self.model = SimpleNetwork(**model_params)
        self.model.eval()

        self.batch: List[torch.Tensor] = []

        self.lr = 0.001

        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = torch.nn.MSELoss(reduction="mean")

    def __call__(self, x: np.ndarray):
        X = torch.from_numpy(x).type(torch.float).unsqueeze(0)
        self.batch.append(X.detach())

        # if len(self.batch) >= 16:
        #    self.train()

        with torch.no_grad():
            Y = self.model(X).detach().numpy()

        return Y

    def train(self):
        self.model.train()
        self.optimiser.zero_grad()

        X = torch.stack(self.batch)

        R, _ = self.model(X)
        loss = self.criterion(X, R)

        loss.backward()
        self.optimiser.step()

        # print(" -- Training results:")
        # print(f" -- loss: {loss.item()}")

        self.batch = []


class Buffer:
    def __init__(self, state_dim, action_dim, max_size=256, device="cpu"):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.reward = np.zeros((max_size, 1))
        self.next_state = np.zeros((max_size, state_dim))
        self.dead = np.zeros((max_size, 1), dtype=np.uint8)
        self.device = device

    def add(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, dead: bool = False):
        self.state[self.ptr] = state
        self.action[self.ptr] = np.copy(action)
        self.reward[self.ptr] = np.copy(reward)
        self.next_state[self.ptr] = np.copy(next_state)
        self.dead[self.ptr] = dead
       
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> Tuple:
        ind = np.random.randint(0, self.size, size=batch_size)

        with torch.no_grad():
            return (
                torch.FloatTensor(self.state[ind]).to(self.device),
                torch.FloatTensor(self.action[ind]).to(self.device),
                torch.FloatTensor(self.reward[ind]).to(self.device),
                torch.FloatTensor(self.next_state[ind]).to(self.device),
                torch.FloatTensor(self.dead[ind]).to(self.device),
            )
            
    def to_dict(self) -> Dict[str, np.ndarray]:
        return {
            "state": np.copy(self.state),
            "action": np.copy(self.action),
            "reward": np.copy(self.reward),
            "next_state": np.copy(self.next_state),
            "dead": np.copy(self.dead),
            "size": self.size,
            "max_size": self.max_size,
        }
        
    def from_dict(self, data: Dict[str, np.ndarray]):
        self.state = data["state"]
        self.action = data["action"]
        self.reward = data["reward"]
        self.next_state = data["next_state"]
        self.dead = data["dead"]
        
        self.size = data["size"]
        self.max_size = data["max_size"]


class SACModel:
    def __init__(
        self,
        train_every: int = 8,
        batch_size: int = 16,
        device: str = "cpu",
        **kwargs,
    ):

        self.model = SAC_Agent(**kwargs, device=device)
        self.training_data = Buffer(
            kwargs.get("state_dim"), kwargs.get("action_dim"), device=device
        )
        self.train_every = train_every
        self.batch_size = batch_size

        self.n = 0

        self.losses: Dict[str, deque] = {
            "a_losses": deque(maxlen=16),
            "q_losses": deque(maxlen=16),
        }

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        dead: bool = False,
    ):

        self.training_data.add(state, action, reward, next_state, dead)
        self.n += 1

        if self.n >= self.train_every:
            a_loss, q_loss = self.model.train(self.training_data)
            self.losses["a_losses"].append(a_loss)
            self.losses["q_losses"].append(q_loss)
            self.n = 0

            print('*' * 50)
            print("* SAC Losses")
            print(f"* {a_loss}    {q_loss}")
            print('*' * 50)

    def get_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        state : np.ndarray
        deterministic : bool

        Returns
        -------
        action : np.ndarray
            Action parameter, each in interval (-1, 1)
        """

        action = self.model.select_action(state, deterministic)

        return action

    def save(self) -> Dict:
        state_dicts = {
            "actor": self.model.actor.state_dict(),
            "q_critic": self.model.q_critic.state_dict(),
        }
        return state_dicts

    def load(self, state_dicts: Dict):
        '''Loads latest weights and training data'''
        self.model.actor.load_state_dict(state_dicts["actor"])
        self.model.q_critic.load_state_dict(state_dicts["q_critic"])
