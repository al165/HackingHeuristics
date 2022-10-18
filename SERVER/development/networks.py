import os
import sys
from typing import Dict, List, Tuple

import numpy as np

import torch

sys.path.append("..")
from models import LinearVAE, SimpleNetwork


class Translator:
    def __init__(self):
        pass

    def __call__(self, x: np.ndarray):
        pass


class VAENetwork:
    def __init__(self, model_params: Dict[str, float], saved_model: str = None):

        self.model_params = model_params
        self.model = LinearVAE(**model_params)
        self.model.eval()

        self.batch = []

        self.lr = 0.0001

        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = torch.nn.MSELoss(reduction='mean')

    def __call__(self, x: np.ndarray):
        X = torch.from_numpy(x).type(torch.float)
        self.batch.append(X.detach())

        if len(self.batch) >= 16:
            self.train()

        with torch.no_grad():
            Y = self.model.encode(X).detach().numpy()

        return Y

    def train(self):
        self.model.train()
        self.optimiser.zero_grad()

        X = torch.stack(self.batch)

        R, _ = self.model(X)
        loss = self.criterion(X, R)

        loss.backward()
        self.optimiser.step()

        print(" -- Training results:")
        print(f" -- loss: {loss.item()}")

        self.batch = []


class LinearNetwork:

    def __init__(self, model_params: Dict[str, float], saved_model: str = None):

        self.model_params = model_params
        self.model = SimpleNetwork(**model_params)
        self.model.eval()

        self.batch = []

        self.lr = 0.001

        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = torch.nn.MSELoss(reduction='mean')

    def __call__(self, x: np.ndarray):
        X = torch.from_numpy(x).type(torch.float).unsqueeze(0)
        self.batch.append(X.detach())

        #if len(self.batch) >= 16:
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

        print(" -- Training results:")
        print(f" -- loss: {loss.item()}")

        self.batch = []
