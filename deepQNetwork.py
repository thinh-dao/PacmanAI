import torch
from torch import nn
import numpy as np

class DeepQNetwork():
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.
    """
    def __init__(self, state_dim, action_dim):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("Running on ", self.device)
        self.num_actions = action_dim
        self.state_size = state_dim

        self.learning_rate = 0.2
        self.numTrainingGames = 2000
        self.batch_size = 64
        self.cnt = 0

        self.model = nn.Sequential(
            nn.Linear(in_features=self.state_size, out_features=512, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256, bias=True),
            nn.ReLU(),  
            nn.Linear(in_features=256, out_features=self.num_actions, bias=True)
        ).to(self.device)

        self.loss_fn = nn.MSELoss().to(self.device)
        self.optimizer = torch.optim.Adagrad(params=self.model.parameters(), lr=0.05)

    def forward(self, x):
        return self.model(x)

    def run(self, states):
        """
        Inputs:
            states: a (batch_size x state_dim) tensor array
            Q_target: a (batch_size x num_actions) tensor array, or None
        Output:
            result: (batch_size x num_actions) tensor array of Q-value
                scores, for each of the actions
        """
        return self.forward(states)

    def gradient_update(self, states, Q_target):
        """
        Inputs:
            states: a (batch_size x state_dim) tensor array
            Q_target: a (batch_size x num_actions) tensor array, or None
        Output:
            None
        """
        self.optimizer.zero_grad()
        loss = self.loss_fn(self.model(states), Q_target).to(self.device)
        loss.backward()
        self.optimizer.step()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 100)
        self.cnt += 1