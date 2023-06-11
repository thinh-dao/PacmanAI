import torch
from torch import nn
import numpy as np

class DeepQNetwork():
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.
    """
    def __init__(self, model):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("Running on ", self.device)
        self.model = model.to(self.device)
        self.learning_rate = 0.2
        self.numTrainingGames = 2000
        self.batch_size = 64
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

class MLP(DeepQNetwork):
    def __init__(self, state_dim, action_dim):
        self.model = nn.Sequential(
            nn.Linear(in_features=state_dim, out_features=512, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256, bias=True),
            nn.ReLU(),  
            nn.Linear(in_features=256, out_features=action_dim, bias=True)
        )
        super().__init__(self.model)

class CNN(DeepQNetwork):
    def __init__(self, state_dim, action_dim, state_history):
        img_height, img_width, n_channels = state_dim
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=n_channels * state_history, out_channels=32, kernel_size=8, stride=4, padding=((4 - 1) * img_height - 4 + 8) // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=64, kernel_size=4, stride=2, padding=((2 - 1) * img_height - 2 + 4) // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=64, kernel_size=3, stride=1, padding=((1 - 1) * img_height - 1 + 3) // 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=64 * img_height * img_width, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=action_dim)
        )
        super().__init__(self.model)