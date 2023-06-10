import nn
import numpy as np
import copy

class DeepQNetwork():
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.
    """
    def __init__(self, state_dim, action_dim):
        self.num_actions = action_dim
        self.state_size = state_dim

        # Remember to set self.learning_rate, self.numTrainingGames,
        # self.parameters, and self.batch_size!
        self.learning_rate = 0.05
        self.numTrainingGames = 6000
        self.batch_size = 64

        self.w_layer1 = nn.Parameter(self.state_size, 512)
        self.b_layer1 = nn.Parameter(1, 512)

        self.w_layer2 = nn.Parameter(512, 512)
        self.b_layer2 = nn.Parameter(1, 512)

        self.w_layer3 = nn.Parameter(512, 256)
        self.b_layer3 = nn.Parameter(1, 256)

        self.w_layer4 = nn.Parameter(256, self.num_actions)
        self.set_weights([self.w_layer1, self.b_layer1, self.w_layer2, self.b_layer2, self.w_layer3, self.b_layer3, self.w_layer4])

    def set_weights(self, layers):
        self.parameters = []
        for i in range(len(layers)):
            self.parameters.append(layers[i])

    def get_loss(self, states, Q_target):
        """
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            loss node between Q predictions and Q_target
        """
        Q_predict = self.run(states)
        loss = nn.SquareLoss(Q_predict, Q_target)
        return loss

    def run(self, states):
        """
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            result: (batch_size x num_actions) numpy array of Q-value
                scores, for each of the actions
        """
        x = states
        x = nn.Linear(x, self.w_layer1)
        x = nn.AddBias(x, self.b_layer1)
        x = nn.ReLU(x)

        x = nn.Linear(x, self.w_layer2)
        x = nn.AddBias(x, self.b_layer2)
        x = nn.ReLU(x)

        x = nn.Linear(x, self.w_layer3)
        x = nn.AddBias(x, self.b_layer3)
        x = nn.ReLU(x)

        x = nn.Linear(x, self.w_layer4)
        return x

    def gradient_update(self, states, Q_target):
        """
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            None
        """
        loss = self.get_loss(states, Q_target)
        gradients = nn.gradients(loss, self.parameters)
        for i in range(len(self.parameters)):
            self.parameters[i].update(gradients[i], -self.learning_rate)