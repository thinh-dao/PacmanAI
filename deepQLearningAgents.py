import torch
import deepQNetwork
from qlearningAgents import PacmanQAgent
from ReplayBuffer import ReplayBuffer, CNNReplayBuffer, PER_ReplayBuffer
import layout
import copy
import os
import time
from torch.utils.tensorboard import SummaryWriter
from input_pipelines import Input
import numpy as np
from deepQNetwork import MLP, CNN

class PacmanMLPQAgent(PacmanQAgent):
    def __init__(self, layout_input="smallGrid", target_update_rate=300, doubleQ=True, train=False, **args):
        PacmanQAgent.__init__(self, **args)
        self.model = None
        self.target_model = None
        self.target_update_rate = target_update_rate
        self.update_amount = 0
        self.epsilon_explore = 1.0
        self.epsilon0 = 0.5
        self.epsilon = self.epsilon0
        self.discount = 0.9
        self.update_frequency = 3
        self.counts = None
        self.replay_memory = ReplayBuffer(5000000)
        self.min_transitions_before_training = 50000
        self.train = train
        self.layout_input = layout_input
        self.save_frequency = 2500

        if self.train == False:
            self.min_transitions_before_training = 0
            self.epsilon0 = 0
            self.alpha = 0
            
        self.td_error_clipping = 50
        torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if isinstance(layout_input, str):
            layout_instantiated = layout.getLayout(layout_input)
        else:
            layout_instantiated = layout_input

        self.input = Input(layout_instantiated, "MLP_input1")
        self.get_features = self.input.get_features
        self.state_dim = self.input.state_dim
        self.initialize_q_networks(self.state_dim)

        self.doubleQ = doubleQ
        if self.doubleQ:
            self.target_update_rate = -1
        self.writer = SummaryWriter("summary/")

    def initialize_q_networks(self, state_dim, action_dim=5):
        self.model = MLP(state_dim, action_dim)
        self.target_model = MLP(state_dim, action_dim)
        path = os.path.join("save_models/", "MLP_" + self.layout_input + ".pth")
        if os.path.exists(path):
            print("Load weights")
            self.model.model.load_state_dict(torch.load(path))
            self.target_model.load_state_dict(torch.load(path))

    def getQValue(self, state, action):
        """
          Should return Q(state,action) as predicted by self.model
        """
        feats = torch.from_numpy(self.get_features(state)).float().to(self.device)
        action_index = self.action_encoding(action)
        return self.model.forward(feats)[action_index]

    def shape_reward(self, reward):
        if reward > 100:
            reward = 10
        elif reward > 0 and reward < 10:
            reward = 2
        elif reward == -1:
            reward = -1
        elif reward < -100:
            reward = -10
        return reward

    def action_encoding(self, action):
        if action == 'Stop': return 0
        if action == 'West': return 1
        if action == 'North': return 2
        if action == 'East': return 3
        if action == 'South': return 4

    def compute_q_targets(self, minibatch, network = None, target_network=None, doubleQ=False):
        """Prepare minibatches
        Args:
            minibatch (List[Transition]): Minibatch of `Transition`
        Returns:
            float: Loss value
        """
        if network is None:
            network = self.model
        if target_network is None:
            target_network = self.target_model
        states = np.vstack([x.state for x in minibatch])
        actions = np.array([x.action for x in minibatch])
        rewards = torch.from_numpy(np.array([x.reward for x in minibatch])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([x.next_state for x in minibatch])).float().to(self.device)
        done = torch.from_numpy(np.array([x.done for x in minibatch])).float().to(self.device)

        state_indices = states.astype(int)
        state_indices = (state_indices[:, 0], state_indices[:, 1])
        exploration_bonus = torch.from_numpy(1 / (2 * np.sqrt((self.counts[state_indices] / 100)))).float().to(self.device)

        with torch.no_grad():
            states = torch.from_numpy(states).float().to(self.device)
            Q_predict = network.forward(states)
            Q_target = Q_predict.detach().clone()
            
            replace_indices = torch.arange(actions.shape[0])
            next_Q = target_network.forward(next_states)
            action_indices = torch.argmax(network.forward(next_states), dim=1)
            target = rewards + exploration_bonus + (1 - done) * self.discount * next_Q[replace_indices, action_indices]

            Q_target[replace_indices, actions] = target.float()
            if self.td_error_clipping is not None:
                error = Q_target - Q_predict
                error = torch.clamp(error, -self.td_error_clipping, self.td_error_clipping)
                return (Q_predict + error).to(self.device)
            else: 
                return Q_target.to(self.device)

    def update(self, state, action, nextState, reward):
        if self.train == True:
            action_index = self.action_encoding(action)
            done = nextState.isLose() or nextState.isWin()
            reward = self.shape_reward(reward)
            if nextState.isWin(): reward += 20

            if self.counts is None:
                x, y = np.array(state.getFood().data).shape
                self.counts = np.ones((x, y))

            state = self.get_features(state)
            nextState = self.get_features(nextState)
            self.counts[int(state[0])][int(state[1])] += 1

            transition = (state, action_index, reward, nextState, done)
            self.replay_memory.push(*transition)

            if len(self.replay_memory) < self.min_transitions_before_training:
                self.epsilon = self.epsilon_explore
            else:
                self.epsilon = self.epsilon0 * (0.99) ** (self.update_amount // 1000)
            
            if len(self.replay_memory) > self.min_transitions_before_training and self.update_amount % self.update_frequency == 0:
                minibatch = self.replay_memory.pop(self.model.batch_size)
                states = np.vstack([x.state for x in minibatch])

                Q_target1 = self.compute_q_targets(minibatch, self.model, self.target_model, doubleQ=self.doubleQ)

                if self.doubleQ:
                    Q_target2 = self.compute_q_targets(minibatch, self.target_model, self.model, doubleQ=self.doubleQ)

                states = torch.from_numpy(states).float().to(self.device)
            
                self.model.gradient_update(states, Q_target1)
                if self.doubleQ:
                    self.target_model.gradient_update(states, Q_target2)

            if self.target_update_rate > 0 and self.update_amount % self.target_update_rate == 0:
                self.target_model.set_weights(copy.deepcopy(self.model.parameters))
            self.update_amount += 1

    def final(self, state):
        """Called at the end of each game."""
        # call the super-class final method
        deltaReward = state.getScore() - self.lastState.getScore()
        self.observeTransition(self.lastState, self.lastAction, state, deltaReward)
        self.stopEpisode()

        # Make sure we have this var
        if not 'episodeStartTime' in self.__dict__:
            self.episodeStartTime = time.time()
        if not 'lastWindowAccumRewards' in self.__dict__:
            self.lastWindowAccumRewards = 0.0
        self.lastWindowAccumRewards += state.getScore()

        NUM_EPS_UPDATE = 100
        if self.episodesSoFar % NUM_EPS_UPDATE == 0:
            print('Reinforcement Learning Status:')
            windowAvg = self.lastWindowAccumRewards / float(NUM_EPS_UPDATE)
            if self.episodesSoFar <= self.numTraining:
                trainAvg = self.accumTrainRewards / float(self.episodesSoFar)
                print('\tCompleted %d out of %d training episodes' % (
                       self.episodesSoFar,self.numTraining))
                print('\tAverage Rewards over all training: %.2f' % (
                        trainAvg))
            else:
                testAvg = float(self.accumTestRewards) / (self.episodesSoFar - self.numTraining)
                print('\tCompleted %d test episodes' % (self.episodesSoFar - self.numTraining))
                print('\tAverage Rewards over testing: %.2f' % testAvg)
            print('\tAverage Rewards for last %d episodes: %.2f'  % (
                    NUM_EPS_UPDATE,windowAvg))
            train_time = (time.time() - self.episodeStartTime)
            print('\tEpisode took %.2f seconds' % train_time)
            self.writer.add_scalar("Average reward", windowAvg, self.episodesSoFar)
            self.writer.add_scalar("Training time", train_time, self.episodesSoFar)
            self.lastWindowAccumRewards = 0.0
            self.episodeStartTime = time.time()

        if self.episodesSoFar == self.numTraining:
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg,'-' * len(msg)))

        # did we finish training?
        if self.train == True and self.episodesSoFar % self.save_frequency == 0:
            if self.episodesSoFar != self.numTraining: 
                model_name = "MLP_" + self.layout_input + "_" + str(self.episodesSoFar) + ".pth"
            else:
                model_name = "MLP_" + self.layout_input + ".pth"
            model_name = "MLP_" + self.layout_input + "_" + str(self.episodesSoFar) + ".pth"
            path = "save_models/"
            save_path = os.path.join(path, model_name)
            torch.save(deepQNetwork, save_path)

class PacmanCNNQAgent(PacmanQAgent):
    def __init__(self, layout_input="smallGrid", target_update_rate=300, doubleQ=True, train=True, **args):
        PacmanQAgent.__init__(self, **args)
        self.model = None
        self.target_model = None
        self.target_update_rate = target_update_rate
        self.update_amount = 0
        self.epsilon_explore = 1.0
        self.epsilon_start = 0.7
        self.epsilon_end = 0.1
        print(self.numTraining)
        self.epsilon_decay_rate = (self.epsilon_start - self.epsilon_end) / self.numTraining
        
        self.epsilon = self.epsilon0
        self.discount = 0.9
        self.update_frequency = 3
        self.state_history = 3
        # self.counts = None
        self.replay_memory = CNNReplayBuffer(capacity=5000000, frame_len=self.state_history)
        self.min_transitions_before_training = 1000
        self.train = train
        self.layout_input = layout_input
        self.save_frequency = 2500
        
        if self.train == False:
            self.min_transitions_before_training = 0
            self.epsilon0 = 0
            self.alpha = 0
            
        self.td_error_clipping = 50
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if isinstance(layout_input, str):
            layout_instantiated = layout.getLayout(layout_input)
        else:
            layout_instantiated = layout_input

        self.input = Input(layout_instantiated, "CNN_input1")
        self.get_features = self.input.get_features
        self.state_dim = self.input.state_dim
        self.initialize_q_networks(self.state_dim)

        self.doubleQ = doubleQ
        if self.doubleQ:
            self.target_update_rate = -1
        self.writer = SummaryWriter("summary/")

    def initialize_q_networks(self, state_dim, action_dim=5):
        self.model = CNN(state_dim, action_dim, self.state_history)
        self.target_model = CNN(state_dim, action_dim, self.state_history)
        path = os.path.join("save_models/", "CNN_" + self.layout_input + ".pth")
        if os.path.exists(path):
            print("Load weights")
            self.model.model.load_state_dict(torch.load(path))
            self.target_model.load_state_dict(torch.load(path))

    def getQValue(self, state, action):
        """
          Should return Q(state,action) as predicted by self.model
        """
        state = self.get_features(state)
        q_input = torch.tensor(self.replay_memory.get_q_input(state), dtype=torch.float, device=self.device)
        action_index = self.action_encoding(action)
        return self.model.forward(q_input)[0, action_index]

    def shape_reward(self, reward):
        if reward > 100:
            reward = 10
        elif reward > 0 and reward < 10:
            reward = 2
        elif reward == -1:
            reward = -1
        elif reward < -100:
            reward = -10
        return reward

    def action_encoding(self, action):
        if action == 'Stop': return 0
        if action == 'West': return 1
        if action == 'North': return 2
        if action == 'East': return 3
        if action == 'South': return 4

    def compute_q_targets(self, minibatch, network = None, target_network=None, doubleQ=False):
        """Prepare minibatches
        Args:
            minibatch (List[Transition]): Minibatch of `Transition`
        Returns:
            float: Loss value
        """
        if network is None:
            network = self.model
        if target_network is None:
            target_network = self.target_model
        states,actions,rewards,next_states,done = minibatch

        # state_indices = states.astype(int)
        # state_indices = (state_indices[:, 0], state_indices[:, 1])
        # exploration_bonus = torch.from_numpy(1 / (2 * np.sqrt((self.counts[state_indices] / 100)))).float().to(self.device)
        with torch.no_grad():
            Q_predict = network.forward(states)
            Q_target = Q_predict.detach().clone()
            
            replace_indices = torch.arange(actions.shape[0])
            next_Q = target_network.forward(next_states)
            action_indices = torch.argmax(network.forward(next_states), dim=1)
            #target = rewards + exploration_bonus + (1 - done) * self.discount * next_Q[replace_indices, action_indices]
            target = rewards + (1 - done) * self.discount * next_Q[replace_indices, action_indices]

            Q_target[replace_indices, actions] = target.float()
            if self.td_error_clipping is not None:
                error = Q_target - Q_predict
                error = torch.clamp(error, -self.td_error_clipping, self.td_error_clipping)
                return (Q_predict + error).to(self.device)
            else: 
                return Q_target.to(self.device)

    def update(self, state, action, nextState, reward):
        if self.train == True:
            action_index = self.action_encoding(action)
            done = nextState.isLose() or nextState.isWin()
            reward = self.shape_reward(reward)
            if nextState.isWin(): reward += 20

            # if self.counts is None:
            #     x, y = np.array(state.getFood().data).shape
            #     self.counts = np.ones((x, y))

            state = self.get_features(state)
            nextState = self.get_features(nextState)
            # self.counts[int(state[0])][int(state[1])] += 1

            transition = (state, action_index, reward, nextState, done)
            self.replay_memory.push(*transition)

            if len(self.replay_memory) < self.min_transitions_before_training:
                self.epsilon = self.epsilon_explore
            else:
                self.epsilon = self.epsilon0 * (0.99) ** (self.update_amount // 1000)
            
            if len(self.replay_memory) > self.min_transitions_before_training and self.update_amount % self.update_frequency == 0:
                minibatch = self.replay_memory.pop(self.model.batch_size)
                
                states = torch.from_numpy(np.vstack([np.expand_dims(x.state, axis=0) for x in minibatch])).float().to(self.device)
                actions = torch.from_numpy(np.array([x.action for x in minibatch])).to(self.device)
                rewards = torch.from_numpy(np.array([x.reward for x in minibatch])).float().to(self.device)
                next_states = torch.from_numpy(np.vstack([np.expand_dims(x.next_state, axis=0) for x in minibatch])).float().to(self.device)
                done = torch.from_numpy(np.array([x.done for x in minibatch])).float().to(self.device)
                
                minibatch = (states, actions, rewards, next_states, done)

                Q_target1 = self.compute_q_targets(minibatch, self.model, self.target_model, doubleQ=self.doubleQ)

                if self.doubleQ:
                    Q_target2 = self.compute_q_targets(minibatch, self.target_model, self.model, doubleQ=self.doubleQ)

                self.model.gradient_update(states, Q_target1)
                if self.doubleQ:
                    self.target_model.gradient_update(states, Q_target2)

            if self.target_update_rate > 0 and self.update_amount % self.target_update_rate == 0:
                self.target_model.set_weights(copy.deepcopy(self.model.parameters))
            self.update_amount += 1

    def final(self, state):
        """Called at the end of each game."""
        # call the super-class final method
        deltaReward = state.getScore() - self.lastState.getScore()
        self.observeTransition(self.lastState, self.lastAction, state, deltaReward)
        self.stopEpisode()

        # Make sure we have this var
        if not 'episodeStartTime' in self.__dict__:
            self.episodeStartTime = time.time()
        if not 'lastWindowAccumRewards' in self.__dict__:
            self.lastWindowAccumRewards = 0.0
        self.lastWindowAccumRewards += state.getScore()

        NUM_EPS_UPDATE = 100
        if self.episodesSoFar % NUM_EPS_UPDATE == 0:
            print('Reinforcement Learning Status:')
            windowAvg = self.lastWindowAccumRewards / float(NUM_EPS_UPDATE)
            if self.episodesSoFar <= self.numTraining:
                trainAvg = self.accumTrainRewards / float(self.episodesSoFar)
                print('\tCompleted %d out of %d training episodes' % (
                       self.episodesSoFar,self.numTraining))
                print('\tAverage Rewards over all training: %.2f' % (
                        trainAvg))
            else:
                testAvg = float(self.accumTestRewards) / (self.episodesSoFar - self.numTraining)
                print('\tCompleted %d test episodes' % (self.episodesSoFar - self.numTraining))
                print('\tAverage Rewards over testing: %.2f' % testAvg)
            print('\tAverage Rewards for last %d episodes: %.2f'  % (
                    NUM_EPS_UPDATE,windowAvg))
            train_time = (time.time() - self.episodeStartTime)
            print('\tEpisode took %.2f seconds' % train_time)
            self.writer.add_scalar("Average reward", windowAvg, self.episodesSoFar)
            self.writer.add_scalar("Training time", train_time, self.episodesSoFar)
            self.lastWindowAccumRewards = 0.0
            self.episodeStartTime = time.time()

        if self.episodesSoFar == self.numTraining:
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg,'-' * len(msg)))

        # did we finish training?
        if self.train == True and self.episodesSoFar % self.save_frequency == 0:
            if self.episodesSoFar != self.numTraining: 
                model_name = "CNN_" + self.layout_input + "_" + str(self.episodesSoFar) + ".pth"
            else:
                model_name = "CNN_" + self.layout_input + ".pth"    
            path = "save_models/"
            save_path = os.path.join(path, model_name)
            torch.save(deepQNetwork, save_path)
            
            
            
            
    
