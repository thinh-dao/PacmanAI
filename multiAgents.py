# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

from qlearningAgents import ApproximateQAgent

def get_StateScore(state: GameState):
    if state.isWin():
        return 2000
    if state.isLose():
        return -2000
    
    GhostStates = state.getGhostStates()
    PacmanPos = state.getPacmanPosition()
    numFood = state.getNumFood()

    closestGhost = 1000

    for ghost in GhostStates:
        dist = manhattanDistance(ghost.getPosition(), PacmanPos)
        closestGhost = min(closestGhost, dist)

    if closestGhost <= 1:
        return -2000
    
    closestFood = 1000
    for food in state.getFood().asList():
        closestFood = min(closestFood, state.data.layout.trueDist[PacmanPos[0]][PacmanPos[1]][food[0]][food[1]])

    return -closestFood + state.getScore() - 10 * numFood

def evaluationFunction(currentGameState: GameState, action):

    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    closestGhost = 1000

    for state in newGhostStates:
        dist = manhattanDistance(state.getPosition(), newPos)
        closestGhost = min(dist, closestGhost)

    if closestGhost <= 1:
        return -1000
    
    numFood = successorGameState.getNumFood()
    prevNumFood = currentGameState.getNumFood()
    minimumDistanceFood = -100
    distancesFoods = []
    for food in newFood.asList():
        distancesFoods.append(successorGameState.data.layout.trueDist[newPos[0]][newPos[1]][food[0]][food[1]])
    if len(distancesFoods) > 0:
        minimumDistanceFood = min(distancesFoods) 
    return -minimumDistanceFood + successorGameState.getScore() - 100*(numFood - prevNumFood)

class ReflexAgent(Agent):
    def getAction(self, gameState: GameState):

        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]
    

class MultiAgentSearchAgent(Agent):
    def __init__(self, depth = '1'):
        self.index = 0 # Pacman is always agent index 0
        self.depth = int(depth)

    def isTerminalState(self, gameState):
        return gameState.isWin() or gameState.isLose()
    
    def isPacman(self, agent):
        return agent == 0


class MinimaxAgent(MultiAgentSearchAgent):

    def getAction(self, gameState: GameState):

        actions = gameState.getLegalActions(self.index)
        highest = float('-inf')
        best_action = None
        max_depth = self.depth * gameState.getNumAgents()

        for action in actions:
            state = gameState.generateSuccessor(self.index, action)
            # score = get_StateScore(state)
            score = self.value(state, 1, max_depth, self.index)
            if score > highest:
                highest, best_action = score, action
        
        return best_action
    
    def value(self, gameState, depth, max_depth, agentIndex):

        if depth == max_depth or gameState.isWin() or gameState.isLose():
            return get_StateScore(gameState)

        nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
        if nextAgentIndex == 0:
            return self.maxValue(gameState, depth+1, max_depth, nextAgentIndex)
        else:
            return self.minValue(gameState, depth+1, max_depth, nextAgentIndex)
        
    def maxValue(self, gameSate, depth, max_depth, agentIndex):
        v = float('-inf')
        actions = gameSate.getLegalActions(agentIndex)
        for action in actions:
            state = gameSate.generateSuccessor(agentIndex, action)
            v = max(v, self.value(state, depth, max_depth, agentIndex))
        return v

    def minValue(self, gameSate, depth, max_depth, agentIndex):
        v = float('inf')
        actions = gameSate.getLegalActions(agentIndex)
        for action in actions:
            state = gameSate.generateSuccessor(agentIndex, action)
            v = min(v, self.value(state, depth, max_depth, agentIndex))
        return v

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def maxValue(self, gameState, agent, depth, alpha, beta):
        bestValue = float("-inf")
        for action in gameState.getLegalActions(agent):
            successor = gameState.generateSuccessor(agent, action)
            v = self.minimax(successor, agent + 1, depth, alpha, beta)
            bestValue = max(bestValue, v)
            if depth == 1 and bestValue == v: 
                self.action = action
            if bestValue > beta: 
                return bestValue
            alpha = max(alpha, bestValue)
        return bestValue

    def minValue(self, gameState, agent, depth, alpha, beta):
        bestValue = float("inf")
        for action in gameState.getLegalActions(agent):
            successor = gameState.generateSuccessor(agent, action)
            v = self.minimax(successor, agent + 1, depth, alpha, beta)
            bestValue = min(bestValue, v)
            if bestValue < alpha: 
                return bestValue
            beta = min(beta, bestValue)
        return bestValue

    def minimax(self, gameState, agent=0, depth=0, alpha=float("-inf"), beta=float("inf")):
        agent = agent % gameState.getNumAgents()

        if self.isTerminalState(gameState):
            return get_StateScore(gameState)

        if self.isPacman(agent):
            if depth < self.depth:
                return self.maxValue(gameState, agent, depth+1, alpha, beta)
            else:
                return get_StateScore(gameState)
        else:
            return self.minValue(gameState, agent, depth, alpha, beta)

    def getAction(self, gameState):
        self.minimax(gameState)
        return self.action

class learningMinimaxAgent(ApproximateQAgent):
    def __init__(self, index = 0, depth = 1, extractor='SimpleExtractor', **args):
        self.index = index 
        self.depth = depth
        ApproximateQAgent.__init__(self, extractor, **args)

    def getAction(self, gameState: GameState):
        if self.episodesSoFar < self.numTraining:
            print('haha', self.episodesSoFar)
            print(self.weights)
            action = ApproximateQAgent.getAction(self, gameState)
            self.doAction(gameState, action)
            return action
        else:
            print('huhu', self.episodesSoFar)
            actions = gameState.getLegalActions(self.index)
            highest = float('-inf')
            best_action = None
            max_depth = self.depth * gameState.getNumAgents()

            for action in actions:
                state = gameState.generateSuccessor(self.index, action)
                score = self.getValue(state)
                score = self.getQValue(gameState, action)
                print('''---------------------------------------------
                ------------------------------------------------------
                ------------------------------------------------------
                ------------------------------------------------------
                ------------------------------------------------------''')
                print("Testinggggg")
                print(state)
                print('hehe--------------------hehe')
                print(self.getValue(state), self.getQValue(gameState, action))
                # score = self.value(state, 1, max_depth, self.index)
                if score > highest:
                    highest, best_action = score, action
            
            self.doAction(gameState, best_action)
            return best_action
        
    def value(self, gameState: GameState, depth, max_depth, agentIndex):

        if depth == max_depth or gameState.isWin() or gameState.isLose():
            return self.getValue(gameState)

        nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
        if nextAgentIndex == 0:
            return self.maxValue(gameState, depth+1, max_depth, nextAgentIndex)
        else:
            return self.minValue(gameState, depth+1, max_depth, nextAgentIndex)
            
    def maxValue(self, gameSate: GameState, depth, max_depth, agentIndex):
        v = float('-inf')
        actions = gameSate.getLegalActions(agentIndex)
        for action in actions:
            state = gameSate.generateSuccessor(agentIndex, action)
            v = max(v, self.value(state, depth, max_depth, agentIndex))
        return v

    def minValue(self, gameSate: GameState, depth, max_depth, agentIndex):
        v = float('inf')
        actions = gameSate.getLegalActions(agentIndex)
        for action in actions:
            state = gameSate.generateSuccessor(agentIndex, action)
            v = min(v, self.value(state, depth, max_depth, agentIndex))
        return v
    

class ExpectimaxAgent(MultiAgentSearchAgent):

    def maxValue(self, gameState, agentIndex, depth):
        agentIndex = agentIndex % gameState.getNumAgents()
        bestValue = float("-inf")
        
        for action in gameState.getLegalActions(agentIndex):
            
            successor = gameState.generateSuccessor(agentIndex, action)
            v = self.expectimax(successor, agentIndex + 1, depth)
            bestValue = max(bestValue, v)
            
            if depth == 1 and bestValue == v: 
                self.action = action

        return bestValue

    def probability(self, legalActions):
        return 1.0 / len(legalActions)

    def expValue(self, gameState, agentIndex, depth):
        agentIndex = agentIndex % gameState.getNumAgents()
        legalActions = gameState.getLegalActions(agentIndex)
        v = 0
        for action in legalActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            p = self.probability(legalActions)
            v += p * self.expectimax(successor, agentIndex + 1, depth)
        
        return v

    def expectimax(self, gameState, agentIndex, depth):

        agentIndex = agentIndex % gameState.getNumAgents()

        if self.isTerminalState(gameState):
            return get_StateScore(gameState)

        if self.isPacman(agentIndex):
            if depth < self.depth:
                return self.maxValue(gameState, agentIndex, depth + 1)
            else:
                return get_StateScore(gameState)
        else:
            return self.expValue(gameState, agentIndex, depth)


    def getAction(self, gameState):
        self.expectimax(gameState, 0, 0)
        return self.action