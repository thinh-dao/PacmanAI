"""
This file contains input pipelines for Neural Networks
"""
import numpy as np

class Input:
    def __init__(self, layout, input_type):
        self.layout = layout
        self.width = layout.width
        self.height = layout.height
        self.wall_pos = self.get_wall_pos()
        self.input_type = input_type

        if (self.network_type == "MLP_input1"):
            self.get_features = self.MLP_input1
            self.state_dim = 2 + 2 * layout.getNumGhosts() + layout.width * layout.height
        elif (self.network_type == "MLP_input2"):
            self.get_features = self.MLP_input2
            self.state_dim = self.height * self.width 
        elif (self.get_features == "CNN_input1"):
            self.get_features = self.CNN_input1
            self.state_dim = (self.height, self.width, 3)
        elif (self.get_features == "CNN_input2"):
            self.get_features = self.CNN_input2
            self.state_dim = (self.height, self.width, 6)

    def get_walls_position(self):
        res = []
        wall_state = self.layout.walls.data
        height = len(wall_state)
        width = len(wall_state[0])
        for i in range(height):
            for j in range(width):
                if wall_state[i][j] == True:
                    res.append((i, j))
        return res
    
    def MLP_input1(self, state):
        """
        Neurons represent the positions of pacman, ghosts, foods, capsules
        Input:
            state: GameState object
        Output:
            (state_dim,) numpy array
        """
        pacman_state = np.array(state.getPacmanPosition())
        ghost_state = np.array(state.getGhostPositions())
        capsules = state.getCapsules()
        food_locations = np.array(state.getFood().data).astype(np.float32)
        for x, y in capsules:
            food_locations[x][y] = 2
        return np.concatenate((pacman_state, ghost_state.flatten(), food_locations.flatten()))

    def MLP_input2(self, state, width, height):
        """
        Neurons represent the each grid in the layout
        Input:
            state: GameState object
        Output:
            (state_dim,) numpy array
        """
        features = np.zeros((width, height))
        pacman_state = np.array(state.getPacmanPosition()).astype(int)
        features[pacman_state[0]][pacman_state[1]] = 1
        ghost_state = np.array(state.getGhostPositions()).astype(int)
        capsules = state.getCapsules().astype(int)
        food_locations = np.array(state.getFood().data).astype(int)
        for (x, y) in ghost_state:
            features[x][y] = -1
        for (x,y) in food_locations:
            features[x][y] = 5
        for (x,y) in capsules:
            features[x][y] = 10
        return features.flatten().astype(np.float32)

    def CNN_input1(self, state):
        """
        Convert GameState object to RGB image
        Pacman -> Yellow
        Walls -> Blue
        Ghost -> Red
        Empty Grids -> Black
        Food -> White
        Capsules -> Green
        Input:
            state: GameState object
        Output:
            (layout_width, layout_height, 3) numpy array
        """
        image = np.zeros((self.width, self.height, 3))

        yellow = (255, 255, 0)
        red = (255, 0, 0)
        white = (255, 255, 255)
        green = (0, 128, 0)
        blue = (0, 0, 255)

        pacman_state = np.array(state.getPacmanPosition()).astype(int)
        for i in range(3):
            x, y = pacman_state
            image[x][y][i] = yellow[i]

        ghost_state = np.array(state.getGhostPositions()).astype(int)
        for x, y in ghost_state:
            for i in range(3):
                image[x][y][i] = red[i]

        capsules = state.getCapsules().astype(int)
        for x, y in capsules:
            for i in range(3):
                image[x][y][i] = green[i]

        food_locations = np.array(state.getFood().data).astype(int)
        for x, y in food_locations:
            for i in range(3):
                image[x][y][i] = white[i]

        for x, y in self.wall_pos:
            for i in range(3):
                image[x][y][i] = blue[i]

    def CNN_input2(self, state):
        """ Return wall, ghosts, food, capsules matrices """ 
        def getWallMatrix(state):
            """ Return matrix with wall coordinates set to 1 """
            matrix = np.zeros((self.height, self.width), dtype=np.int32)
            grid = state.data.layout.walls

            for i in range(grid.height):
                for j in range(grid.width):
                    # Put cell vertically reversed in matrix
                    cell = 1 if grid[j][i] else 0
                    matrix[-1-i][j] = cell
            return matrix

        def getPacmanMatrix(state):
            """ Return matrix with pacman coordinates set to 1 """
            matrix = np.zeros((self.height, self.width), dtype=np.int32)

            for agentState in state.data.agentStates:
                if agentState.isPacman:
                    pos = agentState.configuration.getPosition()
                    cell = 1
                    matrix[-1-int(pos[1])][int(pos[0])] = cell

            return matrix

        def getGhostMatrix(state):
            """ Return matrix with ghost coordinates set to 1 """
            matrix = np.zeros((self.height, self.width), dtype=np.int32)

            for agentState in state.data.agentStates:
                if not agentState.isPacman:
                    if not agentState.scaredTimer > 0:
                        pos = agentState.configuration.getPosition()
                        cell = 1
                        matrix[-1-int(pos[1])][int(pos[0])] = cell

            return matrix

        def getScaredGhostMatrix(state):
            """ Return matrix with ghost coordinates set to 1 """
            matrix = np.zeros((self.height, self.width), dtype=np.int32)

            for agentState in state.data.agentStates:
                if not agentState.isPacman:
                    if agentState.scaredTimer > 0:
                        pos = agentState.configuration.getPosition()
                        cell = 1
                        matrix[-1-int(pos[1])][int(pos[0])] = cell

            return matrix

        def getFoodMatrix(state):
            """ Return matrix with food coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            grid = state.data.food
            matrix = np.zeros((height, width), dtype=np.int8)

            for i in range(grid.height):
                for j in range(grid.width):
                    # Put cell vertically reversed in matrix
                    cell = 1 if grid[j][i] else 0
                    matrix[-1-i][j] = cell

            return matrix
        
        def getCapsulesMatrix(state):
            """ Return matrix with capsule coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            capsules = state.data.layout.capsules
            matrix = np.zeros((height, width), dtype=np.int8)

            for i in capsules:
                # Insert capsule cells vertically reversed into matrix
                matrix[-1-i[1], i[0]] = 1

            return matrix

        # Create observation matrix as a combination of
        # wall, pacman, ghost, food and capsule matrices
        # width, height = state.data.layout.width, state.data.layout.height 
        width, height = self.params['width'], self.params['height']
        observation = np.zeros((6, height, width))

        observation[0] = getWallMatrix(state)
        observation[1] = getPacmanMatrix(state)
        observation[2] = getGhostMatrix(state)
        observation[3] = getScaredGhostMatrix(state)
        observation[4] = getFoodMatrix(state)
        observation[5] = getCapsulesMatrix(state)

        observation = np.swapaxes(observation, 0, 2)

        return observation
        

        
        
        

