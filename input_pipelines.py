"""
This file contains input pipelines for Neural Networks
"""
import numpy as np

class Input:
    def __init__(self, layout, input_type):
        self.layout = layout
        self.width = layout.width
        self.height = layout.height
        self.wall_pos = self.get_walls_position()
        self.input_type = input_type

        if (self.input_type == "MLP_input1"):
            self.get_features = self.MLP_input1
            self.state_dim = 2 + 2 * layout.getNumGhosts() + layout.width * layout.height
        elif (self.input_type == "MLP_input2"):
            self.get_features = self.MLP_input2
            self.state_dim = self.height * self.width 
        elif (self.input_type == "CNN_input1"):
            self.get_features = self.CNN_input1
            self.state_dim = (self.height, self.width, 3)
        elif (self.input_type == "CNN_input2"):
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

    def MLP_input2(self, state):
        """
        Neurons represent the each grid in the layout
        Input:
            state: GameState object
        Output:
            (state_dim,) numpy array
        """
        features = np.array(state.getFood().data).astype(int) * 5
        pacman_state = np.array(state.getPacmanPosition()).astype(int)
        features[-1-int(pacman_state[0])][pacman_state[1]] = 1
        ghost_state = np.array(state.getGhostPositions()).astype(int)
        capsules = np.array(state.getCapsules()).astype(int)
        for (x,y) in ghost_state:
            features[-1-int(x)][y] = -1
        for (x,y) in capsules:
            features[-1-int(x)][y] = 10
        return features.flatten().astype(np.float32)

    def CNN_input1(self, state):
        """
        Convert GameState object to RGB image
        Pacman -> Yellow
        Walls -> Blue
        Ghost -> Red(not scared) or Grey(scared)
        Empty Grids -> Black
        Food -> White
        Capsules -> Green
        Input:
            state: GameState object
        Output:
            (layout_width, layout_height, 3) numpy array
        """
        food_loc = np.array(state.getFood().data) * 255.

        image = np.stack([food_loc, food_loc, food_loc], axis=0).transpose(1,2,0)
        yellow = (255., 255., 0.)
        red = (255., 0., 0.)
        grey = (128., 128., 128.)
        green = (0., 128., 0.)
        blue = (0., 0., 255.)

        for agentState in state.data.agentStates:
            if not agentState.isPacman:
                if agentState.scaredTimer > 0:
                    x, y = agentState.configuration.getPosition()
                    image[-1-int(x)][int(y)] = grey
                else:
                    x, y = agentState.configuration.getPosition()
                    image[-1-int(x)][int(y)] = red
            else:
                x, y = agentState.configuration.getPosition()
                image[-1-int(x)][int(y)] = yellow

        capsules = state.getCapsules()
        image[-1-int(x)][int(y)] = green
                
        for x, y in self.wall_pos:
            image[-1-int(x)][int(y)] = blue
                
        return image.astype(float) * 1/255

    def CNN_input2(self, state):
        """ Return wall, ghosts, food, capsules matrices """ 
        def getWallMatrix(state):
            """ Return matrix with wall coordinates set to 1 """
            matrix = np.zeros((self.height, self.width), dtype=np.int32)
            grid = state.data.layout.walls

            for i in range(grid.height):
                for j in range(grid.width):
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
        width, height = self.width, self.height
        observation = np.zeros((6, height, width))

        observation[0] = getWallMatrix(state)
        observation[1] = getPacmanMatrix(state)
        observation[2] = getGhostMatrix(state)
        observation[3] = getScaredGhostMatrix(state)
        observation[4] = getFoodMatrix(state)
        observation[5] = getCapsulesMatrix(state)

        observation = np.swapaxes(observation, 0, 2)

        return observation.astype(float) * 1/255
        

        
        
        

