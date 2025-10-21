"""
Maze environment for reinforcement learning.

Red rectangle:      agent
Black rectangles:   obstacles  [reward = -1]
Yellow rectangle:   target     [reward = +1]
All other states:   ground     [reward = 0]

in tkinter, the axis of x and y is like:

(0, 0) -------------------> x
  |               |
  |               |
  |               |
  |---------------o (x, y)              
  |
  |
  v y

"""

import numpy as np
import time
import tkinter as tk


UNIT = 40  # pixels
MAZE_H = 4  # grid height
MAZE_W = 4  # grid width

class Maze(tk.Tk):

    def __init__(self):
        super().__init__()
        self.action_space = ['up', 'down', 'left', 'right']
        self.title('Maze')
        self.geometry(f"{MAZE_W * UNIT}x{MAZE_H * UNIT}")
        self._build_maze()

    def _build_rectangle(self, center: np.ndarray, color: str) -> int:
        """
        Build the rectangle based on the center of it
        center: center coordinates of the rectangle
        color: color string
        
        return: an integer ID of the rectangle
        """

        # top left corner
        x1, y1 = center[0] - 10, center[1] - 10

        # bottom right corner
        x2, y2 = center[0] + 10, center[1] + 10

        return self.canvas.create_rectangle(x1, y1, x2, y2, fill=color)


    def _build_maze(self) -> None:
        """
        Build the maze
        """

        self.canvas = tk.Canvas(self, bg='white', height=MAZE_H*UNIT, width=MAZE_W*UNIT)

        # create grids
        for col in range(0, MAZE_W * UNIT, UNIT):
            self.canvas.create_line(col, 0, col, MAZE_H * UNIT)
        for row in range(0, MAZE_H * UNIT, UNIT):
            self.canvas.create_line(0, row, MAZE_W * UNIT, row)
        
        # create origin
        origin = np.array([20, 20])

        # obstacle 1
        self.obstacle1 = self._build_rectangle(origin + np.array([UNIT * 2, UNIT]), 'black')

        # obstacle 2
        self.obstacle2 = self._build_rectangle(origin + np.array([UNIT, UNIT * 2]), 'black')

        # agent
        self.agent = self._build_rectangle(origin, 'red')

        # create target
        self.target = self._build_rectangle(origin + UNIT * 2, 'blue')

        # pack all
        self.canvas.pack()
    
    def reset(self) -> list[float]:

        self.update()
        time.sleep(0.1)
        self.canvas.delete(self.agent)
        origin = np.array([20, 20])
        self.agent = self._build_rectangle(origin, 'red')

        # return observation
        return self.canvas.coords(self.agent)
    
    def step(self, action: str):
        """
        move the agent by one step with action
        action: action string

        return: next state, reward, done flag
        """
        state = self.canvas.coords(self.agent)

        # Move the agent
        base_action = np.array([0, 0])

        if action == "up":
            if state[1] > UNIT:        # the state is above x axis
                base_action[1] -= UNIT # move y up
        elif action == "down":
            if state[1] < UNIT * (MAZE_H - 1):
                base_action[1] += UNIT # move y down
        elif action == "right":
            if state[0] < UNIT * (MAZE_W - 1):
                base_action[0] += UNIT # move x right
        elif action == "left":
            if state[0] > UNIT:
                base_action[0] -= UNIT # move x left

        self.canvas.move(self.agent, base_action[0], base_action[1])  # move the agent
        
        # Get the next state
        state_next = self.canvas.coords(self.agent)

        # Reward function
        if state_next == self.canvas.coords(self.target): # hit the target
            reward = 1
            done = True
            state_next = 'terminal'
        elif state_next in [self.canvas.coords(self.obstacle1), self.canvas.coords(self.obstacle2)]: # hit the obstacle
            reward = -1
            done = True
            state_next = 'terminal'
        else:
            reward = 0
            done = False
        
        return state_next, reward, done
    
    def render(self):
        time.sleep(0.01)
        self.update()


if __name__ == "__main__":
    env = Maze()
    print(env.reset())


