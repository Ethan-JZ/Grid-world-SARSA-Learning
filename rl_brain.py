import numpy as np
import pandas as pd


class RLModel:

    def __init__(self, actions: list, learning_rate: float=0.01, gamma: float=0.9, epsilon: float=0.9):
        self.actions = actions              # a list of actions
        self.learning_rate = learning_rate  # learning rate
        self.gamma = gamma                  # discounted factor
        self.epsilon = epsilon              # epsilon greedy factor

        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)  # pandas data frame with columns index being actions' names

    def choose_action(self, state: str) -> str:
        """
        choose the action based on the current state
        state: string of a list showing the bounding box of the agent [x1, y1, x2, y2]

        return: action string
        """
        self.check_state_exist(state)  # check if the state is in our q table
        
        # action selection
        if np.random.uniform() < self.epsilon:

            # choose the best action
            state_action = self.q_table.loc[state, :]

            # if actions have the same action, action=state_actions.idmax() will only choose the 1st action, so the other action will never be chosen
            action = np.random.choice(state_action[state_action == np.max(state_action)].index) # some actions may have the same value, randomly choose on these actions
        else:
            # choose random action
            action = np.random.choice(self.actions)
        
        return action

    def learn(self, *args):
        pass

    def check_state_exist(self, state: str):

        # if the state is not in the Q table, add it to the table
        if state not in self.q_table.index:

            # append the new state to the q table
            new_row = pd.Series(
                [0] * len(self.actions),
                index=self.q_table.columns,
                name=state,
            )

            # concat the row to the q table
            self.q_table = pd.concat([self.q_table, new_row.to_frame().T])

class QLearningTable(RLModel):

    def __init__(self, actions: list, learning_rate: float=0.01, gamma: float=0.9, epsilon: float=0.9):
        super().__init__(actions, learning_rate, gamma, epsilon)
    
    def learn(self, state: str, action: str, reward: float, state_next: str):
        
        # check if the next state is in our q table
        self.check_state_exist(state_next)
        q_estimate = self.q_table.loc[state, action]

        if state_next != 'terminal':
            q_target = reward + self.gamma * self.q_table.loc[state_next, :].max()
        else:
            q_target = reward
        
        # update Q(s, a)
        self.q_table.loc[state, action] += self.learning_rate * (q_target - q_estimate)



class SarsaTable(RLModel):
    def __init__(self, actions: list, learning_rate: float=0.01, gamma: float=0.9, epsilon: float=0.9):
        super().__init__(actions, learning_rate, gamma, epsilon)

    def learn(self, state: str, action: str, reward: float, state_next: str, action_next: str):
        
        # check if the next state is in our q table
        self.check_state_exist(state_next)  
        q_estimate = self.q_table.loc[state, action]

        if state_next != 'terminal':
            q_target = reward + self.gamma * self.q_table.loc[state_next, action_next]
        else:
            q_target = reward # next state is terminal
        
        # update Q(s, a)
        self.q_table.loc[state, action] += self.learning_rate * (q_target - q_estimate)
