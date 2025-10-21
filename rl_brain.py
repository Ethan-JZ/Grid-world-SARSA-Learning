import numpy as np
import pandas as pd


class RLModel:

    def __init__(self, actions: list, learning_rate: float=0.01, gamma: float=0.9, epsilon: float=0.9):
        self.actions = actions              # a list of actions
        self.learning_rate = learning_rate  # learning rate
        self.gamma = gamma                  # discounted factor
        self.epsilon = epsilon              # epsilon greedy factor

        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)  # pandas data frame with columns index being actions' names
        self.eligibility_trace = None  # only used by sarsa lambda table

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

            # add to eligibility trace if needed
            if self.eligibility_trace is not None:
                self.eligibility_trace = pd.concat([self.eligibility_trace, new_row.to_frame().T])
    

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
    """
    One step sarsa, update for the final step reaching the target, Sarsa(0)
    """
    def __init__(self, actions: list, learning_rate: float=0.01, gamma: float=0.9, epsilon: float=0.9):
        super().__init__(actions, learning_rate, gamma, epsilon)

    def learn(self, state: str, action: str, reward: float, state_next: str, action_next: str):
        """
        Sarsa to update Q(s, a) based on s, a, r, s', a'
        state: current state s
        action: current action a
        reward: immediate reward r
        state_next: next state s'
        action_next: next action a'

        update Q(s, a) = Q(s, a) + alpha * (r + gamma * Q(s', a') - Q(s, a))
        """
        
        # check if the next state is in our q table
        self.check_state_exist(state_next)  
        q_estimate = self.q_table.loc[state, action]

        if state_next != 'terminal':
            q_target = reward + self.gamma * self.q_table.loc[state_next, action_next]
        else:
            q_target = reward # next state is terminal
        
        # update Q(s, a)
        self.q_table.loc[state, action] += self.learning_rate * (q_target - q_estimate)


class SarsaLambdaTable(RLModel):
    """
    One step sarsa, update for the final step reaching the target, Sarsa(0)
    """
    def __init__(self, actions: list, learning_rate: float=0.01, gamma: float=0.9, epsilon: float=0.9, trace_decay: float=0.9):
        super().__init__(actions, learning_rate, gamma, epsilon)

        self.lambda_ = trace_decay
        self.eligibility_trace = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def learn(self, state: str, action: str, reward: float, state_next: str, action_next: str):
        """
        Sarsa to update Q(s, a) based on s, a, r, s', a'
        state: current state s
        action: current action a
        reward: immediate reward r
        state_next: next state s'
        action_next: next action a'

        update Q(s, a) = Q(s, a) + alpha * (r + gamma * Q(s', a') - Q(s, a))
        """
        
        # check if the next state is in our q table
        self.check_state_exist(state_next)  
        q_estimate = self.q_table.loc[state, action]

        if state_next != 'terminal':
            q_target = reward + self.gamma * self.q_table.loc[state_next, action_next]
        else:
            q_target = reward # next state is terminal
        
        # check if the state is necessary, if it is necessay, we replace it with 1
        TD_error = q_target - q_estimate
        self.eligibility_trace.loc[state, :] *= 0
        self.eligibility_trace.loc[state, action] = 1
        
        # update Q(s, a)
        self.q_table += self.learning_rate * TD_error * self.eligibility_trace

        # decay eligibility trace after update
        self.eligibility_trace *= self.gamma * self.lambda_
