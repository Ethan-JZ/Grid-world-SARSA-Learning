from rl_brain import SarsaTable, SarsaLambdaTable
from maze_env import Maze
import matplotlib.pyplot as plt


# Global variables in setting the total episodes
EPISODES = 50

def sarsa_learning():
    
    ############### Sarsa(0) learning with lambda = 0 ###############

    # initialization
    env_sarsa = Maze()
    sarsa_model = SarsaTable(env_sarsa.action_space)
    episode_steps_list_sarsa = []
    episode_list_sarsa = []

    # main loop running all episodes
    for episode in range(EPISODES):

        # initial observation on state
        state = env_sarsa.reset()  # this will return the initial state of the agent
        step_counter = 0

        # RL choose action based on observation
        action = sarsa_model.choose_action(str(state))

        while True:
            
            # fresh environment
            env_sarsa.render() 
            
            # sarsa take action and get next state
            state_next, reward, done = env_sarsa.step(action)

            # sarsa choose action based on next state
            action_next = sarsa_model.choose_action(str(state_next))

            # sarsa learn based on state, action, reward, next state, next action
            sarsa_model.learn(str(state), action, reward, str(state_next), action_next)

            # move to next state and next action
            state = state_next
            action = action_next
            step_counter += 1

            # break while if it is the end of the episode
            if done:
                print(f"Episode {episode+1}/{EPISODES} finished in {step_counter} steps.")
                episode_steps_list_sarsa.append(step_counter)
                episode_list_sarsa.append(episode)
                break

    # print end of the game
    print("Game over")
    env_sarsa.destroy()

    # Plot learning progress
    fig, ax = plt.subplots(figsize=(8, 7)) 
    ax.plot(episode_list_sarsa, episode_steps_list_sarsa, color='red', marker='o', label="steps for each episode")
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps to Finish')
    ax.set_title('Sarsa(λ) Training Progress')
    ax.grid(True)
    ax.legend()

    plt.show()


def sarsa_lambda_learning():
    
    ############### Sarsa(lambda) learning with lambda = 0.9 ###############

    # initialization
    env_sarsa_lambda = Maze()
    sarsa_lambda = SarsaLambdaTable(env_sarsa_lambda.action_space)
    episode_steps_list_sarsa_lambda = []
    episode_list_sarsa_lambda = []

    # main loop running all episodes
    for episode in range(EPISODES):

        # initial observation on state
        state = env_sarsa_lambda.reset()  # this will return the initial state of the agent
        step_counter = 0

        # RL choose action based on observation
        action = sarsa_lambda.choose_action(str(state))

        # initialize the eligibility trace
        sarsa_lambda.eligibility_trace *= 0

        while True:

            # fresh environment
            env_sarsa_lambda.render()

            # sarsa lamdba take action and get next state
            state_next, reward, done = env_sarsa_lambda.step(action)

            # sarsa lambda choose action based on next state
            action_next = sarsa_lambda.choose_action(str(state_next))

            # sarsa lambda learn from this transition
            sarsa_lambda.learn(str(state), action, reward, str(state_next), action_next)

            # move to next state and next action
            state = state_next
            action = action_next
            step_counter += 1

            # break while if it is the end of the episode
            if done:
                print(f"Episode {episode+1}/{EPISODES} finished in {step_counter} steps.")
                episode_steps_list_sarsa_lambda.append(step_counter)
                episode_list_sarsa_lambda.append(episode)
                break
        
        
    
    # print end of the game
    print("Game over")
    env_sarsa_lambda.destroy()

    # Plot learning progress
    fig, ax = plt.subplots(figsize=(8, 7)) 
    ax.plot(episode_list_sarsa_lambda, episode_steps_list_sarsa_lambda, color='red', marker='o', label="steps for each episode")
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps to Finish')
    ax.set_title('Sarsa(λ) Training Progress')
    ax.grid(True)
    ax.legend()

    plt.show()


if __name__ == "__main__":
    
    # run sarsa learning
    # sarsa_learning()

    # run sarsa(lambda) learning
    sarsa_lambda_learning()
