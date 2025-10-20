from rl_brain import SarsaTable
from maze_env import Maze
import matplotlib.pyplot as plt


# Global variables in setting the total episodes
EPISODES = 50

def sarsa_learning():
    
    # initialization
    env = Maze()
    sarsa_model = SarsaTable(env.action_space)
    episode_steps_list = []
    episode_list = []

    # main loop running all episodes
    for episode in range(EPISODES):

        # initial observation on state
        state = env.reset()  # this will return the initial state of the agent
        step_counter = 0

        # RL choose action based on observation
        action = sarsa_model.choose_action(str(state))

        while True:
            
            # fresh environment
            env.render() 
            
            # sarsa take action and get next state
            state_next, reward, done = env.step(action)

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
                episode_steps_list.append(step_counter)
                episode_list.append(episode)
                break
        
    
    # print end of the game
    print("Game over")
    env.destroy()

    # Plot learning progress
    plt.plot(episode_list, episode_steps_list, marker='o', label="steps for each episode")
    plt.xlabel('Episode')
    plt.ylabel('Steps to Finish')
    plt.title('SARSA Training Progress')
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    sarsa_learning()
