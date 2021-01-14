import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

class BanditTestRunner():
    """
    This class object is the tool that runs our experiments and then visualizes
    the results. We feed it ONE bandit environment, and one or more agents
    that will attempt to solve that environment.

    Note that in Figure 2.2 in the text, they use 1000 timesteps and average
    over 2000 runs.

    Attributes:
        agents: One or more agents from the BanditAgent class set.
        num_agents: The number of agents.
        bandit_env: The BanditEnvironment object - the specific bandit problem
            for this set of tests.
    """

    def __init__(self, agents, bandit_env):
        self.agents = agents
        self.num_agents = len(agents)
        self.bandit_env = bandit_env

    def full_reset(self):
        """
        Resets all true action values in the bandit environment, and resets
        all action-value estimates and counts in the agents.
        """
        self.bandit_env.reset_true_action_values()
        for agent in self.agents:
            agent.reset_action_value_estimates()
        
    def perform_runs(self, timesteps, runs):
        """
        Performs experiments in which the agent(s) attempt to learn the 
        bandit.

        Args:
            timesteps: The number of timesteps each agent has to learn the
                        bandit within each run.
            runs: The number of runs to conduct and average over.
        """
        # One history per agent
        rewards_history = np.zeros( (timesteps, len(self.agents)))
        optimal_action_history = np.zeros_like(rewards_history)

        for run in tqdm( range(0, runs) ):
            self.full_reset()
            for timestep in range(0,timesteps):
                for agent_id, agent in enumerate(self.agents):
                    action = agent.choose_action()
                    reward, optimal = self.bandit_env.\
                        take_action_emit_reward(action)
                    agent.update_action_value_estimates(action, reward)
                    rewards_history[timestep, agent_id] += reward
                    if optimal:
                        optimal_action_history[timestep,agent_id] += 1
        
        return rewards_history / runs, optimal_action_history / runs

    def visualize_results(self, save_filename, title, rewards_histories,
        optimal_action_histories): 
        # To label our plots, we reference our list of agents.
        # Our history objects will also be in this order.
        # So we can call the policy __str__ function to label our plot.
        labels = []
        for agent in self.agents:
            labels.append(agent.policy.__str__())

        plt.subplot(2,1,1)
        plt.plot(optimal_action_histories*100)
        print(optimal_action_histories)
        print(optimal_action_histories*100)
        plt.ylim(0,100)
        plt.xlabel("Steps")
        plt.ylabel("% Optimal Action Chosen")
        plt.xticks( np.arange(0, len(rewards_histories), len(rewards_histories)/10) )
        plt.tick_params(axis='x')
        plt.tick_params(axis='y')
        plt.legend(labels)
        
        plt.subplot(2,1,2)
        plt.plot(rewards_histories)
        plt.ylabel("Average Reward")
        plt.xticks( np.arange(0, len(rewards_histories),len(rewards_histories)/10) )
        plt.tick_params(axis='x')
        plt.tick_params(axis='y')
        plt.legend(labels)
        plt.show

        if save_filename is not None:
            plt.savefig(save_filename)    