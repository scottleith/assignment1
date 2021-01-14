from bandit_environments import *
from bandit_agents import *
from bandit_policies import *
from bandit_testrunner import *

bandit = BanditEnvironment(k = 10, mu = 0, sigma = 1)

egreedy_10perc = BanditEpsilonGreedyPolicy()
egreedy_1perc = BanditEpsilonGreedyPolicy(epsilon = 0.01)
softmax_policy = BanditSoftmaxPolicy()
pure_greed = BanditEpsilonGreedyPolicy(epsilon = 0)

agent1 = SampleAverageBanditAgent(policy = egreedy_10perc, bandit_env = bandit)
agent2 = SampleAverageBanditAgent(policy = egreedy_1perc, bandit_env = bandit)
agent3 = GradientBanditAgent(bandit_env = bandit)
agent4 = SampleAverageBanditAgent(policy = pure_greed, bandit_env = bandit)

agents = [agent1, agent3]

runner = BanditTestRunner(agents = agents, bandit_env = bandit)

r, o = runner.perform_runs(timesteps = 1000, runs = 2000)

runner.visualize_results(
    save_filename = None,
        title = "Epsilon-Greedy 10% vs. Gradient Bandit", 
        rewards_histories = r, 
        optimal_action_histories = o
        )

