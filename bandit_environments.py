import numpy as np

class BanditEnvironment():
    """
    The K-Armed Bandit. I called this "BanditEnvironment" because the
    environment is the problem the learner agent is trying to solve; it is
    the thing they act upon that gives reward feedback.

    Attributes:
        k: The number of actions ("levers", "arms") in the bandit problem.
        mu: The mean of the normal distribution from which we draw q*, the 
            true action value for each arm. Note that we don't have to draw
            q* from a normal distribution, and might work that into this class
            or a new class if we wanted to.
        sigma: The standard deviation of the above distribution.
        true_action_values: q*, the true action values.
        optimal action: The action index that has the highest q*.
    """

    def __init__(self, k, mu, sigma):
        self.k = k
        self.mu = mu
        self.sigma = sigma
        self.true_action_values = np.random.normal(mu, sigma, k)
        self.optimal_action = np.random.choice(
            np.where(self.true_action_values == self.true_action_values.max())[0]
        )
    
    def reset_true_action_values(self):
        """
        Resets the values of all actions to 0. This is for resets between
        runs, or if you want to see how your learner responds to a "system
        shock" where the dynamics change suddenly. If you want to provide a 
        more serious shock than the BanditEnvironment object's current mu
        and sigma, you can change these manually, e.g.,
        my_bandit_env.mu = 5, etc.
        """
        self.true_action_values = np.random.normal(self.mu, self.sigma, self.k)
        self.optimal_action = np.random.choice(
            np.where(self.true_action_values == self.true_action_values.max())[0]
        )

    def random_walk_action_values(self, mu, sigma):
        """
        A method that adds a random value to each element of the true action
        values. 

        This is for part of Assignment 1! (Exercise 2.5 in the text)

        Args:
            mu: The mean of the normal distribution from which the random
                walk value is drawn.
            sigma: The standard deviation of the above distribution.
        """

        ### YOUR CODE GOES HERE 

    def take_action_emit_reward(self, action):
        """
        Accepts an action index, draws reward from that q* distribution, and
        outputs a reward for that action + whether or not that action was
        optimal.

        Args:
            action: The index of the action to take.
        Returns:
            reward: A scalar (float) reward value.
        """
        return np.random.normal( self.true_action_values[action]), \
            action == self.optimal_action

