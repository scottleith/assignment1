import numpy as np
from bandit_policies import BanditSoftmaxPolicy ## We will eventually need this
# for the Gradient Bandit agent.

class BanditAgent():
    """
    This is a base class that will not be explicitly used in code.

    So far we have learned that there are various ways of estimating q*. We
    focused on the sample average method so far in our coding, but of course
    there are other ways like the gradient bandit algorithm.

    Because agents all have things in common - choosing actions according to
    a policy, action-value estimates, updating, etc. - we can put the common
    factors in this base class or "parent" class, and have the other agent
    classes just inherit. This keeps our code cleaner and more flexible.

    Attributes:
        policy: An instance of the BanditPolicy class set. Used for choosing
                actions.
        bandit_env: An instance of the BanditEnvironment class set. Used for 
            taking actions, setting action-value vectors (i.e., contains k, 
            which we need for initializing q and n).
        initial_estimates: The value used to populate the initial action-value
        estimates vector (usually 0, can be higher for optimistic initial values).

    """

    def __init__(self, policy, bandit_env, initial_estimates = 0):
        self.policy = policy
        self.bandit_env = bandit_env
        self.initial_estimates = initial_estimates # for easy reuse
        self.action_value_estimates = self.initial_estimates * \
            np.ones(self.bandit_env.k)
        self.action_counts = np.zeros(self.bandit_env.k)
    
    def __str__(self):
        """
        It is generally good practice to use the __str__ method to enable
        quick checking of relevant things about a class. Here, we might want to
        check an Agent's policy type and number of actions.
        """
        return "Policy: {policy}, K = {k}".format(
            policy = self.policy.__str__(), 
            k = self.bandit_env.k
        )
    
    def reset_action_value_estimates(self, new_initial_estimates = None):
        """
        Resets the agent's action-value estimates and counts their initially
        chose estimate (usualy 0), or to a new initial estimate passed as an
        argument. All agents will need this, so we can put it here.
        
        Args:
            new_initial_estimates: A value that will replace the attribute
                initial_estimates in the reset.
        """
        pass
    
    def choose_action(self):
        """
        Chooses an action to take against the environment. All agents will be
        doing this, so we put it in the base/parent class.

        Returns:
            An action index.
        """
        action = self.policy.choose_action(self.action_value_estimates)
        return action



class SampleAverageBanditAgent(BanditAgent): # Note the inheritance
    """
    A basic sample-average based bandit learner.

    Note that because we included all the basic methods in the base class,
    we do not need to repeat ourselves here. Much cleaner! All we need to do
    is override the update_action_value_estimates function and we're done!
    """
    def reset_action_value_estimates(self, new_initial_estimates = None):
        if new_initial_estimates is not None:
            self.action_value_estimates[:] = new_initial_estimates
        else:
            self.action_value_estimates[:] = self.initial_estimates
        self.action_counts[:] = 0

    def update_action_value_estimates(self, action, reward):
        """
        Our basic sample-average incremental update, where the learning rate
        is 1 / N(A)
        
        Args:
            action: The index of the action to have its count and action-value
            estimate updated.
            reward: The reward received as a result of choosing the action above
            that is used in the update equation. 
        """
        self.action_counts[action] += 1
        learning_rate = 1 / self.action_counts[action]
        self.action_value_estimates[action] += learning_rate * \
            (reward - self.action_value_estimates[action])



class GradientBanditAgent(BanditAgent):
    """
    Gradient Bandit as on page 37 of the text. This bandit agent relies on 
    preference values instead of 'raw' action-value estimates of reward.

    It also uses a softmax policy, which chooses actions according to a 
    probability distribution. Because this is a bit different, we have 
    slightly more code to write.

    Attributes:
        policy: A BanditSoftmaxPolicy instance.
        bandit_env: A BanditEnvironment instace.
        initial_estimates: The value used to populate the initial action-value
            estimates vector (usually 0, can be higher for optimistic initial 
            values).
        action_value_estimates: The vector of preference values, of length k.
        learning_rate: The learning rate applied to R - average_reward.
        average_reward: The average reward experienced so far from all actions.
    """   

    def __init__(self, bandit_env, initial_estimates = 0, learning_rate = 0.1):
        self.policy = BanditSoftmaxPolicy()
        self.bandit_env = bandit_env
        self.action_value_estimates = initial_estimates * \
            np.ones(self.bandit_env.k) 
        self.learning_rate = learning_rate
        self.average_reward = 0
        self.initial_estimates = initial_estimates
    
    def reset_action_value_estimates(self, new_initial_estimates = None):
        if new_initial_estimates is not None:
            self.action_value_estimates[:] = new_initial_estimates
        else:
            self.action_value_estimates[:] = self.initial_estimates

    def update_action_value_estimates(self, action, reward):
        """
        Equations 2.12 in the text.
        """
        current_policy = self.policy.softmax(self.action_value_estimates)
        diff = reward - self.average_reward
        self.action_value_estimates[action] += self.learning_rate*diff*\
            (1-current_policy[action])
        for a in range(0,self.bandit_env.k):
            if a == action:
                continue
            else:
                self.action_value_estimates[a] -= self.learning_rate*diff* \
                    current_policy[a]

