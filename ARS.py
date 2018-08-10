# importing libraries
import os
import numpy as np
import gym
from gym import wrappers
import pybullet_envs

# setting hyper parameters 

class Hp():
    def __init__(self):
        # defining variables of objects
        self.nb_steps = 1000 # no. of training loops i.e. no. of times we update the weights
        self.episode_length = 1000 # max. time AI walk on field
        self.learning_rate = 0.02  # how fast AI is learning
        self.nb_directions = 16   # no. of directions
        self.nb_best_directions = 16  # no.of best directions. keep same as nb_directions in starting
        assert self.nb_best_directions <= self.nb_directions
        self.noise = 0.03
        self.seed = 1
        self.env_name = 'HalfCheetahBulletEnv-v0'   # name of environment
        
# Normalizing the states (to improve performance)

class Normalizer():   # refer page 7 section 3.2 of paper
    def __init__(self, nb_inputs):  # nb_inputs is no. of perceptrons
        # initializing the variables required for normalization
        self.n = np.zeros(nb_inputs)  # total number of states. vector of zeros equal to no. of perceptrons
        self.mean = np.zeros(nb_inputs)
        self.mean_diff = np.zeros(nb_inputs)
        self.var = np.zeros(nb_inputs)
    
    def observe(self, x):  # x is new state. function observe is called everytime we observe a new state
        self.n += 1.  # to make it float
        last_mean = self.mean.copy()   # saving mean before updating it
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min = 1e-2)  # clip(min = 1e-2) is used to make sure self.var is never equal to zero
    
    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)  # std. deviation
        return (inputs - obs_mean)/obs_std  # normalized values

# Building the AI 
# Algorithm is based on exploration on the space of policies. We will explore many policies and converge on the one which returns the best output

class Policy():
    def __init__(self, input_size, output_size):  # there are many outputs
        self.theta = np.zeros((output_size, input_size))  # matrix of weights of neurons of perceptron
        
    def evaluate(self, input, delta = None, direction = None):  # page 6 algorithm 2 step5 V2. delta - matrix of small number helps in choosing direction. direction can have 3 values: +ve, -ve and none.
        if direction is None:
            return self.theta.dot(input)  # matrix multiplication theta x input
        elif direction == "positive":
            return (self.theta + hp.noise*delta).dot(input)   # hp is object of class Hp where noise is defined
        else:  # i.e. negative direction
            return (self.theta - hp.noise*delta).dot(input)
    
    def sample_deltas(self):  # step 4
        return [np.random.randn(*self.theta.shape) for i in range(hp.nb_directions)]  # returning matrix(of same size of theta matrix) of random small values of delta. *self.theta.shape gives the dimension of theta matrix 
    
    def update(self, rollouts, sigma_r): # step 7. rollouts contains rewards in positive and negative direction, and delta
        step = np.zeros(self.theta.shape)
        for r_pos, r_neg, d in rollouts:
            step +=  (r_pos - r_neg) * d
        self.theta += hp.learning_rate / (hp.nb_best_directions * sigma_r) * step

# Exploring the policy on one specific direction and over one episode

def explore(env, normalizer, policy, direction = None, delta = None):
    state = env.reset()  # returns first state. env is object of pybullet
    done = False # boolean. true if end of episode. False in starting bcoz episode is not done
    num_plays = 0.  # no. of actions
    sum_rewards = 0  # accumulative rewards
    while not done and num_plays < hp.episode_length:   
        normalizer.observe(state)
        state = normalizer.normalize(state)  # normalizing state
        action = policy.evaluate(state, delta, direction)  # evaluating the policy
        state, reward, done, _ = env.step(action)  # step fn from environment object of pybullet returns next state of the environment the reward obtained after playing the action and whether or not the episode is done
        # +1 for very high positive reward and -1 for very high negative reward, to avoid bais
        reward = max(min(reward, 1), -1)
        sum_rewards += reward
        num_plays += 1
    return sum_rewards

# Training the AI

def train(env, policy, normalizer, hp):  # step 3
    
    for step in range(hp.nb_steps):  # hp.nb_steps refer to no. of training loops
        
        # initializing the perturbations deltas and the positive/negative rewards
        deltas = policy.sample_deltas()  # we get 16 matrixs for one for each 16 directions
        positive_rewards = [0] * hp.nb_directions  # list of 16 zeros
        negative_rewards = [0] * hp.nb_directions  # negative rewards does not mean less than zero. it means rewards in opposite direction
        
        # getting the positive rewards in positive direction
        for k in range(hp.nb_directions):  # looping through the no. of directions
            positive_rewards[k] = explore(env, normalizer, policy, direction = "positive", delta = deltas[k])   # direction == positive as in line 55. deltas[k] refers to delta in kth direction
        
        # getting the negative rewards in negative direction
        for k in range(hp.nb_directions):  # looping through the no. of directions
            negative_rewards[k] = explore(env, normalizer, policy, direction = "negative", delta = deltas[k])   # direction == positive as in line 57. deltas[k] refers to delta in kth direction
        
        # Gethering all the positive and negative rewards to compute the standard deviation of these rewards. section 3.1
        all_rewards = np.array(positive_rewards + negative_rewards)
        sigma_r = all_rewards.std()  # returns std deviation of all rewards
        
        # sorting the rollouts by max(r_pos, r_neg) and selecting the best directions. Step 6
        scores = {k:max(r_pos, r_neg) for k, (r_pos, r_neg) in enumerate(zip(positive_rewards, negative_rewards))}  # storing in a dictionary
        order = sorted(scores.keys(), key = lambda x:scores[x])[:hp.nb_best_directions]  # sorting by keys. we are only considering the best directions.
        rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]
        
        # Updating our policy. Step 7. weights are updated here to reach highest reward
        policy.update(rollouts, sigma_r)
        
        # printing the final reward of the policy after update
        reward_evaluation = explore(env, normalizer, policy)  # direction and delta are none by default
        print ('Step: ', step, 'Reward: ', reward_evaluation)


# Running the main code

def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path
work_dir = mkdir('exp', 'brs')
monitor_dir = mkdir(work_dir, 'monitor')

hp = Hp()  # object of Hp class to get all hyperparameters
np.random.seed(hp.seed)
env = gym.make(hp.env_name)
env = wrappers.Monitor(env, monitor_dir, force = True) # to see the video on monitor. Force = true so that video doesnt stop due to warnings
nb_inputs = env.observation_space.shape[0]
nb_outputs = env.action_space.shape[0]
policy = Policy(nb_inputs, nb_outputs)
normalizer = Normalizer(nb_inputs)
train(env, policy, normalizer, hp)
        
        
        
    

            
        
        
        

        
        
        
        
        
    


