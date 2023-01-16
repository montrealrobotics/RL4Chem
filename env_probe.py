import numpy as np 
from collections import defaultdict

# Thank you https://andyljones.com/
# Taken from https://andyljones.com/posts/rl-debugging.html#probe

class level_one_env(object):
    def __init__(self, episode_length=1):
        # Set episode length
        self.episode_length = episode_length

        # Define action space 
        self.action_space = [0]

        # Define observation space 
        self.observation_space = [[0]]

        # Set observation and action space length
        self.observation_length = 1
        self.action_space_length = 1

        # Initialize episode stats
        self.t = 0
        self.r = 0

    def reward(self):
        return 1

    def reset(self):
        # Initialize episode stats
        self.t = 0
        self.r = 0
        return self.observation_space[0]
    
    def step(self, action):
        assert self.t <= self.episode_length, 'episode has exceeded predefined limit, use env.reset()'
        assert action >=0 and action < self.action_space_length
        info = defaultdict(dict)

        reward = self.reward()

        self.t += 1
        self.r += reward

        if self.t >= self.episode_length:
            done = True
        else:
            done = False

        if done:
            info["episode"]["r"] = self.r
            info["episode"]["l"] = self.t 
        
        return self.observation_space[0], reward, done, info

class level_two_env(object):
    def __init__(self, episode_length=1):
        # Set episode length
        self.episode_length = episode_length

        # Define action space 
        self.action_space = [0]

        # Define observation space 
        self.observation_space = [[-1], [1]]

        # Set observation and action space length
        self.num_observations = 2
        self.observation_length = 1
        self.action_space_length = 1

        # Initialize episode stats
        self.t = 0
        self.r = 0
        self.state = self.observation_space[np.random.randint(self.num_observations)]

    def reward(self):
        if self.state == [-1]:
            reward = -1
        elif self.state == [1]:
            reward = 1
        else:
            raise ValueError
        return reward 

    def reset(self):
        # Initialize episode stats
        self.t = 0
        self.r = 0
        self.state = self.observation_space[np.random.randint(self.num_observations)]
        return self.state
    
    def step(self, action):
        assert self.t <= self.episode_length, 'episode has exceeded predefined limit, use env.reset()'
        assert action >=0 and action < self.action_space_length
        info = defaultdict(dict)

        reward = self.reward()

        self.t += 1
        self.r += reward

        if self.t >= self.episode_length:
            done = True
        else:
            done = False

        if done:
            info["episode"]["r"] = self.r
            info["episode"]["l"] = self.t 
        
        return self.observation_space[np.random.randint(self.num_observations)], reward, done, info

class level_three_env(object):
    def __init__(self, episode_length=2):
        # Set episode length
        self.episode_length = episode_length

        # Define action space 
        self.action_space = [0]

        # Define observation space 
        self.observation_space = [[0], [1]]

        # Set observation and action space length
        self.num_observations = 2
        self.observation_length = 1
        self.action_space_length = 1

        # Initialize episode stats
        self.t = 0
        self.r = 0
        self.state = self.observation_space[0]

    def reward(self):
        if self.state == [0]:
            reward = 0
        elif self.state == [1]:
            reward = 1
        else:
            raise ValueError
        return reward 

    def reset(self):
        # Initialize episode stats
        self.t = 0
        self.r = 0
        self.state = self.observation_space[0]
        return self.state
    
    def step(self, action):
        assert self.t <= self.episode_length, 'episode has exceeded predefined limit, use env.reset()'
        assert action >=0 and action < self.action_space_length
        info = defaultdict(dict)

        reward = self.reward()

        self.t += 1
        self.r += reward

        if self.t >= self.episode_length:
            done = True
            info["episode"]["r"] = self.r
            info["episode"]["l"] = self.t
            self.state = self.observation_space[np.random.randint(self.num_observations)] 
        else:
            done = False
            self.state = self.observation_space[self.t]

        return self.state, reward, done, info

class level_three_env(object):
    def __init__(self, episode_length=2):
        # Set episode length
        self.episode_length = episode_length

        # Define action space 
        self.action_space = [0]

        # Define observation space 
        self.observation_space = [[0], [1]]

        # Set observation and action space length
        self.num_observations = 2
        self.observation_length = 1
        self.action_space_length = 1

        # Initialize episode stats
        self.t = 0
        self.r = 0
        self.state = self.observation_space[0]

    def reward(self):
        if self.state == [0]:
            reward = 0
        elif self.state == [1]:
            reward = 1
        else:
            raise ValueError
        return reward 

    def reset(self):
        # Initialize episode stats
        self.t = 0
        self.r = 0
        self.state = self.observation_space[0]
        return self.state
    
    def step(self, action):
        assert self.t <= self.episode_length, 'episode has exceeded predefined limit, use env.reset()'
        assert action >=0 and action < self.action_space_length
        info = defaultdict(dict)

        reward = self.reward()

        self.t += 1
        self.r += reward

        if self.t >= self.episode_length:
            done = True
            info["episode"]["r"] = self.r
            info["episode"]["l"] = self.t
            self.state = self.observation_space[np.random.randint(self.num_observations)] 
        else:
            done = False
            self.state = self.observation_space[self.t]

        return self.state, reward, done, info

class level_four_env(object):
    def __init__(self, episode_length=1):
        # Set episode length
        self.episode_length = episode_length

        # Define action space 
        self.action_space = [0, 1]

        # Define observation space 
        self.observation_space = [[0]]

        # Set observation and action space length
        self.num_observations = 1
        self.observation_length = 1
        self.action_space_length = 2

        # Initialize episode stats
        self.t = 0
        self.r = 0

    def reward(self, action):
        if action == 0:
            reward = -1
        elif action == 1:
            reward = 1

        return reward 

    def reset(self):
        # Initialize episode stats
        self.t = 0
        self.r = 0
        return self.observation_space[0]
    
    def step(self, action):
        assert self.t <= self.episode_length, 'episode has exceeded predefined limit, use env.reset()'
        assert action >=0 and action < self.action_space_length
        info = defaultdict(dict)

        reward = self.reward(action)

        self.t += 1
        self.r += reward

        if self.t >= self.episode_length:
            done = True
            info["episode"]["r"] = self.r
            info["episode"]["l"] = self.t
        else:
            done = False

        return self.observation_space[0], reward, done, info

class level_five_env(object):
    def __init__(self, episode_length=1):
        # Set episode length
        self.episode_length = episode_length

        # Define action space 
        self.action_space = [0, 1]

        # Define observation space 
        self.observation_space = [[-1], [1]]

        # Set observation and action space length
        self.num_observations = 2
        self.observation_length = 1
        self.action_space_length = 2

        # Initialize episode stats
        self.t = 0
        self.r = 0
        self.state = self.observation_space[np.random.randint(self.num_observations)]

    def reward(self, state, action):
        if action == 0 and state == [-1]:
            reward = 1
        elif action == 1 and state == [1]:
            reward = 1
        elif action == 0 and state == [1]:
            reward = -1
        elif action == 1 and state == [-1]:
            reward = -1
        else:
            raise ValueError

        return reward 

    def reset(self):
        # Initialize episode stats
        self.t = 0
        self.r = 0
        self.state = self.observation_space[np.random.randint(self.num_observations)]
        return self.state
    
    def step(self, action):
        assert self.t <= self.episode_length, 'episode has exceeded predefined limit, use env.reset()'
        assert action >=0 and action < self.action_space_length
        info = defaultdict(dict)

        reward = self.reward(self.state, action)
        self.state = self.observation_space[np.random.randint(self.num_observations)]

        self.t += 1
        self.r += reward

        if self.t >= self.episode_length:
            done = True
            info["episode"]["r"] = self.r
            info["episode"]["l"] = self.t
        else:
            done = False

        return self.state, reward, done, info

if __name__ == '__main__':
    np.random.seed(1)
    probe_envs = [level_one_env, level_two_env]
    
    for env_class in probe_envs:
        env = env_class()
        state = env.reset()
        done = False
        while not done:
            action = np.random.randint(env.action_space_length)
            state, reward, done, info = env.step(action)

            print(state, reward, done, info)