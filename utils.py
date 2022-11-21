import gym
import math
import torch
import warnings
import numpy as np 
import torch.nn as nn
import matplotlib.pyplot as plt

from typing import Iterable
from matplotlib import animation

warnings.filterwarnings("ignore", category=DeprecationWarning) 

# general utils 

def linear_schedule(start_sigma: float, end_sigma: float, duration: int, t: int):
    return end_sigma + (1 - min(t / duration, 1)) * (start_sigma - end_sigma)

# env utils 

class CartPolePOMDPWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        high = np.array(
            [
                self.x_threshold * 2,
                self.theta_threshold_radians * 2,
            ],
            dtype=np.float32,
        )

        self.observation_space = gym.spaces.Box(0, 1, (2,))

    def observation(self, obs):
        return [obs[0], obs[1]]

def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):
    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)

#torch utils

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def get_parameters(modules: Iterable[nn.Module]):
    model_parameters = []
    for module in modules:
        model_parameters += list(module.parameters())
    return model_parameters