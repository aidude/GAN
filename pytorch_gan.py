# GAN using pytorch...!! 

# amritansh

import argparse
import numpy as np
# from scipy.stats import norm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns

sns.set(color_codes=True)

seed = 42
np.random.seed(seed)



class DataDistribution(object):
    def __init__(self):
        self.mu = 4
        self.sigma = 0.5

    def sample(self, N):
        samples = np.random.normal(self.mu, self.sigma, N)
        samples.sort()
        return samples


class GeneratorDistribution(object):
    def __init__(self, range):
        self.range = range

    def sample(self, N):
        return np.linspace(-self.range, self.range, N) + \
            np.random.random(N) * 0.01

class Generator(nn.Module):
	def __init__(self, input, hidden, output):
		super(Generator, self).__init__()
        self.map_01 = nn.Linear(input, hidden)
        self.map_02 = nn.Linear(hidden, hidden)
		self.map_03 = nn.Linear(hidden, output)
	
	def forward(self,x):
		x = F.elu(self.map_01(x))
		x = F.sigmoid(self.map_02(x))
		return self.map_03(x)



class Discriminator(nn.Module):
	def __init__(self, input, hidden, output):
        super(Discriminator, self).__init__()
        self.map_01 = nn.Linear(input, hidden)
        self.map_02 = nn.Linear(hidden, hidden)
        self.map_03 = nn.Linear(hidden, output)

    def forward(self, x):
        x = F.elu(self.map_01(x))
        x = F.elu(self.map_02(x))
		return F.sigmoid(self.map_03(x))
