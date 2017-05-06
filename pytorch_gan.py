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

# sns.set(color_codes=True)

# seed = 42
# np.random.seed(seed)

(name, preprocess, d_input_func) = ("Data and variances", lambda data: decorate_with_diffs(data, 2.0), lambda x: x * 2)
print("Using data [%s]" % (name))


class DataDistribution(object):
    def __init__(self):
        self.mu = 4
        self.sigma = 0.5

    def sample(self):
        return lambda n: torch.Tensor(np.random.normal(mu, sigma, (1, n))) # Gaussian

class GeneratorDistribution(object):
    
    def sample(self):
        return lambda m, n: torch.rand(m, n)

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


def model():
