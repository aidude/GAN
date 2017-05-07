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



(name, preprocess, d_input_func) = ("Data and variances", lambda data: decorate_with_diffs(data, 2.0), lambda x: x * 2)
print("Using data [%s]" % (name))


class Parameters(object):
	data_mean = 4.0
	data_std_dev = 2.0
	gen_input_size = 1     # Random noise dimension coming into generator, per output vector
	gen_hidden_size = 50   # Generator complexity
	gen_output_size = 1    # size of generated output vector
	dis_input_size = 100   # Minibatch size - cardinality of distributions
	dis_hidden_size = 50   # Discriminator complexity
	dis_output_size = 1    # Single dimension for 'real' vs. 'fake'
	mini_batch_size = dis_input_size

	dis_learning_rate = 2e-4  
	gen_learning_rate = 2e-4
	optimum_betas = (0.9, 0.999)
	num_epochs = 30000
	print_interval = 500

	k_steps = 1  # 'k' steps in the original GAN paper. Can put the discriminator on higher training freq than generator
	g_steps = 1


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


def extract_data():
	return v.data.storage().tolist()

def stats(d):
	return [np.mean(d), np.std(d)]

def model():
