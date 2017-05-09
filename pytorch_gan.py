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


def decorate_with_diffs(data, exponent):
    mean = torch.mean(data.data, 1)
    mean_broadcast = torch.mul(torch.ones(data.size()), mean.tolist()[0][0])
    diffs = torch.pow(data - Variable(mean_broadcast), exponent)
    return torch.cat([data, diffs], 1)

d_sampler = get_distribution_sampler(data_mean, data_stddev)
gi_sampler = get_generator_input_sampler()
G = Generator(input_size=gen_input_size, hidden_size=gen_hidden_size, output_size=gen_output_size)
D = Discriminator(input_size=d_input_func(dis_input_size), hidden_size=dis_hidden_size, output_size=dis_output_size)
criterion = nn.BCELoss()  # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss
d_optimizer = optim.Adam(D.parameters(), lr=d_learning_rate, betas=optim_betas)
g_optimizer = optim.Adam(G.parameters(), lr=g_learning_rate, betas=optim_betas)



for epoch in range(num_epochs):
    for d_index in range(d_steps):
        # 1. Train D on real+fake
        D.zero_grad()

        #  1A: Train D on real
        d_real_data = Variable(d_sampler(d_input_size))
        d_real_decision = D(preprocess(d_real_data))
        d_real_error = criterion(d_real_decision, Variable(torch.ones(1)))  # ones = true
        d_real_error.backward() # compute/store gradients, but don't change params

        #  1B: Train D on fake
        d_gen_input = Variable(gi_sampler(minibatch_size, g_input_size))
        d_fake_data = G(d_gen_input).detach()  # detach to avoid training G on these labels
        d_fake_decision = D(preprocess(d_fake_data.t()))
        d_fake_error = criterion(d_fake_decision, Variable(torch.zeros(1)))  # zeros = fake
        d_fake_error.backward()
		d_optimizer.step() # Only optimizes D's parameters; changes based on stored gradients from backward()

# def model():


# def main():
# 	model()


# if __name__ == "__main__":
# 	main()