import os
import time
import pdb
from tqdm import *
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn import Parameter
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from sklearn.cluster import KMeans
import numpy as np
from tqdm import *
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import pandas as pd
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ClusteringLayer(nn.Module):
	def __init__(self, n_clusters=20, hidden=64, cluster_centers=None, alpha=1.0):
		super(ClusteringLayer, self).__init__()
		self.n_clusters = n_clusters
		self.alpha = alpha
		self.hidden = hidden
		if cluster_centers is None:
			initial_cluster_centers = torch.zeros(
			self.n_clusters,
			self.hidden,
			dtype=torch.float
			).cuda()
			nn.init.xavier_uniform_(initial_cluster_centers)
		else:
			initial_cluster_centers = cluster_centers
		self.cluster_centers = Parameter(initial_cluster_centers)
	def forward(self, x):
		norm_squared = torch.sum((x.unsqueeze(1) - self.cluster_centers)**2, 2)
		numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
		power = float(self.alpha + 1) / 2
		numerator = numerator**power
		t_dist = (numerator.t() / torch.sum(numerator, 1)).t() #soft assignment using t-distribution
		return t_dist

class DEC(nn.Module):
	def __init__(self, n_clusters=20, hidden=64, cluster_centers=None, alpha=1.0):
		super(DEC, self).__init__()
		self.n_clusters = n_clusters
		self.alpha = alpha
		self.hidden = hidden
		self.cluster_centers = cluster_centers
		self.clusteringlayer = ClusteringLayer(self.n_clusters, self.hidden, self.cluster_centers, self.alpha)

	def target_distribution(self, q_):
		weight = (q_ ** 2) / torch.sum(q_, 0)
		return (weight.t() / torch.sum(weight, 1)).t()

	def forward(self, x):
		return self.clusteringlayer(x)



	

	
    	