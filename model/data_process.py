import numpy as np
import os
def data_downsampling(data, p = 2,pooling_method='max'):
	"""
	:param data: batchsize * inp_dim(e.g.,784)
	:param p:
	:return:
	"""
	# pooling_method = 'max'
	import torch
	import torch.nn.functional as F
	if len(data.shape) < 4:
		data = torch.unsqueeze(torch.Tensor(data.reshape(-1, 28,28)),dim=1)
	if pooling_method == 'max':
		z = F.max_pool2d(data, p).numpy()
	else:
		z = F.avg_pool2d(data, p).numpy()
	f = z.reshape(-1, int(784/p**2))
	# f = np.float32(f > 0.5)
	import matplotlib.pyplot as plt
	# plt.figure()
	# for i in range(9):
	# 	plt.subplot(3,3,i+1)
	# 	plt.imshow(z[i].squeeze())
	return f

def standardize_data(D, do_norm=None,SET_MEAN = 1):
	""" Performs several standardizations on the data.
			1) Makes sure all values are non-negative.
			2) Sets the mean of example to SET_MEAN.
			3) Applies normalization if desired.
	"""

	# 1. Add the most negative number per column (ORN) to make all values >= 0.
	N, DIM = D.shape
	for col in np.arange(DIM):
		D[:, col] += abs(min(D[:, col]))

	# 2. Set the mean of each row (odor) to be SET_MEAN.
	# SET_MEAN = 100
	for row in np.arange(N):
		# Multiply by: SET_MEAN / current mean. Keeps proportions the same.
		D[row, :] = D[row, :] * ((SET_MEAN / np.mean(D[row, :])))
		D[row, :] = list(map(int, D[row, :]))
		# assert abs(np.mean(D[row, :]) - SET_MEAN) <= 1
	return D

def read_generic_data(filename, N, DIM):
	""" Generic reader for: sift, gist, corel, mnist, glove, audio, msong. """
	D = np.zeros((N, DIM))
 
	# find the path to THIS module
	mypath = os.path.dirname(os.path.realpath(__file__))
	open_path = os.path.join(mypath, filename)
	with open(open_path) as f:
		for line_num, line in enumerate(f):
			cols = line.strip().split(",")
			assert len(cols) == DIM
			# D[line_num,:] = map(float,cols)
			D[line_num, :] = list(map(float, cols))
		# D[line_num,:] *= -1 # to invert distribution?

	assert line_num + 1 == N
	return D

def generate_data(DATASET):
	if DATASET == "sift10k":
		N = 10000
		DIM = 128
		D = read_generic_data("../data/sift/sift10k.txt", N, DIM)

	# Read Gist data: 10,000 images x 960 gist descriptors/features.
	elif DATASET == "gist10k":
		N = 10000
		DIM = 960
		D = read_generic_data("../data/gist/gist10k.txt", N, DIM)

	# Read MNIST data: 10,000 images x 784 pixels.
	elif DATASET == "mnist10k":
		N = 10000
		DIM = 784
		D = read_generic_data("../data/mnist/mnist10k.txt", N, DIM)

	elif DATASET == "mnist10k_label":
		N = 10000
		DIM = 784
		# D = read_generic_data("../data/mnist/mnist10k.txt", N, DIM)
		import json
		mypath = os.path.dirname(os.path.realpath(__file__))

		f1 = open(os.path.join(mypath, '../data/mnist/emnist-mnist-test-labels.json'))
		label = json.load(f1)
		f2 = open(os.path.join(mypath, '../data/mnist/emnist-mnist-test-images.json'))
		image = json.load(f2)
		images = np.array(image)
		y = [eval(x) for x in label]
		D = (images, np.array(y))
		f1.close(), f2.close()

	# Read Glove data: 10,000 words x 300 features.
	elif DATASET == "glove10k":
		N = 10000
		DIM = 300
		D = read_generic_data("../data/glove/glove10k.txt", N, DIM)  # np.random.randn(100_000,self.indim)
	else:
		assert False
	return N, DIM, D


