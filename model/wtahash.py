from .flylsh import flylsh
import numpy as np
import torch
class WTAHash(flylsh):
	# implements Google's WTA hash
	def __init__(self, data, code_length, K=4, device='cpu'):
		"""
		hash_length: code length m in the paper
		"""
		self.hash_length = code_length
		self.embedding_size = K * code_length
		self.data = data
		# self.data = data - np.mean(data, axis=1)[:, None]

		self.thetas = [torch.randint(0, data.shape[1], (K,), device=device) for _ in range(code_length)]
		xindices = torch.arange(data.shape[0], dtype=torch.int32, device=device)[:, None]
		yindices = self.data[:, self.thetas[0]].argmax(dim=1)
		# this line permutes the vectors with theta[0], takes the first K elements and computes
		# the index corresponding to max element

		this_hash = torch.zeros((data.shape[0], K), device=device)  # a K dim binary vector for each data point
		this_hash[xindices, yindices] = 1  # set the positions corresponding to argmax to True
		self.hashes = this_hash[:]

		for t in self.thetas[1:]:
			this_hash = torch.zeros((data.shape[0], K), device=device)
			yindices = self.data[:, t].argmax(axis=1)  # same as line 162 above
			this_hash[xindices, yindices] = True
			self.hashes = torch.concatenate((self.hashes, this_hash), axis=1).to(device)

		self.maxl1distance = 2 * self.hash_length
