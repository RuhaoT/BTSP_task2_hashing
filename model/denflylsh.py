from flylsh import flylsh
import numpy as np

class denseflylsh(flylsh):
	def __init__(self, data, hash_length, sampling_ratio, embedding_size):
		"""
		data: Nxd matrix
		hash_length: scalar
		sampling_ratio: fraction of input dims to sample from when producing a hash
		embedding_size: dimensionality of projection space, m
		Note that in Flylsh, the hash length and embedding_size are NOT the same
		whereas in usual LSH they are
		"""
		self.hash_length = hash_length
		self.embedding_size = embedding_size
		K = embedding_size // hash_length
		self.data = (data - np.mean(data, axis=1)[:, None])
		weights = np.random.random((data.shape[1], embedding_size))
		self.weights = (weights > 1 - sampling_ratio)  # sparse projection vectors
		all_activations = (self.data @ self.weights)
		threshold = 0
		self.hashes = (all_activations >= threshold)  # choose topk activations
		# self.dense_activations=all_activations
		# self.sparse_activations=self.hashes.astype(np.float32)*all_activations #elementwise product
		self.maxl1distance = 2 * self.hash_length
		self.lowd_hashes = all_activations.reshape((-1, hash_length, K)).sum(axis=-1) > 0