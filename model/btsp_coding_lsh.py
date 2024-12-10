import numpy as np
from .lsh import LSH
from .utils import *
import torch


class btsp_lsh(LSH):
    def __init__(
        self,
        data,
        hash_length,
        sampling_ratio,
        embedding_size,
        fq_constant=2,
        binary_mode=False,
        label=None,
        lrs=0.1,
        device="cpu",
    ):
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
        # set the device

        data = torch.Tensor(data.reshape(data.shape[0], -1)).to(device)

        self.data = data
        # self.data = (self.data - self.data.min()) / (self.data.max() - self.data.min())

        self.hashes = torch.zeros((data.shape[0], embedding_size))
        sample_num = int(0.5 * data.shape[1])
        weight_seed = torch.rand(data.shape[1], embedding_size)
        yindices = torch.arange(weight_seed.shape[1])[None, :]
        xindices = weight_seed.argsort(axis=0)[
            -sample_num:, :
        ]  # maintain the top-k values and set to one
        weights_mask = torch.zeros_like(weight_seed)
        weights_mask[xindices, yindices] = 1  # sparse projection vectors

        self.data = data
        # self.data = (self.data - self.data.min()) / (self.data.max() - self.data.min())

        self.hashes = torch.zeros((data.shape[0], embedding_size),device=device)
        sample_num = int(
            sampling_ratio * data.shape[1]
        )  # the sparsity 0.6 is used in our paper
        weight_seed = torch.rand(data.shape[1], embedding_size,device=device)
        yindices = torch.arange(weight_seed.shape[1],device=device)[None, :]
        xindices = weight_seed.argsort(axis=0)[
            -sample_num:, :
        ]  # maintain the top-k values and set to one
        weights_mask = torch.zeros_like(weight_seed,device=device)
        weights_mask[xindices, yindices] = 1  # sparse projection vectors
        # weights = torch.rand(data.shape[1], embedding_size) * weights_mask
        plateaus = torch.rand(data.shape[0], embedding_size, device=device) < (
            fq_constant * hash_length / embedding_size
        )
        plateaus = plateaus.float()
        weights = data.T @ plateaus
        weights = (weights % 2).float() * weights_mask
        self.weights = weights.float()
        # representation dimension
        if not binary_mode:
            sum_spike = 0.0
            inp_spike = self.data > torch.rand(self.data.shape[0], self.data.shape[1], device=device)
            self.mem = inp_spike.float() @ self.weights
            self.hashes = torch.zeros_like(self.mem, device=device)
            xindices = np.arange(data.shape[0])[:, None]
            yindices = self.mem.argsort(axis=1)[:, -hash_length:]
            self.hashes[xindices, yindices] = self.mem[
                xindices, yindices
            ]  # masking the neurons whose mem are less than the threshold nps.
        else:
            self.mem = self.data @ self.weights
            self.hashes = torch.zeros_like(self.mem, device=device)
            xindices = np.arange(data.shape[0])[:, None]
            yindices = self.mem.argsort(axis=1)[:, -hash_length:]
            self.hashes[xindices, yindices] = 1

    def update_hash(self, data, binary_mode=False):
        activations = data @ self.weights
        xindices = np.arange(data.shape[0])[:, None]
        yindices = activations.argsort(axis=1)[
            :, -self.hash_length :
        ]  # representation dimension
        hashes = np.zeros((data.shape[0], self.embedding_size))
        if not binary_mode:
            hashes[xindices, yindices] = activations[
                xindices, yindices
            ]  # masking the neurons whose mem are less than the threshold nps.
        else:
            hashes[xindices, yindices] = 1
        return hashes
