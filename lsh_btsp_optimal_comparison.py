"""This module compares the local-sensitive hashing performance of BTSP-like hashing with other hashing algorithms.

For BTSP algorithm, we fine-tune parameters to find the optimal configuration.
For other hashing algorithms, we use the default configuration."""

import dataclasses
import os
import sys
import itertools
import numpy as np
import torch
import torch.multiprocessing as multiprocessing
import pandas as pd

from experiment_framework_bic.auto_experiment import auto_experiment
from experiment_framework_bic.utils import logging, parameterization
from model import utils, lsh, btsp_coding_lsh, flylsh, wtahash, data_process

import time

utils.setup_seed(1)


@dataclasses.dataclass
class MetaParams:
    """Meta parameters for the experiment."""

    dataset_name: str | list[str] = "none"
    training_data_num: int | list[int] = -1
    hash_length: int | list[int] = -1
    space_ratio: int | list[int] = -1
    btsp_fq_constant: float | list[list[float]] = 2 # in-experiment fine-tuning
    binary_mode: bool | list[bool] = False
    btsp_sampling_ratio: float | list[list[float]] = 0.05 # in-experiment fine-tuning
    random_seed: int | list[int] = 1
    experiment_index: int = -1


class BTSPOptimalLSHExperiment(auto_experiment.ExperimentInterface):
    """This experiment compare LSH algorithms with mAP."""

    def __init__(self) -> None:

        self.meta_params = MetaParams(
            dataset_name=['mnist10k', 'glove10k', 'sift10k', 'gist10k'],
            training_data_num=5000,
            hash_length=[24, 4, 8, 12, 16, 20, 2],
            space_ratio=[20, 4, 8, 12, 1],
            btsp_fq_constant=[[1,2,3,5,7,8,10]],
            binary_mode=True, # we have verified that binary_mode has a negligible effect on the results
            btsp_sampling_ratio=[[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]],
            random_seed=[1,2,3],
        )
        # RH: the above parameter would faciliate your rapid validation. We should consider the following setup for publishable results
        # hash_lengths = [2, 4, 8, 12, 16, 20, 24]
        # space_ratio_list = [ 1, 4, 8, 12, 20]
        
        # for debugging
        # self.meta_params = MetaParams(
        #     dataset_name=['mnist10k', 'sift10k', 'gist10k', 'glove10k'],
        #     training_data_num=5000,
        #     hash_length=24,
        #     space_ratio=16,
        #     btsp_fq_constant=[[2]],
        #     binary_mode=True,
        #     btsp_sampling_ratio=[[0.2]],
        #     random_seed=1,
        # )

        self.experiment_name = "mAPBTSPcomparison_3trials"
        self.experiment_folder = logging.init_experiment_folder(
            data_folder="./results/", experiment_name=self.experiment_name, timed=False
        )

        self.params: pd.DataFrame = pd.DataFrame()
        self.multiprocess_manager = multiprocessing.Manager()
        self.result_dict = self.multiprocess_manager.dict()
        self.result_dict["experiment_index"] = self.multiprocess_manager.list()
        self.result_dict["input_dim"] = self.multiprocess_manager.list()
        self.result_dict["embedding_size"] = self.multiprocess_manager.list()
        self.result_dict["optimal_btsp_fq_constant"] = self.multiprocess_manager.list()
        self.result_dict["optimal_btsp_sampling_ratio"] = self.multiprocess_manager.list()
        self.result_dict["optimal_btsp_fq"] = self.multiprocess_manager.list()
        self.result_dict["btsp_mAP"] = self.multiprocess_manager.list()
        self.result_dict["fly_mAP"] = self.multiprocess_manager.list()
        self.result_dict["wta_mAP"] = self.multiprocess_manager.list()
        self.result_dict["lsh_mAP"] = self.multiprocess_manager.list()

    def load_parameters(self):
        """Generate all possible combinations of parameters."""
        meta_params: list[MetaParams] = parameterization.recursive_iterate_dataclass(
            self.meta_params
        )
        meta_params_dicts = []
        for index, meta_param in enumerate(meta_params):
            meta_param.experiment_index = index
            meta_params_dicts.append(dataclasses.asdict(meta_param))
        self.params = pd.DataFrame(meta_params_dicts)

        return meta_params

    def load_dataset(self):
        """This experiment load dataset JIT."""
        return self.result_dict

    def execute_experiment_process(self, parameters: MetaParams, dataset):
        """Execute the experiment process."""
        # get current device, this require device to be set before the experiment
        if not torch.cuda.is_available():
            device = "cpu"
        else:
            device = "cuda:" + str(torch.cuda.current_device())
        
        # Load the dataset
        data_name = parameters.dataset_name
        num, dim, data = data_process.generate_data(data_name)
        path = self.experiment_folder
        train_num = parameters.training_data_num
        nnn = int(train_num * 0.02)

        hash_length = parameters.hash_length
        space_ratio = parameters.space_ratio

        binary_mode = parameters.binary_mode
        sampling_ratio = parameters.btsp_sampling_ratio
        
        fq_constant = parameters.btsp_fq_constant

        norm_data = (data - data.min()) / (data.max() - data.min())
        input_dim = norm_data.shape[1]
        train_data = norm_data[:train_num]
        train_data = torch.Tensor(train_data).to(device)

        utils.setup_seed(parameters.random_seed)

        # print("Testing hashing length:", hash_length, "space size", space_ratio)
        embedding_size = int(input_dim * space_ratio)
        
        # fine-tune the BTSP-like hashing for optimal performance
        # prepare all combinations of fq_constant and sampling_ratio for fine-tuning
        ft_combinations = list(itertools.product(*[fq_constant, sampling_ratio]))
        best_mAP = 0
        optimal_fq_constant = 0
        optimal_sampling_ratio = 0
        for ft_comb in ft_combinations:
            optimal_btsp_fq = ft_comb[0]
            sampling_ratio = ft_comb[1]
            btsp_model = btsp_coding_lsh.btsp_lsh(
                train_data,
                hash_length,
                sampling_ratio,
                embedding_size,
                fq_constant=optimal_btsp_fq,
                binary_mode=binary_mode,
                device=device,
            )
            btsp_mAP = utils.tesht_map_dist(train_data, btsp_model.hashes)
            if btsp_mAP > best_mAP:
                best_mAP = btsp_mAP
                optimal_fq_constant = optimal_btsp_fq
                optimal_sampling_ratio = sampling_ratio

        optimal_btsp_fq = optimal_fq_constant * hash_length / embedding_size
        # bstp_msg = "mean average precision of flylsh is equal to {:.4f}".format(
        #     btsp_mAP
        # )
        # print("BTSP-like Hashing: ", bstp_msg)
        # the following we compared the performance with advanced hashing algorithmns
        
        # for this experiment, we use default parameters for fly hashing
        fly_model = flylsh.flylsh(
            train_data, # same as btsp
            hash_length, # same as btsp for comparison
            6.0 / float(train_data.shape[1]), # 6 / input_dim 
            embedding_size, # same as btsp
            binary_mode=True,
            device=device,
        )
        fly_mAP = utils.tesht_map_dist(train_data, fly_model.hashes)
        # fly_msg = "mean average precision of flylsh is equal to {:.4f}".format(fly_mAP)
        # print("FLY algo: ", fly_msg)
        WTA_model = wtahash.WTAHash(train_data, hash_length)
        wta_mAP = utils.tesht_map_dist(train_data, WTA_model.hashes)
        # msg = "mean average precision of flylsh is equal to {:.4f}".format(wta_mAP)
        # print("WTA: ", msg)
        lsh_model = lsh.LSH(train_data, hash_length, device=device)
        lsh_mAP = utils.tesht_map_dist(train_data, lsh_model.hashes)
        # msg = "mean average precision of dense_model is equal to {:.4f}".format(lsh_mAP)
        # print("LSH: ", msg)
        # sparse_model = btsp_sparse(train_data, hash_length, sampling_ratio, embedding_size,
        #                       binary_mode=binary_mode)
        # sparse_mAP = tesht_map_dist(train_data, sparse_model.hashes)
        # msg = 'mean average precision of sparse_model is equal to {:.4f}'.format(sparse_mAP)
        # print("Sparse: ", msg)

        dataset["experiment_index"].append(parameters.experiment_index)
        dataset["input_dim"].append(input_dim)
        dataset["embedding_size"].append(embedding_size)
        dataset["optimal_btsp_fq_constant"].append(optimal_fq_constant)
        dataset["optimal_btsp_sampling_ratio"].append(optimal_sampling_ratio)
        dataset["optimal_btsp_fq"].append(optimal_btsp_fq)
        dataset["btsp_mAP"].append(btsp_mAP)
        dataset["fly_mAP"].append(fly_mAP)
        dataset["wta_mAP"].append(wta_mAP)
        dataset["lsh_mAP"].append(lsh_mAP)

    def summarize_results(self):
        """Summarize the results."""

        # convert the result_dict to dict
        self.result_dict = dict(self.result_dict)
        for key in self.result_dict.keys():
            self.result_dict[key] = list(self.result_dict[key])
        
        self.results = pd.DataFrame(self.result_dict)
        print(self.results)
        
        # save both params and results to csv
        self.results.to_csv(os.path.join(self.experiment_folder, "results.csv"))
        self.params.to_csv(os.path.join(self.experiment_folder, "params.csv"))
        
        # merge the results with params
        self.merged_results = pd.merge(self.params, self.results, on="experiment_index")
        # save the merged results to csv
        self.merged_results.to_csv(os.path.join(self.experiment_folder, "merged_results.csv"))
        return super().summarize_results()


if __name__ == "__main__":
    # set torch device to cuda if available
    experiment = auto_experiment.CudaDistributedExperiment(BTSPOptimalLSHExperiment(), "max")
    experiment.run()
    experiment.evaluate()
