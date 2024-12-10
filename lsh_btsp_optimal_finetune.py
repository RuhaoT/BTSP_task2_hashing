"""This module explores the following research question:

Under what conditions does BTSP-like hashing reach its optimal performance?"""

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
    btsp_fq_constant: float | list[float] = 2
    binary_mode: bool | list[bool] = False
    btsp_sampling_ratio: float | list[float] = 0.05
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
            btsp_fq_constant=[1,2,3,5,7,8,10],
            binary_mode=True, # we have verified that binary_mode has a negligible effect on the results
            btsp_sampling_ratio=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            random_seed=1,
        )
        # RH: the above parameter would faciliate your rapid validation. We should consider the following setup for publishable results
        # hash_lengths = [2, 4, 8, 12, 16, 20, 24]
        # space_ratio_list = [ 1, 4, 8, 12, 20]
        
        # for debugging
        self.meta_params = MetaParams(
            dataset_name=['mnist10k', 'sift10k', 'gist10k', 'glove10k'],
            training_data_num=5000,
            hash_length=24,
            space_ratio=16,
            btsp_fq_constant=2,
            binary_mode=True,
            btsp_sampling_ratio=0.2,
            random_seed=1,
        )

        self.experiment_name = "mAPBTSPfinetune_parallel_debug2"
        self.experiment_folder = logging.init_experiment_folder(
            data_folder="./results/", experiment_name=self.experiment_name, timed=False
        )

        self.params: pd.DataFrame = pd.DataFrame()
        self.multiprocess_manager = multiprocessing.Manager()
        self.result_dict = self.multiprocess_manager.dict()
        self.result_dict["experiment_index"] = self.multiprocess_manager.list()
        self.result_dict["input_dim"] = self.multiprocess_manager.list()
        self.result_dict["embedding_size"] = self.multiprocess_manager.list()
        self.result_dict["btsp_fq_constant"] = self.multiprocess_manager.list()
        self.result_dict["btsp_sampling_ratio"] = self.multiprocess_manager.list()
        self.result_dict["btsp_fq"] = self.multiprocess_manager.list()
        self.result_dict["btsp_mAP"] = self.multiprocess_manager.list()

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
        return None

    def execute_experiment_process(self, parameters, dataset):
        """Execute the experiment process."""
        # get current device, this require device to be set before the experiment
        device = "cuda:" + str(torch.cuda.current_device())
        
        # Load the dataset
        data_name = parameters.dataset_name
        num, dim, data = data_process.generate_data(data_name)
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
        btsp_model = btsp_coding_lsh.btsp_lsh(
            train_data,
            hash_length,
            sampling_ratio,
            embedding_size,
            fq_constant=fq_constant,
            binary_mode=binary_mode,
            device=device,
        )
        btsp_mAP = utils.tesht_map_dist(train_data, btsp_model.hashes)
        btsp_fq = fq_constant * hash_length / embedding_size

        # bstp_msg = "mean average precision of flylsh is equal to {:.4f}".format(
        #     btsp_mAP
        # )
        # print("BTSP-like Hashing: ", bstp_msg)

        self.result_dict["experiment_index"].append(parameters.experiment_index)
        self.result_dict["input_dim"].append(input_dim)
        self.result_dict["embedding_size"].append(embedding_size)
        self.result_dict["btsp_fq_constant"].append(fq_constant)
        self.result_dict["btsp_sampling_ratio"].append(sampling_ratio)
        self.result_dict["btsp_fq"].append(btsp_fq)
        self.result_dict["btsp_mAP"].append(btsp_mAP)

    def summarize_results(self):
        """Summarize the results."""

        # convert the results to pandas dataframe
        self.result_dict = dict(self.result_dict)
        for key in self.result_dict:
            self.result_dict[key] = list(self.result_dict[key])
        
        self.results = pd.DataFrame(self.result_dict)
        # print(self.results)
        
        # save both params and results to csv
        self.results.to_csv(os.path.join(self.experiment_folder, "results.csv"))
        self.params.to_csv(os.path.join(self.experiment_folder, "params.csv"))
        
        # merge the results with params
        self.merged_results = pd.merge(self.params, self.results, on="experiment_index")
        # save the merged results to csv
        self.merged_results.to_csv(os.path.join(self.experiment_folder, "merged_results.csv"))
        return super().summarize_results()


if __name__ == "__main__":
    interface = BTSPOptimalLSHExperiment()
    experiment = auto_experiment.CudaDistributedExperiment(interface, cuda="max")
    experiment.run()
    experiment.evaluate()
    
