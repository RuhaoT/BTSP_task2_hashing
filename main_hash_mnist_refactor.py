"""Refactored experiment code."""
import dataclasses
import os
import sys
import numpy as np
import torch
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
    binary_mode: bool | list[bool] = False
    sampling_ratio: float | list[float] = 0.05
    random_seed: int | list[int] = 1
    experiment_index: int = -1


class LSHmAPExperiment(auto_experiment.ExperimentInterface):
    """This experiment compare LSH algorithms with mAP."""

    def __init__(self) -> None:

        self.meta_params = MetaParams(
            dataset_name=['mnist10k', 'glove10k', 'sift10k', 'gist10k'],
            training_data_num=5000,
            hash_length=[2, 4, 8, 12, 16, 20, 24],
            space_ratio=[1, 4, 8, 12, 20],
            binary_mode=[True, False],
            sampling_ratio=[0.01, 0.05, 0.1, 0.2, 0.3, 0.6],
            random_seed=1,
        )
        # RH: the above parameter would faciliate your rapid validation. We should consider the following setup for publishable results
        # hash_lengths = [2, 4, 8, 12, 16, 20, 24]
        # space_ratio_list = [ 1, 4, 8, 12, 20]
        
        # for debugging
        # self.meta_params = MetaParams(
        #     dataset_name=['mnist10k', 'sift10k', 'gist10k', 'glove10k'],
        #     training_data_num=9000,
        #     hash_length=24,
        #     space_ratio=16,
        #     binary_mode=False,
        #     sampling_ratio=0.2,
        #     random_seed=1,
        # )

        self.experiment_name = "LSHmAPExperiment_debug"
        self.experiment_folder = logging.init_experiment_folder(
            data_folder="./results/", experiment_name=self.experiment_name, timed=False
        )

        self.params: pd.DataFrame = pd.DataFrame()
        self.result_dict: dict = {
            "experiment_index": [],
            "input_dim": [],
            "embedding_size": [],
            "btsp_fq": [],
            "btsp_mAP": [],
            "fly_mAP": [],
            "wta_mAP": [],
            "lsh_mAP": [],
        }
        self.results: pd.DataFrame = pd.DataFrame(
            columns=["experiment_index", "btsp_mAP", "fly_mAP", "wta_mAP", "lsh_mAP"]
        )

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

    def execute_experiment_process(self, parameters: MetaParams, dataset):
        """Execute the experiment process."""
        # Load the dataset
        data_name = parameters.dataset_name
        num, dim, data = data_process.generate_data(data_name)
        path = self.experiment_folder
        train_num = parameters.training_data_num
        nnn = int(train_num * 0.02)

        hash_length = parameters.hash_length
        space_ratio = parameters.space_ratio

        binary_mode = parameters.binary_mode
        sampling_ratio = parameters.sampling_ratio

        norm_data = (data - data.min()) / (data.max() - data.min())
        input_dim = norm_data.shape[1]
        train_data = norm_data[:train_num]
        train_data = torch.Tensor(train_data).to("cuda")

        utils.setup_seed(parameters.random_seed)

        # print("Testing hashing length:", hash_length, "space size", space_ratio)
        embedding_size = int(input_dim * space_ratio)
        btsp_fq = 2 * hash_length / embedding_size
        # print(
        #     "\n Sample ratio of input patterns: ",
        #     sampling_ratio,
        #     "space ratio",
        #     space_ratio,
        # )
        btsp_model = btsp_coding_lsh.btsp_lsh(
            train_data,
            hash_length,
            sampling_ratio,
            embedding_size,
            binary_mode=binary_mode,
            device="cuda",
        )
        btsp_mAP = utils.tesht_map_dist(train_data, btsp_model.hashes)
        # bstp_msg = "mean average precision of flylsh is equal to {:.4f}".format(
        #     btsp_mAP
        # )
        # print("BTSP-like Hashing: ", bstp_msg)
        # the following we compared the performance with advanced hashing algorithmns
        fly_model = flylsh.flylsh(
            train_data,
            hash_length,
            sampling_ratio,
            embedding_size,
            binary_mode=binary_mode,
        )
        fly_mAP = utils.tesht_map_dist(train_data, fly_model.hashes)
        # fly_msg = "mean average precision of flylsh is equal to {:.4f}".format(fly_mAP)
        # print("FLY algo: ", fly_msg)
        WTA_model = wtahash.WTAHash(train_data, hash_length)
        wta_mAP = utils.tesht_map_dist(train_data, WTA_model.hashes)
        # msg = "mean average precision of flylsh is equal to {:.4f}".format(wta_mAP)
        # print("WTA: ", msg)
        lsh_model = lsh.LSH(train_data, hash_length)
        lsh_mAP = utils.tesht_map_dist(train_data, lsh_model.hashes)
        # msg = "mean average precision of dense_model is equal to {:.4f}".format(lsh_mAP)
        # print("LSH: ", msg)
        # sparse_model = btsp_sparse(train_data, hash_length, sampling_ratio, embedding_size,
        #                       binary_mode=binary_mode)
        # sparse_mAP = tesht_map_dist(train_data, sparse_model.hashes)
        # msg = 'mean average precision of sparse_model is equal to {:.4f}'.format(sparse_mAP)
        # print("Sparse: ", msg)

        self.result_dict["experiment_index"].append(parameters.experiment_index)
        self.result_dict["input_dim"].append(input_dim)
        self.result_dict["embedding_size"].append(embedding_size)
        self.result_dict["btsp_fq"].append(btsp_fq)
        self.result_dict["btsp_mAP"].append(btsp_mAP)
        self.result_dict["fly_mAP"].append(fly_mAP)
        self.result_dict["wta_mAP"].append(wta_mAP)
        self.result_dict["lsh_mAP"].append(lsh_mAP)

    def summarize_results(self):
        """Summarize the results."""

        # for debugging
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
    # set torch device to cuda if available
    torch.set_default_device('cuda')
    with torch.device("cuda"):
        experiment = auto_experiment.SimpleBatchExperiment(LSHmAPExperiment(), 1)
        experiment.run()
        experiment.evaluate()
