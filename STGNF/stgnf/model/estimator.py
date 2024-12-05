"""
Copyright 2022 ServiceNow
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

>> The estimator object for TACTiS, which is used with GluonTS and PyTorchTS.
"""

from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from gluonts.dataset.field_names import FieldName
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.torch.util import copy_parameters
from gluonts.transform import (
    AddObservedValuesIndicator,
    CDFtoGaussianTransform,
    Chain,
    InstanceSampler,
    InstanceSplitter,
    RenameFields,
    TestSplitSampler,
    Transformation,
    cdf_to_gaussian_forward_transform,
    SimpleTransformation,
    VstackFeatures,
)
from gluonts.dataset.common import Dataset
from typing import NamedTuple, Optional
from pts import Trainer
from pts.model import PyTorchEstimator
from pts.model.utils import get_module_forward_input_names
import sys
sys.path.append('../')
sys.path.append('../../preprocess')
from .network1 import PredictionNetwork, TrainingNetwork


class TrainOutput(NamedTuple):
    transformation: Transformation
    trained_net: nn.Module
    predictor: PyTorchPredictor
    prediction_model: nn.Module

class SingleInstanceSampler(InstanceSampler):
    """
    Randomly pick a single valid window in the given time series.
    This fix the bias in ExpectedNumInstanceSampler which leads to varying sampling frequency
    of time series of unequal length, not only based on their length, but when they were sampled.
    """

    def __call__(self, ts: np.ndarray) -> np.ndarray:
        a, b = self._get_bounds(ts)
        window_size = b - a + 1   #可取值范围

        if window_size <= 0:
            return np.array([], dtype=int)

        indices = np.random.randint(window_size, size=1)
        return indices + a  #随机一个可以开始的点


class STGNFEstimator(PyTorchEstimator):
    """
    The compatibility layer between TACTiS and GluonTS / PyTorchTS.
    """

    def __init__(
        self,
        model_parameters: Dict[str, Any],
        num_series: int,
        history_length: int,
        prediction_length: int,
        freq: str,
        trainer: Trainer,
        use_feat_dynamic_real: bool = False,
        cdf_normalization: bool = False,
        num_parallel_samples: int = 1,
    ):
        """
        A PytorchTS wrapper for TACTiS

        Parameters:
        -----------
        model_parameters: Dict[str, Any]
            The parameters that will be sent to the TACTiS model.
        num_series: int
            The number of series in the multivariate data.
        history_length: int
            How many time steps will be sent to the model as observed.
        prediction_length: int
            How many time steps will be sent to the model as unobserved, to be predicted.
        freq: str
            The frequency of the series to be forecasted.
        trainer: Trainer
            A Pytorch-TS trainer object
        cdf_normalization: bool, default to False
            If set to True, then the data will be transformed using an estimated CDF from the
            historical data points, followed by the inverse CDF of a Normal(0, 1) distribution.
            Should not be used concurrently with the standardization normalization option in TACTiS.
        num_parallel_samples: int, default to 1
            How many samples to draw at the same time during forecast.
        """
        super().__init__(trainer=trainer)

        self.model_parameters = model_parameters

        self.num_series = num_series
        self.history_length = history_length
        self.prediction_length = prediction_length
        self.freq = freq

        self.cdf_normalization = cdf_normalization
        self.num_parallel_samples = num_parallel_samples
        self.use_feat_dynamic_real=use_feat_dynamic_real
        print("history->prediction:"+str(self.history_length)+"->"+str(self.prediction_length))

    def create_training_network(self, device: torch.device) -> nn.Module:
        """
        Create the encapsulated TACTiS model which can be used for training.

        Parameters:
        -----------
        device: torch.device
            The device where the model parameters should be placed.

        Returns:
        --------
        model: nn.Module
            An instance of TACTiSTrainingNetwork.
        """
        print("create_training_network")
        network =TrainingNetwork(
            num_series=self.num_series,
            model_parameters=self.model_parameters,
        )

        # # import torch.nn as nn
        # # if torch.cuda.device_count() > 1:
        # #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        # #     # 就这一行
        # #     import torch.nn as nn
        # #     network = nn.DataParallel(network)
        #
        # if torch.cuda.is_available():
        #     if torch.cuda.device_count() > 1:
        #         print("Let's use", torch.cuda.device_count(), "GPUs!")
        #         #available_gpus = torch.cuda.device('cuda:0,1,3,5')
        #         #cuda_ids='cuda:0,1,3,5'
        #         import torch.nn as nn
        #         #network = nn.DataParallel(network)
        #
        #         #network=network.cuda(cuda_ids)
        #         local_rank = torch.distributed.get_rank()
        #         network=torch.nn.parallel.DistributedDataParallel(network,device_ids=int(0),find_unused_parameters=True)
        #     else:
        #         network=torch.nn.DataParallel(network)
        #         #cudnn.bemchmark=True
        #         network=network.cuda()

        #print("load dict{}".format( '../reports/pems04/tactis/2023-12-03_22 41 27/checkpoint/model.best.pth'))
        #network.load_state_dict(torch.load('../reports/pems04/tactis/2023-12-03_22 41 27/checkpoint/model.best.pth'))

        network=network.to(device=device)
        # return TACTiSTrainingNetwork(
        #     num_series=self.num_series,
        #     model_parameters=self.model_parameters,
        # ).to(device=device)

        print('*****************Model Parameter*****************')
        for name, param in network.named_parameters():
            print(name, param.shape, param.requires_grad)
        total_num = sum([param.nelement() for param in network.parameters()])
        print('Total params num: {}'.format(total_num))
        print('*****************Finish Parameter****************')


        return network

    def create_instance_splitter(self, mode: str) -> Transformation:
        """
        Create and return the instance splitter needed for training, validation or testing.

        Parameters:
        -----------
        mode: str, "training", "validation", or "test"
            Whether to split the data for training, validation, or test (forecast)

        Returns
        -------
        Transformation
            The InstanceSplitter that will be applied entry-wise to datasets,
            at training, validation and inference time based on mode.
        """
        assert mode in ["training", "validation", "test"]

        if mode == "training":
            print("traininginstance")
            instance_sampler = SingleInstanceSampler( #一共有24*3-24=48个实例
                min_past=self.history_length,   # Will not pick incomplete sequences 24*3
                min_future=self.prediction_length,  #24
            )
            #print(instance_sampler)
        elif mode == "validation":
            instance_sampler = SingleInstanceSampler(
                min_past=self.history_length,  # Will not pick incomplete sequences
                min_future=self.prediction_length,
            )
        elif mode == "test":
            # This splitter takes the last valid window from each multivariate series,
            # so any multi-window split must be done in the data definition.
            instance_sampler = TestSplitSampler()   #每个预测数据 最后的时间点 (b)

        if self.cdf_normalization:  #False
            normalize_transform = CDFtoGaussianTransform(
                cdf_suffix="_norm",
                target_field=FieldName.TARGET,
                target_dim=self.num_series,
                max_context_length=self.history_length,
                observed_values_field=FieldName.OBSERVED_VALUES,
            )
        else:
            normalize_transform = RenameFields(
                {
                    f"past_{FieldName.TARGET}": f"past_{FieldName.TARGET}_norm",
                    f"future_{FieldName.TARGET}": f"future_{FieldName.TARGET}_norm",
                    f"past_{FieldName.FEAT_DYNAMIC_REAL}": f"past_mask",
                    f"future_{FieldName.FEAT_DYNAMIC_REAL}": f"future_mask",
                    f"past_{FieldName.FEAT_DYNAMIC_CAT}": f"past_time_feat",
                    f"future_{FieldName.FEAT_DYNAMIC_CAT}": f"future_time_feat",

                }
            )

        return (
            InstanceSplitter(
                target_field=FieldName.TARGET,  #"target"
                is_pad_field=FieldName.IS_PAD,
                start_field=FieldName.START,
                forecast_start_field=FieldName.FORECAST_START,
                instance_sampler=instance_sampler,
                past_length=self.history_length,
                future_length=self.prediction_length,
                #time_series_fields=[FieldName.OBSERVED_VALUES],#"observed_values"
                time_series_fields=[FieldName.OBSERVED_VALUES ,FieldName.FEAT_DYNAMIC_REAL,FieldName.FEAT_DYNAMIC_CAT],  # "observed_values"
            )
            + normalize_transform
        )

    def create_transformation(self) -> Transformation:
        """
        Add a transformation that replaces NaN in the input data with zeros,
        and mention whether the data was a NaN or not in another field.

        Returns:
        --------
        transformation: Transformation
            The chain of transformations defined for TACTiS.
        """
        return Chain(
            [
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                ),
                # VstackFeatures(
                #     output_field=FieldName.FEAT_TIME,
                #     input_fields= (
                #                      [FieldName.FEAT_DYNAMIC_REAL]
                #                      if self.use_feat_dynamic_real
                #                      else []
                #                  ),
                # ),
            ]
        )

    def create_predictor(
        self, transformation: Transformation, trained_network: nn.Module, device: torch.device
    ) -> PyTorchPredictor:
        """
        Create the predictor which can be used by GluonTS to do inference.

        Parameters:
        -----------
        transformation: Transformation
            The transformation to apply to the data prior to being sent to the model.
        trained_network: nn.Module
            An instance of TACTiSTrainingNetwork with trained parameters.
        device: torch.device
            The device where the model parameters should be placed.

        Returns:
        --------
        predictor: PyTorchPredictor
            The PyTorchTS predictor object.
        """
        prediction_network = PredictionNetwork(
            num_series=self.num_series,
            model_parameters=self.model_parameters,
            prediction_length=self.prediction_length,
            num_parallel_samples=self.num_parallel_samples,
        ).to(device=device)
        #copy_parameters(trained_network, prediction_network)

        print("load dict{}".format(self.trainer.checkpoint_dir + '/model.best.pth'))
        prediction_network.load_state_dict(torch.load(self.trainer.checkpoint_dir + '/model.best.pth'))

        # print("load dict{}".format( '../reports/pems04/tactis/2024-04-22_23 41 43/checkpoint/model.best.pth'))
        # a=torch.load( '../reports/pems04/tactis/2024-04-22_23 41 43/checkpoint/model.best.pth')
        # print(a)
        # prediction_network.load_state_dict(torch.load( '../reports/pems04/tactis/2024-04-22_23 41 43/checkpoint/model.best.pth'))

        output_transform = cdf_to_gaussian_forward_transform if self.cdf_normalization else None  #False
        input_names = get_module_forward_input_names(prediction_network)
        prediction_splitter = self.create_instance_splitter("test")

        return PyTorchPredictor(
            input_transform=transformation + prediction_splitter,  #预测数据 self.create_transformer  预测test分割
            output_transform=output_transform,
            input_names=input_names,
            prediction_net=prediction_network,
            batch_size=self.trainer.batch_size,
            freq=self.freq,
            prediction_length=self.prediction_length,
            device=device,
        )


