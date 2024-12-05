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

>> Compatibility shells between the TACTiS models and the GluonTS and PyTorchTS libraries.
"""

from typing import Any, Dict

import torch
from torch import nn


from .stgnf import STGNF

class TrainingNetwork(nn.Module):
    """
    A shell on top of the TACTiS module, to be used during training only.
    """

    def __init__(
        self,
        num_series: int,
        model_parameters: Dict[str, Any],
    ):
        """
        Parameters:
        -----------
        num_series: int
            Number of series of the data which will be sent to the model.
        model_parameters: Dict[str, Any]
            The parameters of the underlying TACTiS model, as a dictionary.
        """
        super().__init__()

        self.model = STGNF(num_series, **model_parameters)



    def forward(
        self,
        past_target_norm: torch.Tensor,
        future_target_norm: torch.Tensor,
        past_mask:torch.Tensor,
        future_mask:torch.Tensor,
        past_time_feat:torch.Tensor,
        future_time_feat:torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters:
        -----------
        past_target_norm: torch.Tensor [batch, time steps, series]
            The historical data that will be available at inference time.
        future_target_norm: torch.Tensor [batch, time steps, series]
            The data to be forecasted at inference time.

        Returns:
        --------
        loss: torch.Tensor []
            The loss function, averaged over all batches.
        """
        # The data coming from Gluon is not in the shape we use in the model, so transpose it.


        b,t1,n=past_target_norm.shape
        b,t2,n=future_target_norm.shape
        v=3
        location=n//v


        hist_value = past_target_norm.transpose(1, 2).reshape(-1,location,v,t1)  #(B,n,v,T1)


        pred_value = future_target_norm.transpose(1, 2).reshape(-1,location,v,t2)  #(B,N,v,T2)


        # For the time steps, we take for granted that the data is aligned with a constant frequency

        hist_time=past_time_feat.float()

        pred_time=future_time_feat.float()


        return self.model.loss(hist_time=hist_time, hist_value=hist_value, pred_time=pred_time, pred_value=pred_value)



class PredictionNetwork(nn.Module):
    """
    A shell on top of the TACTiS module, to be used during inference only.
    """

    def __init__(
        self,
        num_series: int,
        model_parameters: Dict[str, Any],
        prediction_length: int,
        num_parallel_samples: int,
    ):
        """
        Parameters:
        -----------
        num_series: int
            Number of series of the data which will be sent to the model.
        model_parameters: Dict[str, Any]
            The parameters of the underlying TACTiS model, as a dictionary.
        """
        super().__init__()

        self.model = STGNF(num_series, **model_parameters)

        self.num_parallel_samples = num_parallel_samples
        self.prediction_length = prediction_length

    def forward(
        self,
        past_target_norm: torch.Tensor,
        past_mask: torch.Tensor,
        past_time_feat: torch.Tensor,
        future_time_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters:
        -----------
        past_target_norm: torch.Tensor [batch, time steps, series]
            The historical data that are available.

        Returns:
        --------
        samples: torch.Tensor [samples, batch, time steps, series]
            Samples from the forecasted distribution.
        """
        # The data coming from Gluon is not in the shape we use in the model, so transpose it.

        b,t1,n=past_target_norm.shape
        _,t2,_=future_time_feat.shape
        v = 3
        location = n // v
        hist_value = past_target_norm.transpose(1, 2).reshape(-1,location,v,t1)  #(b,n,v,t)



        hist_time = past_time_feat.float()  #(b,t1,c)
        pred_time = future_time_feat.float() #(b.t2,c)

        samples= self.model.sample(
            num_samples=self.num_parallel_samples, hist_time=hist_time, hist_value=hist_value, pred_time=pred_time,
        )  #(b,nv,t2,sample)

        print(samples.shape)

        return samples.permute(0,3,2,1)
