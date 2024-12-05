

from typing import Any, Dict, Optional, Tuple

import numpy
import torch
from torch import nn


from .decoder_ntv3 import Decoder
from collections import OrderedDict
import torch.nn.functional as F

from .encoder_1 import SelfAttentionLayer as SelfAttentionLayer_emb

class PositionalEncoding(nn.Module):
    """
    A class implementing the positional encoding for Transformers described in Vaswani et al. (2017).
    Somewhat generalized to allow unaligned or unordered time steps, as long as the time steps are integers.

    Adapted from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, embedding_dim: int, dropout: float = 0.1, max_length: int = 5000):
        """
        Parameters:
        -----------
        embedding_dim: int
            The dimension of the input and output embeddings for this encoding.
        dropout: float, default to 0.1
            Dropout parameter for this encoding.
        max_length: int, default to 5000
            The maximum time steps difference which will have to be handled by this encoding.
        """
        super().__init__()

        assert embedding_dim % 2 == 0, "PositionEncoding needs an even embedding dimension"

        self.dropout = nn.Dropout(p=dropout)

        pos_encoding = torch.zeros(max_length, embedding_dim)
        possible_pos = torch.arange(0, max_length, dtype=torch.float)[:, None]
        factor = torch.exp(torch.arange(0, embedding_dim, 2, dtype=torch.float) * (-numpy.log(10000.0) / embedding_dim))

        # Alternate between using sine and cosine
        pos_encoding[:, 0::2] = torch.sin(possible_pos * factor)
        pos_encoding[:, 1::2] = torch.cos(possible_pos * factor)

        # Register as a buffer, to automatically be sent to another device if the model is sent there
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, input_encoded: torch.Tensor, timesteps: torch.IntTensor) -> torch.Tensor:
        """
        Parameters:
        -----------
        input_encoded: torch.Tensor [batch, series, time steps, embedding dimension]
            An embedding which will be modified by the position encoding.
        timesteps: torch.IntTensor [batch, series, time steps] or [batch, 1, time steps]
            The time step for each entry in the input.

        Returns:
        --------
        output_encoded: torch.Tensor [batch, series, time steps, embedding dimension]
            The modified embedding.
        """
        # Use the time difference between the first time step of each batch and the other time steps.
        # min returns two outputs, we only keep the first.
        min_t = timesteps.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
        delta_t = timesteps - min_t

        output_encoded = input_encoded + self.pos_encoding[delta_t]
        return self.dropout(output_encoded)


class TimeEncoding(nn.Module):
    """
    A class implementing the positional encoding for Transformers described in Vaswani et al. (2017).
    Somewhat generalized to allow unaligned or unordered time steps, as long as the time steps are integers.

    Adapted from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, input_embedding: int, embedding_dim: int, dropout: float = 0.1):
        """
        Parameters:
        -----------
        embedding_dim: int
            The dimension of the input and output embeddings for this encoding.
        dropout: float, default to 0.1
            Dropout parameter for this encoding.
        max_length: int, default to 5000
            The maximum time steps difference which will have to be handled by this encoding.
        """
        super().__init__()

        assert embedding_dim % 2 == 0, "PositionEncoding needs an even embedding dimension"

        self.dropout = nn.Dropout(p=dropout)
        self.time_embedd = nn.Sequential(OrderedDict([
            ('embd', nn.Linear(input_embedding, embedding_dim, bias=True)),
            ('relu1', nn.ReLU()),
            ('fc', nn.Linear(embedding_dim, embedding_dim, bias=True)),
            ('relu2', nn.ReLU()),
        ]))

    def forward(self, input_encoded: torch.Tensor, timesteps: torch.IntTensor) -> torch.Tensor:
        """
        Parameters:
        -----------
        input_encoded: torch.Tensor [batch, series, time steps, embedding dimension]
            An embedding which will be modified by the position encoding.
        timesteps: torch.IntTensor [batch, series, time steps] or [batch, 1, time steps]
            The time step for each entry in the input.

        Returns:
        --------
        output_encoded: torch.Tensor [batch, series, time steps, embedding dimension]
            The modified embedding.
        """

        time_feat = self.time_embedd(timesteps)
        # print(time_feat.shape)
        output_encoded = input_encoded + time_feat
        return self.dropout(output_encoded)


class NormalizationIdentity:
    """
    Trivial normalization helper. Do nothing to its data.
    """

    def __init__(self, hist_value: torch.Tensor):
        """
        Parameters:
        -----------
        hist_value: torch.Tensor [batch, series, time steps]
            Historical data which can be used in the normalization.
        """
        pass

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        """
        Normalize the given values according to the historical data sent in the constructor.

        Parameters:
        -----------
        value: Tensor [batch, series, time steps]
            A tensor containing the values to be normalized.

        Returns:
        --------
        norm_value: Tensor [batch, series, time steps]
            The normalized values.
        """
        return value

    def denormalize(self, norm_value: torch.Tensor) -> torch.Tensor:
        """
        Undo the normalization done in the normalize() function.

        Parameters:
        -----------
        norm_value: Tensor [batch, series, time steps, samples]
            A tensor containing the normalized values to be denormalized.

        Returns:
        --------
        value: Tensor [batch, series, time steps, samples]
            The denormalized values.
        """
        return norm_value


class NormalizationStandardization:
    """
    Normalization helper for the standardization.

    The data for each batch and each series will be normalized by:
    - substracting the historical data mean,
    - and dividing by the historical data standard deviation.

    Use a lower bound of 1e-8 for the standard deviation to avoid numerical problems.
    """

    def __init__(self, hist_value: torch.Tensor):
        """
        Parameters:
        -----------
        hist_value: torch.Tensor [batch, series, time steps]
            Historical data which can be used in the normalization.
        """

        std, mean = torch.std_mean(hist_value, dim=2, keepdim=True)
        self.std = std.clamp(min=1e-8)
        self.mean = mean


    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        """
        Normalize the given values according to the historical data sent in the constructor.
        Parameters:
        -----------
        value: Tensor [batch, series, time steps]
            A tensor containing the values to be normalized.
        Returns:
        --------
        norm_value: Tensor [batch, series, time steps]
            The normalized values.
        """

        value = (value - self.mean) / self.std

        return value

    def denormalize(self, norm_value: torch.Tensor) -> torch.Tensor:
        """
        Undo the normalization done in the normalize() function.

        Parameters:
        -----------
        norm_value: Tensor [batch, series, time steps, samples]
            A tensor containing the normalized values to be denormalized.

        Returns:
        --------
        value: Tensor [batch, series, time steps, samples]
            The denormalized values.
        """

        norm_value = (norm_value * self.std[:, :, :, None]) + self.mean[:, :, :, None]
        return norm_value



class NormalizationStandardizationall:
    """
    Normalization helper for the standardization.

    The data for each batch and each series will be normalized by:
    - substracting the historical data mean,
    - and dividing by the historical data standard deviation.

    Use a lower bound of 1e-8 for the standard deviation to avoid numerical problems.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        """
        Parameters:
        -----------
        hist_value: torch.Tensor [batch, series, time steps]
            Historical data which can be used in the normalization.
        """

        self.mean = mean.float()[None, :, :, None]  # ()
        self.std = std.float()[None, :, :, None]

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        """
        Normalize the given values according to the historical data sent in the constructor.
        Parameters:
        -----------
        value: Tensor [batch, series, time steps]
            A tensor containing the values to be normalized.
        Returns:
        --------
        norm_value: Tensor [batch, series, time steps]
            The normalized values.
        """

        value = (value - self.mean) / self.std


        return value

    def denormalize(self, norm_value: torch.Tensor) -> torch.Tensor:
        """
        Undo the normalization done in the normalize() function.

        Parameters:
        -----------
        norm_value: Tensor [batch, series, time steps, samples]
            A tensor containing the normalized values to be denormalized.

        Returns:
        --------
        value: Tensor [batch, series, time steps, samples]
            The denormalized values.
        """

        norm_value = (norm_value * self.std[:, :, :, :, None]) + self.mean[:, :, :, :, None]
        return norm_value


class STGNF(nn.Module):


    def __init__(
            self,
            num_series: int,
            num_variables: int,
            series_embedding_dim: int,
            input_encoder_layers: int,
            bagging_size: Optional[int] = None,
            input_encoding_normalization: bool = True,
            data_normalization: str = "none",
            loss_normalization: str = "series",
            mean: torch.Tensor = None,
            std: torch.Tensor = None,
            data_node: torch.Tensor = None,
            time_num:int=None,
            positional_encoding: Optional[Dict[str, Any]] = None,
            time_encoding: Optional[Dict[str, Any]] = None,
            encoder: Optional[Dict[str, Any]] = None,
            temporal_encoder: Optional[Dict[str, Any]] = None,
            copula_decoder: Optional[Dict[str, Any]] = None,
            gaussian_decoder: Optional[Dict[str, Any]] = None,
            gaussian_dist: Optional[Dict[str, Any]] = None,
    ):
        """
        Parameters:
        -----------
        num_series: int  N个节点
            Number of series of the data which will be sent to the model.
        series_embedding_dim: int
            The dimensionality of the per-series embedding.
        input_encoder_layers: int
            Number of layers in the MLP which encodes the input data.
        bagging_size: Optional[int], default to None
            If set, the loss() method will only consider a random subset of the series at each call.
            The number of series kept is the value of this parameter.
        input_encoding_normalization: bool, default to True
            If true, the encoded input values (prior to the positional encoding) are scaled
            by the square root of their dimensionality.
        data_normalization: str ["", "none", "standardization"], default to "series"
            How to normalize the input values before sending them to the model.
        loss_normalization: str ["", "none", "series", "timesteps", "both"], default to "series"
            Scale the loss function by the number of series, timesteps, or both.
        positional_encoding: Optional[Dict[str, Any]], default to None
            If set to a non-None value, uses a PositionalEncoding for the time encoding.
            The options sent to the PositionalEncoding is content of this dictionary.
        encoder: Optional[Dict[str, Any]], default to None
            If set to a non-None value, uses a Encoder as the encoder.
            The options sent to the Encoder is content of this dictionary.
        temporal_encoder: Optional[Dict[str, Any]], default to None
            If set to a non-None value, uses a TemporalEncoder as the encoder.
            The options sent to the TemporalEncoder is content of this dictionary.
        copula_decoder: Optional[Dict[str, Any]], default to None
            If set to a non-None value, uses a CopulaDecoder as the decoder.
            The options sent to the CopulaDecoder is content of this dictionary.
        gaussian_decoder: Optional[Dict[str, Any]], default to None
            If set to a non-None value, uses a GaussianDecoder as the decoder.
            The options sent to the GaussianDecoder is content of this dictionary.
        """
        super().__init__()


        data_normalization = data_normalization.lower()

        loss_normalization = loss_normalization.lower()

        self.num_series = num_series
        self.bagging_size = bagging_size
        self.series_embedding_dim = series_embedding_dim
        self.input_encoder_layers = input_encoder_layers
        self.input_encoding_normalization = input_encoding_normalization
        self.loss_normalization = loss_normalization

        self.data_normalization = {
            "": NormalizationIdentity,
            "none": NormalizationIdentity,
            "standardization": NormalizationStandardization,
            "standardizationall": NormalizationStandardizationall(mean, std),
        }[data_normalization]
        self.normalization = data_normalization



        if copula_decoder is not None:
            self.decoder = Decoder(input_dim=self.series_embedding_dim*5, **copula_decoder)


        self.input_proj = nn.Linear(1, self.series_embedding_dim*5)  #1->s
        self.tod_embedding=nn.Embedding(288, self.series_embedding_dim*5)
        self.dow_embedding = nn.Embedding(7, self.series_embedding_dim*5)
        self.time_embedding = nn.Embedding(time_num, self.series_embedding_dim*5)

        self.variable_encoder = nn.Embedding(num_embeddings=num_variables, embedding_dim=self.series_embedding_dim*5)

        self.series_encoder = nn.Parameter(data=data_node, requires_grad=True)

        self.num_layer=temporal_encoder['attention_layers']
        print(self.num_layer)
        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer_emb(model_dim=self.series_embedding_dim*5, feed_forward_dim=256, num_heads=4, dropout=0.1)
                for _ in range(self.num_layer)
            ]
        )

        self.attn_layers_s = nn.ModuleList(
            [
                SelfAttentionLayer_emb(model_dim=self.series_embedding_dim*5, feed_forward_dim=256, num_heads=4, dropout=0.1)
                for _ in range(self.num_layer)
            ]

        )
        self.attn_layers_v = nn.ModuleList(
            [
                SelfAttentionLayer_emb(model_dim=self.series_embedding_dim*5, feed_forward_dim=256, num_heads=4, dropout=0.1)
                for _ in range(self.num_layer)
            ]
        )


    def loss(
            self, hist_time: torch.Tensor, hist_value: torch.Tensor, pred_time: torch.Tensor, pred_value: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the loss function of the model.

        Parameters:
        -----------
        hist_time: Tensor [batch, series, time steps] or [batch, 1, time steps] or [batch, time steps]
            A tensor containing the time steps associated with the values of hist_value.
            If the series dimension is singleton or missing, then the time steps are taken as constant across all series.
        hist_value: Tensor [batch, series, time steps]   #[B,T1]  [0---T1]
            A tensor containing the values that will be available at inference time.
        pred_time: Tensor [batch, series, time steps] or [batch, 1, time steps] or [batch, time steps]
            A tensor containing the time steps associated with the values of pred_value.
            If the series dimension is singleton or missing, then the time steps are taken as constant across all series.
        pred_value: Tensor [batch, series, time steps]  #(B,T1-T2)B的每个T1-T2都一样 [T1---T2]
            A tensor containing the values that the model should learn to forecast at inference time.

        Returns:
        --------
        loss: torch.Tensor []
            The loss function of TACTiS, with lower values being better. Averaged over batches.
        """

        num_batches = hist_value.shape[0]  # b
        num_series = hist_value.shape[1]  # n
        num_variables = hist_value.shape[2]  # v
        num_hist_timesteps = hist_value.shape[3]  # t1
        num_pred_timesteps = pred_value.shape[3]  # t2
        device = hist_value.device




        if self.normalization == "standardizationall":
            normalizer = self.data_normalization
        else:
            normalizer = self.data_normalization(hist_value)

        hist_value = normalizer.normalize(hist_value)  #
        pred_value = normalizer.normalize(pred_value)  # (b,n,v,t2)


        #全连接
        hist_value_features=self.input_proj(hist_value[...,None])  #(b,n,v,t1,c)
        pred_value_features=self.input_proj(torch.zeros_like(pred_value[...,None],device=device))

        #时间初始化
        tod_emb_hist = self.tod_embedding( hist_time[:,:,3].long() )  # (batch_size, in_steps, tod_embedding_dim)
        tod_emb_hist=tod_emb_hist[:,None,None,:,:].expand(-1,num_series,num_variables,-1,-1)  #(b,n,v,t1,c)

        tod_emb_hist1 = self.time_embedding(hist_time[:, :, 6].long())  # (batch_size, in_steps, tod_embedding_dim)
        tod_emb_hist1 = tod_emb_hist1[:, None, None, :, :].expand(-1, num_series, num_variables, -1, -1)  # (b,n,v,t1,c)

        dow_emb_hist = self.dow_embedding(hist_time[:,:,0].long())  # (batch_size, in_steps,  dow_embedding_dim)
        dow_emb_hist = dow_emb_hist[:, None, None, :, :].expand(-1, num_series, num_variables, -1, -1)  # (b,n,v,t1,c)

        variable_emb = self.variable_encoder(torch.arange(num_variables, device=device)) #(v,c)
        series_emb= self.series_encoder

        tod_emb_pred = self.tod_embedding(pred_time[:, :, 3].long())  # (batch_size, in_steps, tod_embedding_dim)
        tod_emb_pred = tod_emb_pred[:, None, None, :, :].expand(-1, num_series, num_variables, -1, -1)  # (b,n,v,t2,c)

        tod_emb_pred1 = self.time_embedding(pred_time[:, :, 6].long())  # (batch_size, in_steps, tod_embedding_dim)
        tod_emb_pred1 = tod_emb_pred1[:, None, None, :, :].expand(-1, num_series, num_variables, -1, -1)  # (b,n,v,t2,c)

        dow_emb_pred = self.dow_embedding(pred_time[:, :, 0].long())  # (batch_size, in_steps,  dow_embedding_dim)
        dow_emb_pred = dow_emb_pred[:, None, None, :, :].expand(-1, num_series, num_variables, -1, -1)  # (b,n,v,t2,c)



        hist_encoded = hist_value_features + tod_emb_hist + tod_emb_hist1 + dow_emb_hist + \
                       series_emb[None, :, None, None, :].expand(num_batches, -1, num_variables, num_hist_timesteps,-1) + \
                       variable_emb[None, None, :, None, :].expand(num_batches, num_series, -1, num_hist_timesteps, -1)


        pred_encoded = tod_emb_pred +tod_emb_pred1+ dow_emb_pred +\
                       series_emb[None, :, None, None, :].expand(num_batches, -1, num_variables, num_pred_timesteps,-1) + \
                       variable_emb[None, None, :, None, :].expand(num_batches, num_series, -1, num_pred_timesteps, -1)

        data=torch.cat([hist_encoded,pred_encoded],dim=3) #(b,n,v,t1+t2,c)


        time_hist=tod_emb_hist+tod_emb_hist1+dow_emb_hist  #(b,n,v,t1,c)
        time_pred=tod_emb_pred+tod_emb_pred1+dow_emb_pred  #(b,n,v,t2,c)
        time=torch.cat([time_hist,time_pred],dim=3) #(b,n,v,t1+t2,c)
        time=time[None,...].expand(4,-1,-1,-1,-1,-1).reshape(4*num_batches,num_series,num_variables,num_hist_timesteps+num_pred_timesteps,-1) #(4b,n,v,t,c)


        for i in range(self.num_layer):

            data = self.attn_layers_t[i](data, dim=3,attn_embedding=time)
            data = self.attn_layers_s[i](data, dim=1, attn_embedding=series_emb)
            data = self.attn_layers_v[i](data, dim=2, attn_embedding=variable_emb)

        encoded = data.reshape(num_batches, num_series * num_variables, num_hist_timesteps + num_pred_timesteps, -1)

        true_value = torch.cat(
            [
                hist_value.reshape(num_batches, num_series * num_variables, num_hist_timesteps),
                pred_value.reshape(num_batches, num_series * num_variables, num_pred_timesteps),
            ],
            dim=2,
        )  # (b,nv,t1+t2 真实值)

        loss = self.decoder.loss(encoded, true_value)

        loss = loss.reshape(num_batches, num_series).sum(1)

        if self.loss_normalization in {"series", "both"}:
            loss = loss / num_series
        if self.loss_normalization in {"timesteps", "both"}:
            loss = loss / num_pred_timesteps

        return loss.mean()

    def sample(
            self, num_samples: int, hist_time: torch.Tensor, hist_value: torch.Tensor, pred_time: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate the given number of samples from the forecasted distribution.

        Parameters:
        -----------
        num_samples: int
            How many samples to generate, must be >= 1.
        hist_time: Tensor [batch, series, time steps] or [batch, 1, time steps] or [batch, time steps]
            A tensor containing the times associated with the values of hist_value.
            If the series dimension is singleton or missing, then the time steps are taken as constant across all series.
        hist_value: Tensor [batch, series, time steps]
            A tensor containing the available values
        pred_time: Tensor [batch, series, time steps] or [batch, 1, time steps] or [batch, time steps]
            A tensor containing the times at which we want forecasts.
            If the series dimension is singleton or missing, then the time steps are taken as constant across all series.

        Returns:
        --------
        samples: torch.Tensor [batch, series, time steps, samples]
            Samples from the forecasted distribution.
        """
        num_batches = hist_value.shape[0]
        num_series = hist_value.shape[1]
        num_variables = hist_value.shape[2]

        num_hist_timesteps = hist_value.shape[3]
        num_pred_timesteps = pred_time.shape[1]
        device = hist_value.device


        # 归一化数据
        if self.normalization == "standardizationall":
            normalizer = self.data_normalization
        else:
            normalizer = self.data_normalization(hist_value)
        # normalizer=NormalizationStandardization(hist_value)
        hist_value = normalizer.normalize(hist_value)  #


        hist_value_features = self.input_proj(hist_value[..., None])  # (b,n,v,t1,c)
        pred_value_features = self.input_proj(
            torch.zeros((num_batches,num_series,num_variables,num_pred_timesteps,1), device=device))  #(b,n,v,t2,c)

        #时间
        tod_emb_hist = self.tod_embedding(hist_time[:, :, 3].long())  # (batch_size, in_steps, tod_embedding_dim)
        tod_emb_hist = tod_emb_hist[:, None, None, :, :].expand(-1, num_series, num_variables, -1, -1)  # (b,n,v,t1,c)

        tod_emb_hist1 = self.time_embedding(hist_time[:, :, 6].long())  # (batch_size, in_steps, tod_embedding_dim)
        tod_emb_hist1 = tod_emb_hist1[:, None, None, :, :].expand(-1, num_series, num_variables, -1, -1)  # (b,n,v,t1,c)

        dow_emb_hist = self.dow_embedding(hist_time[:, :, 0].long())  # (batch_size, in_steps,  dow_embedding_dim)
        dow_emb_hist = dow_emb_hist[:, None, None, :, :].expand(-1, num_series, num_variables, -1, -1)  # (b,n,v,t1,c)

        variable_emb = self.variable_encoder(torch.arange(num_variables, device=device))  # (n,c)
        series_emb=self.series_encoder

        tod_emb_pred = self.tod_embedding(pred_time[:, :, 3].long())  # (batch_size, in_steps, tod_embedding_dim)
        tod_emb_pred = tod_emb_pred[:, None, None, :, :].expand(-1, num_series, num_variables, -1, -1)  # (b,n,v,t2,c)

        tod_emb_pred1 = self.time_embedding(pred_time[:, :, 6].long())  # (batch_size, in_steps, tod_embedding_dim)
        tod_emb_pred1 = tod_emb_pred1[:, None, None, :, :].expand(-1, num_series, num_variables, -1, -1)  # (b,n,v,t2,c)

        dow_emb_pred = self.dow_embedding(pred_time[:, :, 0].long())  # (batch_size, in_steps,  dow_embedding_dim)
        dow_emb_pred = dow_emb_pred[:, None, None, :, :].expand(-1, num_series, num_variables, -1, -1)  # (b,n,v,t2,c)


        hist_encoded = hist_value_features + tod_emb_hist + tod_emb_hist1 + dow_emb_hist + \
                       series_emb[None, :, None, None, :].expand(num_batches, -1, num_variables, num_hist_timesteps,
                                                                 -1) + \
                       variable_emb[None, None, :, None, :].expand(num_batches, num_series, -1, num_hist_timesteps, -1)


        pred_encoded = tod_emb_pred + tod_emb_pred1 + dow_emb_pred + \
                       series_emb[None, :, None, None, :].expand(num_batches, -1, num_variables, num_pred_timesteps,
                                                                 -1) + \
                       variable_emb[None, None, :, None, :].expand(num_batches, num_series, -1, num_pred_timesteps, -1)

        data=torch.cat([hist_encoded,pred_encoded],dim=3) #(b,n,v,t1+t2,c)

        time_hist = tod_emb_hist+tod_emb_hist1+dow_emb_hist  # (b,n,v,t1,c)
        time_pred = tod_emb_pred+ tod_emb_pred1+dow_emb_pred  # (b,n,v,t2,c)
        time=torch.cat([time_hist,time_pred],dim=3) #(b,n,v,t1+t2,c)
        time=time[None,...].expand(4,-1,-1,-1,-1,-1).reshape(4*num_batches,num_series,num_variables,num_hist_timesteps+num_pred_timesteps,-1) #(4b,n,v,t1+t2,c)



        for  i in range(self.num_layer):
            data = self.attn_layers_t[i](data, dim=3,attn_embedding=time)
            data = self.attn_layers_s[i](data, dim=1,attn_embedding=series_emb)
            data = self.attn_layers_v[i](data, dim=2,attn_embedding=variable_emb)

        encoded = data.reshape(num_batches, num_series * num_variables, num_hist_timesteps + num_pred_timesteps, -1)

        true_value = torch.cat(
            [
                hist_value.reshape(num_batches, num_series * num_variables, num_hist_timesteps),
                torch.zeros(num_batches, num_series * num_variables, num_pred_timesteps, device=device),
            ],
            dim=2,
        )

        samples = self.decoder.sample(num_samples, encoded[:, :, :num_hist_timesteps, :],
                                      true_value[:, :, :num_hist_timesteps],
                                      encoded[:, :, num_hist_timesteps:, :])  # (b,nv,t2,s)
        samples = samples.reshape(num_batches, num_series, num_variables, num_pred_timesteps, num_samples)
        samples = normalizer.denormalize(samples)  # (b,n,v,t2,s)
        samples = samples.reshape(num_batches, num_series * num_variables, num_pred_timesteps, num_samples)

        # print(samples[:, -66:-65, 10:20, :3])
        # print(samples[:, -66:-65, 36:46, :3])
        return samples  # (b,nv,t2,s)
