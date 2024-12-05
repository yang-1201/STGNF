import math
from typing import Any, Dict, Optional, Tuple, Type

import torch
from torch import nn
import torch.nn.functional as F

from .flows import RealNVP



class Decoder(nn.Module):
    """
    A decoder which forecast using a distribution built from a copula and marginal distributions.
    """

    def __init__(
        self,
        input_dim: int,
        attention_heads: int,
        attention_layers: int,
        attention_dim: int,
        attention_feedforward_dim: int,
        mlp_layers: int,
        mlp_dim: int,
        dropout: float = 0.1,
        n_blocks=6,
        input_size=3,
        n_hidden=4,
        batch_norm=True,
    ):
        """
        Parameters:
        -----------
        input_dim: int
            The dimension of the encoded representation (upstream data encoder).
        min_u: float, default to 0.0
        max_u: float, default to 1.0
            The values sampled from the copula will be scaled from [0, 1] to [min_u, max_u] before being sent to the marginal.
        skip_sampling_marginal: bool, default to False
            If set to True, then the output from the copula will not be transformed using the marginal during sampling.
            Does not impact the other transformations from observed values to the [0, 1] range.
        trivial_copula: Dict[str, Any], default to None
            If set to a non-None value, uses a TrivialCopula.
            The options sent to the TrivialCopula is content of this dictionary.
        attentional_copula: Dict[str, Any], default to None
            If set to a non-None value, uses a AttentionalCopula.
            The options sent to the AttentionalCopula is content of this dictionary.
        dsf_marginal: Dict[str, Any], default to None
            If set to a non-None value, uses a DSFMarginal.
            The options sent to the DSFMarginal is content of this dictionary.
        """
        super().__init__()


        self.input_dim=input_dim
        self.attention_layers = attention_layers
        self.attention_heads = attention_heads
        self.attention_dim = attention_dim
        self.attention_feedforward_dim = attention_feedforward_dim
        self.dropout = dropout
        self.mlp_layers=mlp_layers
        self.mlp_dim=mlp_dim


        # one per layer and per head
        # The key and value creators take the input embedding together with the sampled [0,1] value as an input
        self.dimension_shifting_layer = nn.Linear(self.input_dim, self.attention_heads * self.attention_dim)
        self.key_creators = nn.ModuleList(
                [
                    nn.ModuleList(
                        [
                            nn.Linear((self.input_dim+1)*3, self.attention_dim)
                            for _ in range(self.attention_heads)
                        ]
                    )
                    for _ in range(self.attention_layers)
                ]
        )
        self.value_creators = nn.ModuleList(
                [
                    nn.ModuleList(
                        [

                            nn.Linear((self.input_dim + 1) * 3, self.attention_dim)

                            for _ in range(self.attention_heads)
                        ]
                    )
                    for _ in range(self.attention_layers)
                ]
            )

        # one per layer
        self.attention_dropouts = nn.ModuleList([nn.Dropout(self.dropout) for _ in range(self.attention_layers)])
        self.attention_layer_norms = nn.ModuleList(
                [nn.LayerNorm(self.attention_heads * self.attention_dim) for _ in range(self.attention_layers)]
            )
        self.feed_forwards = nn.ModuleList(
            [
                nn.Sequential(
                        nn.Linear(self.attention_heads * self.attention_dim, self.attention_heads * self.attention_dim),                            nn.ReLU(),
                        nn.Dropout(self.dropout),
                        nn.Linear(self.attention_heads * self.attention_dim, self.attention_heads * self.attention_dim),
                        nn.Dropout(dropout),
                )
                for _ in range(self.attention_layers)
            ]
        )

        self.feed_forward_layer_norms = nn.ModuleList(
                [nn.LayerNorm(self.attention_heads * self.attention_dim) for _ in range(self.attention_layers)]
        )



        self.dimension_shifting_layer = nn.Linear(self.input_dim*3,
                                                  self.attention_dim * self.attention_heads)



        hidden_size = self.attention_dim * self.attention_heads
        cond_label_size = self.attention_dim * self.attention_heads
        self.nf = RealNVP(n_blocks, input_size, hidden_size, n_hidden, cond_label_size=cond_label_size, batch_norm=batch_norm)
        print("t时n独立")




    def loss(self, encoded: torch.Tensor,  true_value: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss function of the decoder.


        Returns:
        --------
        loss: torch.Tensor [batch]
            The loss function, equal to the negative log likelihood of the distribution.
        """
        b,nv,t,c=encoded.shape
        v=3
        n=nv//v
        t2=12
        k=3

        device=encoded.device



        encoded=encoded.reshape(b*n,v,t,c)
        true_value=true_value.reshape(b*n,v,t)

        encoded_value=torch.cat([encoded,true_value[:,:,:,None]],dim=-1)  #(b*n,v,t,c+1)
        encoded_value=encoded_value.permute(0,2,1,3).reshape(b*n,t,v*(c+1)) #(b*n,t,v*(c+1))


        keys = [
            torch.cat([linear(encoded_value)[:, None, :, :] for linear in self.key_creators[layer]], axis=1)
            for layer in range(self.attention_layers)
        ]  #(bn,c,t,v)->(bn,c,t,1)->(bn,h,c,t,1)->(bn,h,t,c,1)->(bn,h,t,c)
        #(bn,t,v*(c+1))->(bn,t,c)->(bn,h,t,c)
        values = [
            torch.cat([linear(encoded_value)[:, None, :, :] for linear in self.value_creators[layer]], axis=1)
            for layer in range(self.attention_layers)
        ]  #l层 (bn,c,t,v)->(bn,c,t,1)->(bn,h,c,t,1)->(bn,h,t,c,1)->(bn,h,t,c)
        #(bn,t,v*(c+1))->(bn,t,c)->(bn,h,t,c)

        product_mask = torch.ones(
            b*n,
            self.attention_heads,
            t2,t,
            device=device,
        )
        product_mask = torch.tril(float("inf") * product_mask).flip((2, 3))  # 保留主对角线和下三角部分，（2，3）维翻转

        att_value=self.dimension_shifting_layer(encoded[:,:,-t2:,:].permute(0,2,1,3).reshape(b*n,t2,-1)) #(bn,v,t2,c)->(bn,t2,v*c)->(bn,t2,c)

        for layer in range(self.attention_layers):
            # Split the hidden layer into its various heads

            att_value_heads = att_value.reshape(
                att_value.shape[0], att_value.shape[1], self.attention_heads, self.attention_dim
            )
            product_base = torch.einsum("bvhi,bhwi->bhvw", att_value_heads, keys[layer]) #(bnv,t2,h,c)*(bnv,h,t,c)->(bnv,h,t2,t)

            product = product_base - product_mask #(bn,h,t2,t)
            product = self.attention_dim ** (-0.5) * product
            weights = nn.functional.softmax(product, dim=-1)

            att = torch.einsum("bhvw,bhwj->bvhj", weights,values[layer])  # (bnv,h,t2,t) *(bnv,h,t,c)->(bnv,t2,h,c)

            # Merge back the various heads to allow the feed forwards module to share information between heads
            att_merged_heads = att.reshape(att.shape[0], att.shape[1],att.shape[2] * att.shape[3])  # (bnv,t2,h*c)
            att_merged_heads = self.attention_dropouts[layer](att_merged_heads)
            att_value = att_value + att_merged_heads
            att_value = self.attention_layer_norms[layer](att_value)  # (bnv,t2,h*attn_dim)
            att_feed_forward = self.feed_forwards[layer](att_value)
            att_value = att_value + att_feed_forward  # (bnv,t2,h*atttn_dim)
            att_value = self.feed_forward_layer_norms[layer](att_value)  # (bn,t2,h*attn_dim)

        true_value=true_value.reshape(b,n,v,t)[:,:,:,-t2:]
        true_value=true_value.reshape(b*n,v,t2).permute(0,2,1) #(bn,t2,v)

        log_prob=self.nf.log_prob(true_value,att_value)  #(bn,t2,v) (bn,t2,c) 特征n*v个1-(t-1)时刻  (b,t2)


        return -log_prob.sum(1)

    def sample(
            self, num_samples: int, hist_encoded: torch.Tensor, hist_true_value: torch.Tensor,
            pred_encoded: torch.Tensor ) -> torch.Tensor:


        b, nv, t1, c = hist_encoded.shape
        v = 3
        n = nv // v
        t2 = 12

        device = hist_encoded.device


        hist_encoded = hist_encoded.reshape(b * n,v, t1, c)
        hist_true_value = hist_true_value.reshape(b * n,v, t1)
        pred_encoded=pred_encoded.reshape(b*n,v,t2,c)

        hist_value = torch.cat([hist_encoded, hist_true_value[:, :,:, None]], dim=-1)  # (b*n,v,t1,c+1)
        hist_value = hist_value.permute(0, 2,1,3).reshape(b*n,t1,v*(c+1))  # (bn,t1,v*(c+1))

        keys_hist = [
            torch.cat([linear(hist_value)[:, None, :,:] for linear in self.key_creators[layer]], axis=1)
            for layer in range(self.attention_layers)
        ]
        #(bn,t1,v*(c+1))->(bn,t1,c)->(bn,h,t1,c)

        values_hist = [
            torch.cat([linear(hist_value)[:, None, :, :] for linear in self.value_creators[layer]], axis=1)
            for layer in range(self.attention_layers)
        ]  #(bn,c,t1,v)->(bn,c,t1,1)->(bn,h,c,t1,1)->(bn,h,t1,c,1)->(bn,h,t1,c)
        #(bn,t1,v*(c+1))->(bn,t1,c)->(bn,h,t1,c)

        samples = torch.zeros(b, nv,t2, num_samples).to(device)
        keys_samples = [
            torch.zeros(
                b*n, num_samples, self.attention_heads, t2, self.attention_dim, device=device
            )
            for _ in range(self.attention_layers)
        ]
        values_samples = [
            torch.zeros(
                b*n, num_samples, self.attention_heads, t2, self.attention_dim, device=device
            )
            for _ in range(self.attention_layers)
        ]


        for i in range(t2):
            current_pred_encoded=pred_encoded[:,:,i,:][:,:,None,:].expand(-1,-1,num_samples,-1)#(bn,v,s,c)
            current_pred_encoded1=current_pred_encoded.permute(0,2,1,3).reshape(b*n,num_samples,-1) #(bn,s,v,c)->(bn,s,v*c)
            att_value = self.dimension_shifting_layer(current_pred_encoded1) #(bn,s,c) t时刻


            for layer in range(self.attention_layers):
                att_value_heads = att_value.reshape(
                    att_value.shape[0], att_value.shape[1], self.attention_heads, self.attention_dim
                )#(bn,s,h,c)


                product_hist = torch.einsum("bnhi,bhwi->bnhw", att_value_heads, keys_hist[layer])
                # (bn,s,h,c)*(bn,h,t1,c)->(bn,s,h,t1)

                # keys_samples is full of zero starting at i of the 4th dimension (w)
                product_samples = torch.einsum(
                    "bnhi,bnhwi->bnhw", att_value_heads, keys_samples[layer][:, :, :, 0:i, :]
                )  # (bn,s,h,c)*(bn,s,h,t2[:i],c)->(bn,s,h,t2[:i]) s个t时刻和t2的关系

                product = torch.cat([product_hist, product_samples], axis=3)  # (bnv,s,h,t1+t2[:i]))
                product = self.attention_dim ** (-0.5) * product

                weights = nn.functional.softmax(product, dim=3)
                weights_hist = weights[:, :, :, :t1]
                weights_samples = weights[:, :, :, t1:]

                att_hist = torch.einsum("bnhw,bhwj->bnhj", weights_hist, values_hist[layer])
                att_samples = torch.einsum(
                    "bnhw,bnhwj->bnhj", weights_samples, values_samples[layer][:, :, :, 0:i, :]
                )  # i >= 1 (bn,s,h,t2[:i])*(bn,s,h,t2[:i],c)->(bn,s,h,c)

                att = att_hist + att_samples  # (bn,s,h,c)

                # # Merge back the various heads to allow the feed forwards module to share information between heads
                att_merged_heads = att.reshape(att.shape[0], att.shape[1], att.shape[2] * att.shape[3])  # (bnv,s,h*c)
                att_merged_heads = self.attention_dropouts[layer](att_merged_heads)
                att_value = att_value + att_merged_heads
                att_value = self.attention_layer_norms[layer](att_value)  # (bn,s,h*attn_dim)
                att_feed_forward = self.feed_forwards[layer](att_value)
                att_value = att_value + att_feed_forward  # (bn,s,h*atttn_dim)
                att_value = self.feed_forward_layer_norms[layer](att_value)  # (bn,s,h*attn_dim)

            #**************************
            samples_t=self.nf.sample(att_value.shape[:-1],att_value) #(bn,num_samples,v) (bn,s,v*c) ->(bn,s,v)
            samples_t=samples_t.permute(0,2,1) #(bn,v,s)
            key_value_input = torch.cat([current_pred_encoded, samples_t[:, :,:, None]],axis=-1)  # (bn,v,s,c)(bn,v,s,1)->(bn,v,s,c+1)

            key_value_input=key_value_input.permute(0,2,1,3).reshape(b*n,num_samples,-1)#(bn,s,v*(c+1))

            for layer in range(self.attention_layers):
                new_keys = torch.cat([k(key_value_input)[:, :, None, :] for k in self.key_creators[layer]], axis=2)  #(bn,s,v*(c+1))->(bn,s,c)->(bn,s,h,c)
                new_values = torch.cat([v(key_value_input)[:, :, None, :] for v in self.value_creators[layer]], axis=2) #(bn,s,v*(c+1))->(bn,s,c)->(bn,s,h,c)
                keys_samples[layer][:, :, :, i, :] = new_keys  #(bn,s,h,t,c)
                values_samples[layer][:, :, :, i, :] = new_values

            samples[:,:,i,:]=samples_t.reshape(b,n*v,num_samples)  #(b,nv,t2,s)

        return samples





















