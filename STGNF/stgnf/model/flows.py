import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def create_masks(
    input_size, hidden_size, n_hidden, input_order="sequential", input_degrees=None
):
    # MADE paper sec 4:
    # degrees of connections between layers -- ensure at most in_degree - 1 connections
    degrees = []

    # set input degrees to what is provided in args (the flipped order of the previous layer in a stack of mades);
    # else init input degrees based on strategy in input_order (sequential or random)
    if input_order == "sequential":
        degrees += (
            [torch.arange(input_size)] if input_degrees is None else [input_degrees]
        )
        for _ in range(n_hidden + 1):
            degrees += [torch.arange(hidden_size) % (input_size - 1)]
        degrees += (
            [torch.arange(input_size) % input_size - 1]
            if input_degrees is None
            else [input_degrees % input_size - 1]
        )

    elif input_order == "random":
        degrees += (
            [torch.randperm(input_size)] if input_degrees is None else [input_degrees]
        )
        for _ in range(n_hidden + 1):
            min_prev_degree = min(degrees[-1].min().item(), input_size - 1)
            degrees += [torch.randint(min_prev_degree, input_size, (hidden_size,))]
        min_prev_degree = min(degrees[-1].min().item(), input_size - 1)
        degrees += (
            [torch.randint(min_prev_degree, input_size, (input_size,)) - 1]
            if input_degrees is None
            else [input_degrees - 1]
        )

    # construct masks
    masks = []
    for (d0, d1) in zip(degrees[:-1], degrees[1:]):
        masks += [(d1.unsqueeze(-1) >= d0.unsqueeze(0)).float()]

    return masks, degrees[0]


class FlowSequential(nn.Sequential):
    """ Container for layers of a normalizing flow """

    def forward(self, x, y):
        sum_log_abs_det_jacobians = 0
        for module in self:
            x, log_abs_det_jacobian = module(x, y)
            sum_log_abs_det_jacobians += log_abs_det_jacobian
        return x, sum_log_abs_det_jacobians

    def inverse(self, u, y):
        sum_log_abs_det_jacobians = 0
        i=0
        for module in reversed(self):
            # print(i)
            # print(u.mean())
            u, log_abs_det_jacobian = module.inverse(u, y)
            # print(u.mean())
            i=i+1
            sum_log_abs_det_jacobians += log_abs_det_jacobian
        return u, sum_log_abs_det_jacobians


class BatchNorm(nn.Module):
    """ RealNVP BatchNorm layer """  #actnorm

    def __init__(self, input_size, momentum=0.9, eps=1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps

        self.log_gamma = nn.Parameter(torch.zeros(input_size))
        self.beta = nn.Parameter(torch.zeros(input_size))

        self.register_buffer("running_mean", torch.zeros(input_size))
        self.register_buffer("running_var", torch.ones(input_size))

    def forward(self, x, cond_y=None):
        # print(x.mean())
        if self.training:
            # self.batch_mean = x.view(-1, x.shape[-1]).mean(0)
            # # note MAF paper uses biased variance estimate; ie x.var(0, unbiased=False)
            # self.batch_var = x.view(-1, x.shape[-1]).var(0)
            self.batch_mean = x.reshape(-1, x.shape[-1]).mean(0)
            # note MAF paper uses biased variance estimate; ie x.var(0, unbiased=False)
            self.batch_var = x.reshape(-1, x.shape[-1]).var(0)
            # update running mean
            self.running_mean.mul_(self.momentum).add_(
                self.batch_mean.data * (1 - self.momentum)
            )
            # print(self.batch_mean.data)
            # print(self.running_mean)
            self.running_var.mul_(self.momentum).add_(
                self.batch_var.data * (1 - self.momentum)
            )

            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        # compute normalized input (cf original batch norm paper algo 1)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        y = self.log_gamma.exp() * x_hat + self.beta

        # compute log_abs_det_jacobian (cf RealNVP paper)
        log_abs_det_jacobian = self.log_gamma - 0.5 * torch.log(var + self.eps)
        #        print('in sum log var {:6.3f} ; out sum log var {:6.3f}; sum log det {:8.3f}; mean log_gamma {:5.3f}; mean beta {:5.3f}'.format(
        #            (var + self.eps).log().sum().data.numpy(), y.var(0).log().sum().data.numpy(), log_abs_det_jacobian.mean(0).item(), self.log_gamma.mean(), self.beta.mean()))
        return y, log_abs_det_jacobian.expand_as(x)

    def inverse(self, y, cond_y=None):
        # print(y.mean())
        # print(self.training)
        if self.training:
            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (y - self.beta) * torch.exp(-self.log_gamma)
        # print(x_hat.mean())
        # print(var.mean())
        # print(mean.mean())
        x = x_hat * torch.sqrt(var + self.eps) + mean
        # print(x.mean())
        log_abs_det_jacobian = 0.5 * torch.log(var + self.eps) - self.log_gamma

        return x, log_abs_det_jacobian.expand_as(x)

class ActNormFlow(nn.Module):

    def __init__(self,in_features,mask,inverse=False):
        super().__init__()
        self.register_buffer("mask", mask)
        self.in_features=in_features
        self.log_scale=nn.Parameter(torch.Tensor(in_features))
        self.bias=nn.Parameter(torch.Tensor(in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.log_scale,mean=0,std=0.05)
        nn.init.constant_(self.bias,0.)

    def forward(self,x:torch.Tensor,cond_y=None):
            #->tuple[torch.Tensor,torch.Tensor]:
       """

       :param input: input :Tensor
                         input tensor[batch,N1,N2,...,N1,in_features]
                     mask:Tensor
                        mask tensor[batch,N1,N2,...,N1]
       :return: out:Tensor,logdet:Tensor
                out:[batch,N1,N2,...,in_features],the output of the flow
                logdet:[batch]mthe log deteminant of :math:'/partial output/\partial input'
       """
       dim=x.dim()   #(bn,t2,v)
       _,t2,_=x.shape
       #out=input*self.log_scale.exp()+self.bias
       out = x * torch.exp(self.log_scale) + self.bias

       #print(self.mask.unsqueeze(dim=-1).shape)
       #out=out*self.mask.unsqueeze(dim=-1)
       #out = out * self.mask

       #logdet=self.log_scale.sum(dim=0,keepdim=True) #(1)
       #logdet = self.log_scale  # (1)
       #logdet = self.log_scale*self.mask
       logdet = self.log_scale*t2
            # if dim>2:
       #     num=mask.view(out.size(0),-1).sum(dim=1)  #(v)->(bn,3)
       #     logdet=logdet*num
       #print(logdet.shape)
       return out,logdet.expand_as(x)

    def inverse(self,y:torch.Tensor,y_cond=None):

        out=(y-self.bias)*torch.exp(-self.log_scale)
        #out=out*self.mask

        log_abs_det_jacobian=0

        return out,log_abs_det_jacobian

class ActNorm(nn.Module):
    """
    ActNorm layer.

    [Kingma and Dhariwal, 2018.]
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mu = nn.Parameter(torch.zeros(dim, dtype = torch.float))
        self.log_sigma = nn.Parameter(torch.zeros(dim, dtype = torch.float))

    def forward(self, x,cond_y=None):  #(b,c,h,w)  ()
        z = x * torch.exp(self.log_sigma) + self.mu
        #log_det = torch.sum(self.log_sigma)
        log_det = self.log_sigma
        #print(z.shape)
        #print(log_det.shape)
        return z, log_det

    def inverse(self, z,cond_y=None):
        x = (z - self.mu) / torch.exp(self.log_sigma)
        #log_det = -torch.sum(self.log_sigma)
        log_det = -self.log_sigma
        #return x, log_det.expand_as(x)
        #print(log_det.shape)
        return x, log_det.expand_as(x)


class ActNorm1(nn.Module):
    """
    ActNorm layer.

    [Kingma and Dhariwal, 2018.]
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mu = nn.Parameter(torch.zeros(dim, dtype = torch.float))
        self.log_sigma = nn.Parameter(torch.zeros(dim, dtype = torch.float))

    def forward(self, x,cond_y=None):  #(b,c,h,w)  ()
        z = x * torch.exp(self.log_sigma) + self.mu
        log_det = torch.sum(self.log_sigma)
        #print(z.shape)
        return z, log_det

    def inverse(self, z,cond_y=None):
        x = (z - self.mu) / torch.exp(self.log_sigma)
        log_det = -torch.sum(self.log_sigma)

        return x, log_det.expand_as(x)





class LinearMaskedCoupling(nn.Module):
    """ Modified RealNVP Coupling Layers per the MAF paper """

    def __init__(self, input_size, hidden_size, n_hidden, mask, cond_label_size=None):
        super().__init__()

        self.register_buffer("mask", mask)

        # scale function
        s_net = [
            nn.Linear(
                input_size + (cond_label_size if cond_label_size is not None else 0),
                hidden_size,
            )
        ]
        for _ in range(n_hidden):
            s_net += [nn.Tanh(), nn.Linear(hidden_size, hidden_size)]
        s_net += [nn.Tanh(), nn.Linear(hidden_size, input_size)]
        self.s_net = nn.Sequential(*s_net)

        # translation function
        self.t_net = copy.deepcopy(self.s_net)
        # replace Tanh with ReLU's per MAF paper
        for i in range(len(self.t_net)):
            if not isinstance(self.t_net[i], nn.Linear):
                self.t_net[i] = nn.ReLU()

    def forward(self, x, y=None):
        # apply mask
        #print(x.mean())
        # print(x.shape)
        # print(y.shape)
        mx = x * self.mask  #(bn,t2,v)  y(bn,t2,v*c)

        # run through model
        s = self.s_net(mx if y is None else torch.cat([y, mx], dim=-1))  #(bn,t2,v+c)——>(bn,t2,v)
        t = self.t_net(mx if y is None else torch.cat([y, mx], dim=-1)) * (
            1 - self.mask
        )

        #print(s)
        # cf RealNVP eq 8 where u corresponds to x (here we're modeling u)
        log_s = torch.tanh(s) * (1 - self.mask)
        u = x * torch.exp(log_s) + t
        #print(u)
        # print(s)
        # print(log_s)
        #print(t)

        # u = (x - t) * torch.exp(log_s)
        # u = mx + (1 - self.mask) * (x - t) * torch.exp(-s)

        # log det du/dx; cf RealNVP 8 and 6; note, sum over input_size done at model log_prob
        # log_abs_det_jacobian = -(1 - self.mask) * s
        # log_abs_det_jacobian = -log_s #.sum(-1, keepdim=True)
        log_abs_det_jacobian = log_s
        #print(log_abs_det_jacobian.shape)  #(bn,t2,v)
        #print(u.shape)
        return u, log_abs_det_jacobian


    # def forward(self, x, y=None):
    #
    #     mx = x * self.mask
    #
    #     # run through model
    #     s = self.s_net(mx if y is None else torch.cat([y, mx], dim=-1))
    #     t = self.t_net(mx if y is None else torch.cat([y, mx], dim=-1)) * (
    #         1 - self.mask
    #     )
    #
    #     # cf RealNVP eq 8 where u corresponds to x (here we're modeling u)
    #     log_s = torch.tanh(s) * (1 - self.mask)
    #     u = x * torch.exp(log_s) + t
    #
    #     log_abs_det_jacobian = log_s
    #
    #     return u, log_abs_det_jacobian

    def inverse(self, u, y=None):
        # apply mask

        #print(u.mean())
        mu = u * self.mask

        # run through model
        s = self.s_net(mu if y is None else torch.cat([y, mu], dim=-1))
        t = self.t_net(mu if y is None else torch.cat([y, mu], dim=-1)) * (
            1 - self.mask
        )


        log_s = torch.tanh(s) * (1 - self.mask)
        x = (u - t) * torch.exp(-log_s)
        #print(x.mean())
        #print(x.mean())
        # x = u * torch.exp(log_s) + t
        # x = mu + (1 - self.mask) * (u * s.exp() + t)  # cf RealNVP eq 7


        # log_abs_det_jacobian = (1 - self.mask) * s  # log det dx/du
        # log_abs_det_jacobian = log_s #.sum(-1, keepdim=True)
        log_abs_det_jacobian = -log_s
        #print(log_abs_det_jacobian.shape)
        return x, log_abs_det_jacobian


class MaskedLinear(nn.Linear):
    """ MADE building block layer """

    def __init__(self, input_size, n_outputs, mask, cond_label_size=None):
        super().__init__(input_size, n_outputs)

        self.register_buffer("mask", mask)

        self.cond_label_size = cond_label_size
        if cond_label_size is not None:
            self.cond_weight = nn.Parameter(
                torch.rand(n_outputs, cond_label_size) / math.sqrt(cond_label_size)
            )

    def forward(self, x, y=None):
        out = F.linear(x, self.weight * self.mask, self.bias)
        if y is not None:
            out = out + F.linear(y, self.cond_weight)
        return out


class MADE(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        n_hidden,
        cond_label_size=None,
        activation="ReLU",
        input_order="sequential",
        input_degrees=None,
    ):
        """
        Args:
            input_size -- scalar; dim of inputs
            hidden_size -- scalar; dim of hidden layers
            n_hidden -- scalar; number of hidden layers
            activation -- str; activation function to use
            input_order -- str or tensor; variable order for creating the autoregressive masks (sequential|random)
                            or the order flipped from the previous layer in a stack of MADEs
            conditional -- bool; whether model is conditional
        """
        super().__init__()
        # base distribution for calculation of log prob under the model
        self.register_buffer("base_dist_mean", torch.zeros(input_size))
        self.register_buffer("base_dist_var", torch.ones(input_size))

        # create masks
        masks, self.input_degrees = create_masks(
            input_size, hidden_size, n_hidden, input_order, input_degrees
        )

        # setup activation
        if activation == "ReLU":
            activation_fn = nn.ReLU()
        elif activation == "Tanh":
            activation_fn = nn.Tanh()
        else:
            raise ValueError("Check activation function.")

        # construct model
        self.net_input = MaskedLinear(
            input_size, hidden_size, masks[0], cond_label_size
        )
        self.net = []
        for m in masks[1:-1]:
            self.net += [activation_fn, MaskedLinear(hidden_size, hidden_size, m)]
        self.net += [
            activation_fn,
            MaskedLinear(hidden_size, 2 * input_size, masks[-1].repeat(2, 1)),
        ]
        self.net = nn.Sequential(*self.net)

    @property
    def base_dist(self):
        return Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, y=None):
        # MAF eq 4 -- return mean and log std
        m, loga = self.net(self.net_input(x, y)).chunk(chunks=2, dim=-1)
        u = (x - m) * torch.exp(-loga)
        # MAF eq 5
        log_abs_det_jacobian = -loga
        return u, log_abs_det_jacobian

    def inverse(self, u, y=None, sum_log_abs_det_jacobians=None):
        # MAF eq 3
        # D = u.shape[-1]
        x = torch.zeros_like(u)
        # run through reverse model
        for i in self.input_degrees:
            m, loga = self.net(self.net_input(x, y)).chunk(chunks=2, dim=-1)
            x[..., i] = u[..., i] * torch.exp(loga[..., i]) + m[..., i]
        log_abs_det_jacobian = loga
        return x, log_abs_det_jacobian

    def log_prob(self, x, y=None):
        u, log_abs_det_jacobian = self.forward(x, y)

        return torch.sum(self.base_dist.log_prob(u) + log_abs_det_jacobian, dim=-1)


class Flow(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.__scale = None
        self.net = None

        # base distribution for calculation of log prob under the model
        self.register_buffer("base_dist_mean", torch.zeros(input_size))
        self.register_buffer("base_dist_var", torch.ones(input_size))

    @property
    def base_dist(self):
        return Normal(self.base_dist_mean, self.base_dist_var)

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, scale):
        self.__scale = scale

    def forward(self, x, cond):
        if self.scale is not None:
            x /= self.scale
        u, log_abs_det_jacobian = self.net(x, cond)
        return u, log_abs_det_jacobian

    def inverse(self, u, cond):

        x, log_abs_det_jacobian = self.net.inverse(u, cond)

        if self.scale is not None:
            x *= self.scale
            log_abs_det_jacobian += torch.log(torch.abs(self.scale))

        return x, log_abs_det_jacobian

    def log_prob(self, x, cond):
        u, sum_log_abs_det_jacobians = self.forward(x, cond)

        return torch.sum(self.base_dist.log_prob(u) + sum_log_abs_det_jacobians, dim=-1)

    def sample(self, sample_shape=torch.Size(), cond=None):
        if cond is not None:
            shape = cond.shape[:-1]
        else:
            shape = sample_shape

        u = self.base_dist.sample(shape)
        sample, _ = self.inverse(u, cond)
        return sample


class RealNVP(Flow):
    def __init__(
        self,
        n_blocks,
        input_size,
        hidden_size,
        n_hidden,
        cond_label_size=None,
        batch_norm=True,
    ):
        super().__init__(input_size)

        # construct model
        modules = []
        mask = torch.arange(input_size).float() % 2
        for i in range(n_blocks):
            modules += [
                LinearMaskedCoupling(
                    input_size, hidden_size, n_hidden, mask, cond_label_size
                )
            ]
            mask = 1 - mask

            #modules += batch_norm * [BatchNorm(input_size)]  #batch_norm=false
            modules += batch_norm * [ActNorm(input_size)]
        self.net = FlowSequential(*modules)


class MAF(Flow):
    def __init__(
        self,
        n_blocks,
        input_size,
        hidden_size,
        n_hidden,
        cond_label_size=None,
        activation="ReLU",
        input_order="sequential",
        batch_norm=True,
    ):
        super().__init__(input_size)

        # construct model
        modules = []
        self.input_degrees = None
        for i in range(n_blocks):
            modules += [
                MADE(
                    input_size,
                    hidden_size,
                    n_hidden,
                    cond_label_size,
                    activation,
                    input_order,
                    self.input_degrees,
                )
            ]
            self.input_degrees = modules[-1].input_degrees.flip(0)
            modules += batch_norm * [BatchNorm(input_size)]

        self.net = FlowSequential(*modules)
