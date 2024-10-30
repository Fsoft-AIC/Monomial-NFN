import math
import numpy as np
import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange
from nfn.common import NetworkSpec, WeightSpaceFeatures
from nfn.layers.layer_utils import set_init_, shape_wsfeat_symmetry, unshape_wsfeat_symmetry, set_init_einsum_
import torch.nn.functional as F
import math


class NPPool(nn.Module):
    def __init__(self, network_spec: NetworkSpec):
        super().__init__()
        self.network_spec = network_spec

    def forward(self, wsfeat: WeightSpaceFeatures) -> torch.Tensor:
        out = []
        for i in range(len(self.network_spec)):
            weight, bias = wsfeat[i]
            out.append(weight.mean(dim=(2,3)).unsqueeze(-1))
            out.append(bias.mean(dim=-1).unsqueeze(-1))
        return torch.cat([torch.flatten(o, start_dim=2) for o in out], dim=-1)

    @staticmethod
    def get_num_outs(network_spec):
        """Returns the number of outputs of the global pooling layer."""
        filter_facs = [int(np.prod(spec.shape[2:])) for spec in network_spec.weight_spec]
        num_outs = 0
        for fac in filter_facs:
            num_outs += fac + 1
        return num_outs


class HNPPool(nn.Module):
    def __init__(self, network_spec: NetworkSpec):
        super().__init__()
        self.network_spec = network_spec

    def forward(self, wsfeat: WeightSpaceFeatures) -> torch.Tensor:
        out = []
        for i in range(len(self.network_spec)):
            weight, bias = wsfeat[i]
            if i == 0:
                out.append(weight.mean(dim=2))  # average over rows
            elif i == len(wsfeat) - 1:
                out.append(weight.mean(dim=3))  # average over cols
            else:
                out.append(weight.mean(dim=(2,3)).unsqueeze(-1))
            if i == len(wsfeat) - 1: out.append(bias)
            else: out.append(bias.mean(dim=-1).unsqueeze(-1))
        return torch.cat([torch.flatten(o, start_dim=2) for o in out], dim=-1)

    @staticmethod
    def get_num_outs(network_spec):
        """Returns the number of outputs of the global pooling layer."""
        filter_facs = [int(np.prod(spec.shape[2:])) for spec in network_spec.weight_spec]
        n_in, n_out = network_spec.get_io()
        num_outs = 0
        for i, fac in enumerate(filter_facs):
            if i == 0:
                num_outs += n_in * fac + 1
            elif i == len(filter_facs) - 1:
                num_outs += n_out * fac + n_out
            else:
                num_outs += fac + 1
        return num_outs


class HNPSNormalize(nn.Module):
    def __init__(self, network_spec: NetworkSpec, nfn_channels, mode_normalize="param_mul_L2"):
        super().__init__()
        self.network_spec = network_spec
        self.mode_normalize = mode_normalize
        self.nfn_channels = nfn_channels
        if self.mode_normalize == "param_mul_L2":
            for i in range(len(network_spec)):
                if len(network_spec.weight_spec[i].shape) == 4: #CNN
                    self.add_module(f"regularize_W_{i}", 
                                    ElementwiseParamNormalize(nfn_channels * 
                                                math.prod(network_spec.weight_spec[i].shape[-2:]),
                                                mode_normalize=mode_normalize)
                                    )
                else:
                    self.add_module(f"regularize_W_{i}", ElementwiseParamNormalize(nfn_channels, mode_normalize=mode_normalize))
                self.add_module(f"regularize_b_{i}", ElementwiseParamNormalize(nfn_channels, mode_normalize=mode_normalize))
                                               

    def forward(self, wsfeat: WeightSpaceFeatures) -> torch.Tensor:
        out_weights = []
        out_biases = []
        for i in range(len(self.network_spec)):
            weight, bias = wsfeat[i]            
            if self.mode_normalize == "param_mul_L2":
                regularizer_w = getattr(self, f"regularize_W_{i}")
                regularizer_b = getattr(self, f"regularize_b_{i}")
            else:
                regularizer_w = self.regularize_without_param
                regularizer_b = self.regularize_without_param
            
            
            if i == 0:
                weight_regularized = regularizer_w(weight)
                out_weights.append(weight_regularized)
            
            elif i == len(wsfeat) - 1:
                weight_regularized = regularizer_w(weight)
                out_weights.append(weight_regularized)
            
            else:
                weight_regularized = regularizer_w(weight)
                out_weights.append(weight_regularized)
            
            # bias_regularized = F.normalize(bias, dim=1, p=2.0)
            bias_regularized = regularizer_b(bias)
            out_biases.append(bias_regularized)

        return WeightSpaceFeatures(out_weights, out_biases)

    def regularize_without_param(self, weight):
        if self.mode_normalize == "L1":
            if weight.dim() == 6:
                weight_shape = weight.shape
                weight = rearrange(weight, 'b c i j k l -> b (c k l) i j')
                weight_regularized = F.normalize(weight, dim=1, p=1.0)
                weight_regularized = rearrange(weight_regularized, 'b (c k l) i j -> b c i j k l',
                                                c = weight_shape[1], k = weight_shape[-2], 
                                                l = weight_shape[-1])
            else:
                weight_regularized = F.normalize(weight, dim=1, p=2.0)
        elif self.mode_normalize == "L2":
            if weight.dim() == 6:
                weight_shape = weight.shape
                weight = rearrange(weight, 'b c i j k l -> b (c k l) i j')
                weight_regularized = F.normalize(weight, dim=1, p=2.0)
                weight_regularized = rearrange(weight_regularized, 'b (c k l) i j -> b c i j k l',
                                                c = weight_shape[1], k = weight_shape[-2], 
                                                l = weight_shape[-1])
            else:
                weight_regularized = F.normalize(weight, dim=1, p=2.0)
        elif self.mode_normalize == "L2_square":
            if weight.dim() == 6:
                weight_shape = weight.shape
                weight = rearrange(weight, 'b c i j k l -> b (c k l) i j')
                weight_regularized = F.normalize(weight, dim=1, p=2.0) ** 2
                weight_regularized = rearrange(weight_regularized, 'b (c k l) i j -> b c i j k l',
                                                c = weight_shape[1], k = weight_shape[-2], 
                                                l = weight_shape[-1])
            else:
                weight_regularized = F.normalize(weight, dim=1, p=2.0) ** 2
        
        return weight_regularized

class HNPSPool(nn.Module):
    def __init__(self, network_spec: NetworkSpec, nfn_channels, mode_pooling="param_mul_L2"):
        super().__init__()
        self.network_spec = network_spec
        self.mode_pooling = mode_pooling
        self.nfn_channels = nfn_channels
        if self.mode_pooling == "param_mul_L2":
            for i in range(len(network_spec)):
                if len(network_spec.weight_spec[i].shape) == 4: #CNN
                    self.add_module(f"regularize_W_{i}", 
                                    ElementwiseParamNormalize(nfn_channels * 
                                                math.prod(network_spec.weight_spec[i].shape[-2:]),
                                                mode_normalize=mode_pooling)
                                    )
                else:
                    self.add_module(f"regularize_W_{i}", ElementwiseParamNormalize(nfn_channels, mode_normalize=mode_pooling))
                self.add_module(f"regularize_b_{i}", ElementwiseParamNormalize(nfn_channels, mode_normalize=mode_pooling))
                                               

    def forward(self, wsfeat: WeightSpaceFeatures) -> torch.Tensor:
        out = []
        for i in range(len(self.network_spec)):
            weight, bias = wsfeat[i]            
            if self.mode_pooling == "param_mul_L2":
                regularizer_w = getattr(self, f"regularize_W_{i}")
                regularizer_b = getattr(self, f"regularize_b_{i}")
            else:
                regularizer_w = self.regularize_without_param
                regularizer_b = self.regularize_without_param
            
            
            if i == 0:
                weight_regularized = regularizer_w(weight)
                out.append(weight_regularized.mean(dim=2))  # average over rows
            
            elif i == len(wsfeat) - 1:
                weight_regularized = regularizer_w(weight)
                out.append(weight_regularized.mean(dim=3))  # average over cols
            
            else:
                weight_regularized = regularizer_w(weight)
                out.append(weight_regularized.mean(dim=(2,3)).unsqueeze(-1))
            
            if i == len(wsfeat) - 1: 
                out.append(bias)
            else: 
                # bias_regularized = F.normalize(bias, dim=1, p=2.0)
                bias_regularized = regularizer_b(bias)
                out.append(bias_regularized.mean(dim=-1).unsqueeze(-1))

        return torch.cat([torch.flatten(o, start_dim=2) for o in out], dim=-1)

    def regularize_without_param(self, weight):
        if self.mode_pooling == "L1":
            if weight.dim() == 6:
                weight_shape = weight.shape
                weight = rearrange(weight, 'b c i j k l -> b (c k l) i j')
                weight_regularized = F.normalize(weight, dim=1, p=1.0)
                weight_regularized = rearrange(weight_regularized, 'b (c k l) i j -> b c i j k l',
                                                c = weight_shape[1], k = weight_shape[-2], 
                                                l = weight_shape[-1])
            else:
                weight_regularized = F.normalize(weight, dim=1, p=2.0)
        elif self.mode_pooling == "L2":
            if weight.dim() == 6:
                weight_shape = weight.shape
                weight = rearrange(weight, 'b c i j k l -> b (c k l) i j')
                weight_regularized = F.normalize(weight, dim=1, p=2.0)
                weight_regularized = rearrange(weight_regularized, 'b (c k l) i j -> b c i j k l',
                                                c = weight_shape[1], k = weight_shape[-2], 
                                                l = weight_shape[-1])
            else:
                weight_regularized = F.normalize(weight, dim=1, p=2.0)
        elif self.mode_pooling == "L2_square":
            if weight.dim() == 6:
                weight_shape = weight.shape
                weight = rearrange(weight, 'b c i j k l -> b (c k l) i j')
                weight_regularized = F.normalize(weight, dim=1, p=2.0) ** 2
                weight_regularized = rearrange(weight_regularized, 'b (c k l) i j -> b c i j k l',
                                                c = weight_shape[1], k = weight_shape[-2], 
                                                l = weight_shape[-1])
            else:
                weight_regularized = F.normalize(weight, dim=1, p=2.0) ** 2
        
        return weight_regularized

    @staticmethod
    def get_num_outs(network_spec):
        """Returns the number of outputs of the global pooling layer."""
        filter_facs = [int(np.prod(spec.shape[2:])) for spec in network_spec.weight_spec]
        n_in, n_out = network_spec.get_io()
        num_outs = 0
        for i, fac in enumerate(filter_facs):
            if i == 0:
                num_outs += n_in * fac + 1
            elif i == len(filter_facs) - 1:
                num_outs += n_out * fac + n_out
            else:
                num_outs += fac + 1
        return num_outs

class ElementwiseParamNormalize(nn.Module):
    def __init__(self, hidden, mode_normalize) -> None:
        super().__init__()
        self.hidden = hidden
        self.mode_normalize = mode_normalize
        self.weight = nn.Parameter(torch.ones(hidden))
        self.bias = nn.Parameter(torch.ones(hidden))
        nn.init.normal_(self.weight)
        nn.init.normal_(self.bias)
        
    
    def forward(self, input):
        if self.mode_normalize == "param_mul_L2":
            if input.dim() == 6: #C NN
                    input_shape = input.shape
                    input = rearrange(input, 'b c i j k l -> b i j (c k l)')
                    input_regularized = F.normalize(input, p=2.0, dim=-1)
                    input_regularized = self.weight * input_regularized + self.bias
                    input_regularized = rearrange(input_regularized, 'b i j (c k l) -> b c i j k l',
                                                    c = input_shape[1], k = input_shape[-2], 
                                                    l = input_shape[-1])
            elif input.dim() == 4: # MLP
                input = rearrange(input, 'b c i j-> b i j c')
                input_regularized = F.normalize(input, p=2.0, dim=-1)
                input_regularized = self.weight * input_regularized + self.bias
                input_regularized = rearrange(input_regularized, 'b i j c -> b c i j')
                
            elif input.dim() == 3: #bias
                input = rearrange(input, 'b c j-> b j c')
                input_regularized = F.normalize(input, p=2.0, dim=-1)
                input_regularized = self.weight * input_regularized + self.bias
                input_regularized = rearrange(input_regularized, 'b j c -> b c j')   
        return input_regularized
    
class Pointwise(nn.Module):
    """Assumes full row/col exchangeability of weights in each layer."""
    def __init__(self, network_spec: NetworkSpec, in_channels, out_channels):
        super().__init__()
        self.network_spec = network_spec
        self.in_channels = in_channels
        self.out_channels = out_channels
        # register num_layers in_channels -> out_channels linear layers
        self.weight_maps, self.bias_maps = nn.ModuleList(), nn.ModuleList()
        filter_facs = [int(np.prod(spec.shape[2:])) for spec in network_spec.weight_spec]
        for i in range(len(network_spec)):
            fac_i = filter_facs[i]
            self.weight_maps.append(nn.Conv2d(fac_i * in_channels, fac_i * out_channels, 1))
            self.bias_maps.append(nn.Conv1d(in_channels, out_channels, 1))

    def forward(self, wsfeat: WeightSpaceFeatures):
        wsfeat = shape_wsfeat_symmetry(wsfeat, self.network_spec)
        # weights is a list of tensors, each with shape (B, C_in, nrow, ncol)
        # each tensor is reshaped to (B, nrow * ncol, C_in), passed through a linear
        # layer, then reshaped back to (B, C_out, nrow, ncol)
        out_weights, out_biases = [], []
        for i in range(len(self.network_spec)):
            weight, bias = wsfeat[i]
            out_weights.append(self.weight_maps[i](weight))
            out_biases.append(self.bias_maps[i](bias))
        return unshape_wsfeat_symmetry(WeightSpaceFeatures(out_weights, out_biases), self.network_spec)

    def __repr__(self):
        return f"Pointwise(in_channels={self.in_channels}, out_channels={self.out_channels})"


class NPLinear(nn.Module):
    """Assume permutation symmetry of input and output layers, as well as hidden."""
    def __init__(self, network_spec: NetworkSpec, in_channels, out_channels, io_embed=False, init_type="pytorch_default"):
        super().__init__()
        self.c_in, self.c_out = in_channels, out_channels
        self.network_spec = network_spec
        L = len(network_spec)
        filter_facs = [int(np.prod(spec.shape[2:])) for spec in network_spec.weight_spec]
        n_rc_inp = L + sum(filter_facs)
        for i in range(L):
            fac_i = filter_facs[i]
            # pointwise
            self.add_module(f"layer_{i}", nn.Conv2d(fac_i * in_channels, fac_i * out_channels, 1))
            # broadcasts over rows and columns  
            self.add_module(f"layer_{i}_rc", nn.Linear(n_rc_inp * in_channels, fac_i * out_channels))

            # broadcasts over rows or columns
            row_in, col_in = fac_i * in_channels, (fac_i + 1) * in_channels
            if i > 0:
                fac_im1 = filter_facs[i - 1]
                row_in += (fac_im1 + 1) * in_channels
            if i < len(network_spec) - 1:
                fac_ip1 = filter_facs[i + 1]
                col_in += fac_ip1 * in_channels
            self.add_module(f"layer_{i}_r", nn.Conv1d(row_in, fac_i * out_channels, 1))
            self.add_module(f"layer_{i}_c", nn.Conv1d(col_in, fac_i * out_channels, 1))

            # pointwise
            self.add_module(f"bias_{i}", nn.Conv1d(col_in, out_channels, 1))
            self.add_module(f"bias_{i}_rc", nn.Linear(n_rc_inp * in_channels, out_channels))
            set_init_(
                getattr(self, f"layer_{i}"),
                getattr(self, f"layer_{i}_rc"),
                getattr(self, f"layer_{i}_r"),
                getattr(self, f"layer_{i}_c"),
                init_type=init_type,
            )
            set_init_(getattr(self, f"bias_{i}"), getattr(self, f"bias_{i}_rc"), init_type=init_type)
        self.io_embed = io_embed
        if io_embed:
            # initialize learned position embeddings to break input and output symmetry
            n_in, n_out = network_spec.get_io()
            self.in_embed = nn.Parameter(torch.randn(1, filter_facs[0] * in_channels, 1, n_in))
            self.out_embed = nn.Parameter(torch.randn(1, filter_facs[-1] * in_channels, n_out, 1))
            self.out_bias_embed = nn.Parameter(torch.randn(1, in_channels, n_out))

    def forward(self, wsfeat: WeightSpaceFeatures) -> WeightSpaceFeatures:
        wsfeat = shape_wsfeat_symmetry(wsfeat, self.network_spec)
        if self.io_embed:
            new_weights = (wsfeat.weights[0] + self.in_embed, *wsfeat.weights[1:-1], wsfeat.weights[-1] + self.out_embed)
            new_biases = (*wsfeat.biases[:-1], wsfeat.biases[-1] + self.out_bias_embed)
            wsfeat = WeightSpaceFeatures(new_weights, new_biases)
        weights, biases = wsfeat.weights, wsfeat.biases
        # weights[i] shape: [B, C, row, collumn]
        row_means = [w.mean(dim=-2) for w in weights]
        col_means = [w.mean(dim=-1) for w in weights]
        rowcol_means = [w.mean(dim=(-2, -1)) for w in weights]  # (B, C_in)
        bias_means = [b.mean(dim=-1) for b in biases]  # (B, C_in)
        wb_means = torch.cat(rowcol_means + bias_means, dim=-1)  # (B, 2 * C_in * n_layers)
        out_weights, out_biases = [], []
        for i in range(len(self.network_spec)): 
            weight, bias = wsfeat[i]
            # weights shape: [B, C, row, collumn]
            z1 = getattr(self, f"layer_{i}")(weight)  # (B, C_out, nrow, ncol)
            z2 = getattr(self, f"layer_{i}_rc")(wb_means)[..., None, None]
            row_bdcst = [row_means[i]]  # (B, C_in, ncol)
            col_bdcst = [col_means[i], bias]  # (B, 2 * C_in, nrow)
            if i > 0:
                row_bdcst.extend([col_means[i-1], biases[i-1]])  # (B, C_in, ncol)
            if i < len(wsfeat.weights) - 1:
                col_bdcst.append(row_means[i+1])  # (B, C_in, nrow)
            col_bdcst = torch.cat(col_bdcst, dim=-2)
            z3 = getattr(self, f"layer_{i}_r")(torch.cat(row_bdcst, dim=-2)).unsqueeze(-2)  # (B, C_out, 1, ncol)
            z4 = getattr(self, f"layer_{i}_c")(col_bdcst).unsqueeze(-1)  # (B, C_out, nrow, 1)
            out_weights.append(z1 + z2 + z3 + z4)

            u1 = getattr(self, f"bias_{i}")(col_bdcst)  # (B, C_out, nrow)
            u2 = getattr(self, f"bias_{i}_rc")(wb_means).unsqueeze(-1)  # (B, C_out, 1)
            out_biases.append(u1 + u2)
        return unshape_wsfeat_symmetry(WeightSpaceFeatures(out_weights, out_biases), self.network_spec)

    def __repr__(self):
        return f"NPLinear(in_channels={self.c_in}, out_channels={self.c_out})"


class HNPLinear(nn.Module):
    def __init__(self, network_spec: NetworkSpec, in_channels, out_channels, init_type="pytorch_default"):
        super().__init__()
        self.network_spec = network_spec
        n_in, n_out = network_spec.get_io()
        self.n_in, self.n_out = n_in, n_out
        self.c_in, self.c_out = in_channels, out_channels
        self.L = len(network_spec)
        filter_facs = [int(np.prod(spec.shape[2:])) for spec in network_spec.weight_spec]
        n_rc_inp = n_in * filter_facs[0] + n_out * filter_facs[-1] + self.L + n_out
        for i in range(self.L):
            n_rc_inp += filter_facs[i]
        for i in range(self.L):
            fac_i = filter_facs[i]
            if i == 0:
                fac_ip1 = filter_facs[i+1]
                rpt_in = (fac_i * n_in + fac_ip1 + 1)
                if i + 1 == self.L - 1:
                    rpt_in += n_out * fac_ip1
                self.w0_rpt = nn.Conv1d(rpt_in * in_channels, n_in * fac_i * out_channels, 1)
                self.w0_rbdcst = nn.Linear(n_rc_inp * in_channels, n_in * fac_i * out_channels)

                self.bias_0 = nn.Conv1d(rpt_in * in_channels, out_channels, 1)
                self.bias_0_rc = nn.Linear(n_rc_inp * in_channels, out_channels)
                set_init_(self.bias_0, self.bias_0_rc)
            elif i == self.L - 1:
                fac_im1 = filter_facs[i-1]
                cpt_in = (fac_i * n_out + fac_im1)
                if i - 1 == 0:
                    cpt_in += n_in * fac_im1
                self.wfin_cpt = nn.Conv1d(cpt_in * in_channels, n_out * fac_i * out_channels, 1)
                self.wfin_cbdcst = nn.Linear(n_rc_inp * in_channels, n_out * fac_i * out_channels)
                set_init_(self.wfin_cpt, self.wfin_cbdcst)

                self.bias_fin_rc = nn.Linear(n_rc_inp * in_channels, n_out * out_channels)
            else:
                self.add_module(f"layer_{i}", nn.Conv2d(fac_i * in_channels, fac_i * out_channels, 1))
                self.add_module(f"layer_{i}_rc", nn.Linear(n_rc_inp * in_channels, fac_i * out_channels))

                fac_im1, fac_ip1 = filter_facs[i-1], filter_facs[i+1]
                row_in, col_in = (fac_im1 + fac_i + 1) * in_channels, (fac_ip1 + fac_i + 1) * in_channels
                if i == 1: row_in += n_in * filter_facs[0] * in_channels
                if i == self.L - 2: col_in += n_out * filter_facs[-1] * in_channels
                self.add_module(f"layer_{i}_r", nn.Conv1d(row_in, fac_i * out_channels, 1))
                self.add_module(f"layer_{i}_c", nn.Conv1d(col_in, fac_i * out_channels, 1))
                set_init_(
                    getattr(self, f"layer_{i}"),
                    getattr(self, f"layer_{i}_rc"),
                    getattr(self, f"layer_{i}_r"),
                    getattr(self, f"layer_{i}_c"),
                )

                self.add_module(f"bias_{i}", nn.Conv1d(col_in, out_channels, 1))
                self.add_module(f"bias_{i}_rc", nn.Linear(n_rc_inp * in_channels, out_channels))
                set_init_(getattr(self, f"bias_{i}"), getattr(self, f"bias_{i}_rc"))
        self.rearr1_wt1 = Rearrange("b c_in nrow ncol -> b (c_in ncol) nrow")
        self.rearr1_wtL = Rearrange("b c_in n_out nrow -> b (c_in n_out) nrow")
        self.rearr1_outwt = Rearrange("b (c_out ncol) nrow -> b c_out nrow ncol", ncol=n_in)
        self.rearrL_wtL = Rearrange("b c_in nrow ncol -> b (c_in nrow) ncol")
        self.rearrL_wt1 = Rearrange("b c_in ncol n_in -> b (c_in n_in) ncol")
        self.rearrL_outwt = Rearrange("b (c_out nrow) ncol -> b c_out nrow ncol", nrow=n_out)
        self.rearrL_outbs = Rearrange("b (c_out nrow) -> b c_out nrow", nrow=n_out)
        self.rearr2_wt1 = Rearrange("b c ncol n_in -> b (c n_in) ncol")
        self.rearrLm1_wtL = Rearrange("b c n_out nrow -> b (c n_out) nrow")

    def forward(self, wsfeat: WeightSpaceFeatures):
        wsfeat = shape_wsfeat_symmetry(wsfeat, self.network_spec)
        weights, biases = wsfeat.weights, wsfeat.biases
        col_means = [w.mean(dim=-1) for w in weights]  # shapes: (B, C_in, nrow_i=ncol_i+1)
        row_means = [w.mean(dim=-2) for w in weights]  # shapes: (B, C_in, ncol_i=nrow_i-1)
        rc_means = [w.mean(dim=(-1, -2)) for w in weights]  # shapes: (B, C_in)
        bias_means = [b.mean(dim=-1) for b in biases]  # shapes: (B, C_in)
        rm0 = torch.flatten(row_means[0], start_dim=-2)  # b c_in ncol -> b (c_in ncol)
        cmL = torch.flatten(col_means[-1], start_dim=-2)  # b c_in nrowL -> b (c_in nrowL)
        final_bias = torch.flatten(biases[-1], start_dim=-2)  # b c_in nrow -> b (c_in nrow)
        # (B, C_in * (2 * L + n_in + 2 * n_out))
        rc_inp = torch.cat(rc_means + bias_means + [rm0, cmL, final_bias], dim=-1)

        out_weights, out_biases = [], []
        for i in range(self.L):
            weight, bias = wsfeat[i]
            if i == 0:
                rpt = [self.rearr1_wt1(weight), row_means[1], bias]
                if i+1 == self.L - 1:
                    rpt.append(self.rearr1_wtL(weights[-1]))
                rpt = torch.cat(rpt, dim=-2)  # repeat ptwise across rows
                z1 = self.w0_rpt(rpt)
                z2 = self.w0_rbdcst(rc_inp)[..., None]  # (b, c_out * ncol, 1)
                z = self.rearr1_outwt(z1 + z2)
                u1 = self.bias_0(rpt)  # (B, C_out, nrow)
                u2 = self.bias_0_rc(rc_inp).unsqueeze(-1)  # (B, C_out, 1)
                u = u1 + u2
            elif i == self.L - 1:
                cpt = [self.rearrL_wtL(weight), col_means[-2]]  # b c_in ncol 
                if i - 1 == 0:
                    cpt.append(self.rearrL_wt1(weights[0]))
                z1 = self.wfin_cpt(torch.cat(cpt, dim=-2))  # (B, C_out * nrow, ncol)
                z2 = self.wfin_cbdcst(rc_inp)[..., None]  # (b, c_out * nrow, 1)
                z = self.rearrL_outwt(z1 + z2)
                u = self.rearrL_outbs(self.bias_fin_rc(rc_inp))
            else:
                z1 = getattr(self, f"layer_{i}")(weight)  # (B, C_out, nrow, ncol)
                z2 = getattr(self, f"layer_{i}_rc")(rc_inp)[..., None, None]
                row_bdcst = [row_means[i], col_means[i-1], biases[i-1]]
                col_bdcst = [col_means[i], bias, row_means[i+1]]
                if i == 1:
                    row_bdcst.append(self.rearr2_wt1(weights[0]))
                if i == len(weights) - 2:
                    col_bdcst.append(self.rearrLm1_wtL(weights[-1]))
                row_bdcst = torch.cat(row_bdcst, dim=-2)  # (B, (3 + ?n_in) * C_in, ncol)
                col_bdcst = torch.cat(col_bdcst, dim=-2)  # (B, (3 + ?n_out) * C_in, nrow)
                z3 = getattr(self, f"layer_{i}_r")(row_bdcst).unsqueeze(-2)  # (B, C_out, 1, ncol)
                z4 = getattr(self, f"layer_{i}_c")(col_bdcst).unsqueeze(-1)  # (B, C_out, nrow, 1)
                z = z1 + z2 + z3 + z4
                u1 = getattr(self, f"bias_{i}")(col_bdcst)  # (B, C_out, nrow)
                u2 = getattr(self, f"bias_{i}_rc")(rc_inp).unsqueeze(-1)  # (B, C_out, 1)
                u = u1 + u2
            out_weights.append(z)
            out_biases.append(u)
        return unshape_wsfeat_symmetry(WeightSpaceFeatures(out_weights, out_biases), self.network_spec)

    def __repr__(self):
        return f"HNPLinear(in_channels={self.c_in}, out_channels={self.c_out}, n_in={self.n_in}, n_out={self.n_out}, num_layers={self.L})"

class HNPS_SirenLinear(nn.Module):
    def __init__(self, network_spec: NetworkSpec, in_channels, out_channels, init_type="pytorch_default"):
        super().__init__()
        self.c_in, self.c_out = in_channels, out_channels
        self.network_spec = network_spec
        layer_in_weight_shape, layer_out_weight_shape = network_spec.get_io_matrices_shape()
        L = len(network_spec)
        filter_facs = [int(np.prod(spec.shape[2:])) for spec in network_spec.weight_spec]
        for i in range(L):
            fac_i = filter_facs[i]
            if i == 0:
                self.layer_0_Y_W = EinsumLayer(equation="mnqk, bnjq -> bmjk", 
                                              weight_shape=[fac_i * out_channels, fac_i * in_channels, 
                                                             layer_in_weight_shape[-1], layer_in_weight_shape[-1]],
                                              fan_in_mask = [0, 1, 1, 0])
                self.layer_0_Y_b = EinsumLayer(equation="mnk, bnj -> bmjk",
                                               weight_shape=[fac_i * out_channels, in_channels, 
                                                             layer_in_weight_shape[-1]],
                                               fan_in_mask= [0, 1, 0])
                self.layer_0_z_W = EinsumLayer(equation="mnq, bnjq -> bmj",
                                               weight_shape=[out_channels, fac_i * in_channels, 
                                                             layer_in_weight_shape[-1]],
                                               fan_in_mask=[0, 1, 1]
                                               )
                self.layer_0_z_b = EinsumLayer(equation="mn, bnj -> bmj",
                                               weight_shape=[out_channels, in_channels],
                                               fan_in_mask=[0, 1])

                set_init_einsum_(
                    self.layer_0_Y_W,
                    self.layer_0_Y_b,
                    init_type=init_type,
                )
                set_init_einsum_(
                    self.layer_0_z_W,
                    self.layer_0_z_b,
                    init_type=init_type,
                )

            elif i == L-1:
                self.add_module(f"layer_{i}_Y_W",
                                EinsumLayer(equation="mnpj, bnpk -> bmjk",
                                            weight_shape=[fac_i * out_channels, fac_i * in_channels,
                                                          layer_out_weight_shape[-2], layer_out_weight_shape[-2]],
                                            fan_in_mask=[0, 1, 1, 0])
                                )
                # New layer for SIREN
                self.add_module(f"layer_{i}_Y_b",
                                EinsumLayer(equation="mnj, bnk -> bmjk",
                                            weight_shape=[fac_i * out_channels, fac_i * in_channels,
                                                          layer_out_weight_shape[-2]],
                                            fan_in_mask=[0, 1, 0])
                                )
                self.add_module(f"layer_{i}_z_b",
                                EinsumLayer(equation="mnpj, bnp -> bmj",
                                            weight_shape=[out_channels, fac_i * in_channels,
                                                          layer_out_weight_shape[-2], layer_out_weight_shape[-2]],
                                            fan_in_mask=[0, 1, 1, 0])
                                )
                self.add_module(f"layer_{i}_z_tau",
                                EinsumLayer(equation="mj, b-> bmj",
                                            weight_shape=[out_channels, layer_out_weight_shape[-2]],
                                            fan_in_mask=[0, 0])
                                )
                
                set_init_einsum_(
                    getattr(self, f"layer_{i}_Y_W"),
                    init_type=init_type,
                )
                set_init_einsum_(
                    getattr(self, f"layer_{i}_Y_b"),
                    init_type=init_type,
                )
                set_init_einsum_(
                    getattr(self, f"layer_{i}_z_b"),
                    getattr(self, f"layer_{i}_z_tau"),
                    init_type=init_type,
                )
                
            elif i == L-2:
                self.add_module(f"layer_{i}_Y_W",
                                EinsumLayer(equation="mn, bnjk -> bmjk",
                                            weight_shape=[fac_i * out_channels, fac_i * in_channels],
                                            fan_in_mask=[0, 1])
                                )
                # New layer for SIREN
                self.add_module(f"layer_{i}_z_W",
                                EinsumLayer(equation="mnp, bnpj -> bmj",
                                            weight_shape=[fac_i * out_channels, fac_i * in_channels, layer_out_weight_shape[-2]],
                                            fan_in_mask=[0, 1, 1])
                                )
                self.add_module(f"layer_{i}_z_b",
                                EinsumLayer(equation="mn, bnj -> bmj",
                                            weight_shape=[out_channels, in_channels],
                                            fan_in_mask=[0, 1])
                                )
                
                set_init_einsum_(
                    getattr(self, f"layer_{i}_Y_W"),
                    init_type=init_type,
                )
                set_init_einsum_(
                    getattr(self, f"layer_{i}_z_W"),
                    init_type=init_type,
                )
                set_init_einsum_(
                    getattr(self, f"layer_{i}_z_b"),
                    init_type=init_type,
                )

            else:
                self.add_module(f"layer_{i}_Y_W",
                                EinsumLayer(equation="mn, bnjk -> bmjk",
                                            weight_shape=[fac_i * out_channels, fac_i * in_channels],
                                            fan_in_mask=[0, 1])
                                )
                self.add_module(f"layer_{i}_z_b",
                                EinsumLayer(equation="mn, bnj -> bmj",
                                            weight_shape=[out_channels, in_channels],
                                            fan_in_mask=[0, 1])
                                )
                
                set_init_einsum_(
                    getattr(self, f"layer_{i}_Y_W"),
                    init_type=init_type,
                )
                set_init_einsum_(
                    getattr(self, f"layer_{i}_z_b"),
                    init_type=init_type,
                )

    
    def forward(self, wsfeat: WeightSpaceFeatures) -> WeightSpaceFeatures:
        wsfeat = shape_wsfeat_symmetry(wsfeat, self.network_spec)
        out_weights, out_biases = [], []
        L = len(self.network_spec)
        for i in range(L):
            weight, bias = wsfeat[i]
            if  i == 0:
                Y_W = self.layer_0_Y_W(weight)
                Y_b = self.layer_0_Y_b(bias)
                out_weights.append(Y_W + Y_b)
                
                z_W = self.layer_0_z_W(weight)
                z_b = self.layer_0_z_b(bias)
                out_biases.append(z_W + z_b)

            elif i == L-1:
                prev_bias = wsfeat[i-1][1]
                Y_W = getattr(self, f"layer_{i}_Y_W")(weight)
                Y_b = getattr(self, f"layer_{i}_Y_b")(prev_bias) # Previous bias
                out_weights.append(Y_W + Y_b)
                
                z_b = getattr(self, f"layer_{i}_z_b")(bias)
                z_tau = getattr(self, f"layer_{i}_z_tau")(torch.tensor([1], device=weight.device))
                out_biases.append(z_b + z_tau)
            
            elif i == L-2:
                next_weight = wsfeat[i+1][0]
                Y_W = getattr(self, f"layer_{i}_Y_W")(weight)
                out_weights.append(Y_W)
                
                z_W = getattr(self, f"layer_{i}_z_W")(next_weight) # Next weight
                z_b = getattr(self, f"layer_{i}_z_b")(bias)
                out_biases.append(z_b + z_W)
            
            else:
                Y_W = getattr(self, f"layer_{i}_Y_W")(weight)
                out_weights.append(Y_W)
                
                z_b = getattr(self, f"layer_{i}_z_b")(bias)
                out_biases.append(z_b)
                
        return unshape_wsfeat_symmetry(WeightSpaceFeatures(out_weights, out_biases), self.network_spec)

class HNPSLinear(nn.Module):
    def __init__(self, network_spec: NetworkSpec, in_channels, out_channels, init_type="pytorch_default"):
        super().__init__()
        self.c_in, self.c_out = in_channels, out_channels
        self.network_spec = network_spec
        layer_in_weight_shape, layer_out_weight_shape = network_spec.get_io_matrices_shape()
        L = len(network_spec)
        filter_facs = [int(np.prod(spec.shape[2:])) for spec in network_spec.weight_spec]
        for i in range(L):
            fac_i = filter_facs[i]
            if i == 0:
                self.layer_0_Y_W = EinsumLayer(equation="mnqk, bnjq -> bmjk", 
                                              weight_shape=[fac_i * out_channels, fac_i * in_channels, 
                                                             layer_in_weight_shape[-1], layer_in_weight_shape[-1]],
                                              fan_in_mask = [0, 1, 1, 0])
                self.layer_0_Y_b = EinsumLayer(equation="mnk, bnj -> bmjk",
                                               weight_shape=[fac_i * out_channels, in_channels, 
                                                             layer_in_weight_shape[-1]],
                                               fan_in_mask= [0, 1, 0])
                self.layer_0_z_W = EinsumLayer(equation="mnq, bnjq -> bmj",
                                               weight_shape=[out_channels, fac_i * in_channels, 
                                                             layer_in_weight_shape[-1]],
                                               fan_in_mask=[0, 1, 1]
                                               )
                self.layer_0_z_b = EinsumLayer(equation="mn, bnj -> bmj",
                                               weight_shape=[out_channels, in_channels],
                                               fan_in_mask=[0, 1])

                set_init_einsum_(
                    self.layer_0_Y_W,
                    self.layer_0_Y_b,
                    init_type=init_type,
                )
                set_init_einsum_(
                    self.layer_0_z_W,
                    self.layer_0_z_b,
                    init_type=init_type,
                )

            elif i == L-1:
                self.add_module(f"layer_{i}_Y_W",
                                EinsumLayer(equation="mnpj, bnpk -> bmjk",
                                            weight_shape=[fac_i * out_channels, fac_i * in_channels,
                                                          layer_out_weight_shape[-2], layer_out_weight_shape[-2]],
                                            fan_in_mask=[0, 1, 1, 0])
                                )
                self.add_module(f"layer_{i}_z_b",
                                EinsumLayer(equation="mnpj, bnp -> bmj",
                                            weight_shape=[out_channels, fac_i * in_channels,
                                                          layer_out_weight_shape[-2], layer_out_weight_shape[-2]],
                                            fan_in_mask=[0, 1, 1, 0])
                                )
                self.add_module(f"layer_{i}_z_tau",
                                EinsumLayer(equation="mj, b-> bmj",
                                            weight_shape=[out_channels, layer_out_weight_shape[-2]],
                                            fan_in_mask=[0, 0])
                                )
                
                set_init_einsum_(
                    getattr(self, f"layer_{i}_Y_W"),
                    init_type=init_type,
                )
                set_init_einsum_(
                    getattr(self, f"layer_{i}_z_b"),
                    getattr(self, f"layer_{i}_z_tau"),
                    init_type=init_type,
                )
                
            else:
                self.add_module(f"layer_{i}_Y_W",
                                EinsumLayer(equation="mn, bnjk -> bmjk",
                                            weight_shape=[fac_i * out_channels, fac_i * in_channels],
                                            fan_in_mask=[0, 1])
                                )
                self.add_module(f"layer_{i}_z_b",
                                EinsumLayer(equation="mn, bnj -> bmj",
                                            weight_shape=[out_channels, in_channels],
                                            fan_in_mask=[0, 1])
                                )
                
                set_init_einsum_(
                    getattr(self, f"layer_{i}_Y_W"),
                    init_type=init_type,
                )
                set_init_einsum_(
                    getattr(self, f"layer_{i}_z_b"),
                    init_type=init_type,
                )

    
    def forward(self, wsfeat: WeightSpaceFeatures) -> WeightSpaceFeatures:
        wsfeat = shape_wsfeat_symmetry(wsfeat, self.network_spec)
        out_weights, out_biases = [], []
        L = len(self.network_spec)
        for i in range(L):
            weight, bias = wsfeat[i]
            if  i == 0:
                Y_W = self.layer_0_Y_W(weight)
                Y_b = self.layer_0_Y_b(bias)
                out_weights.append(Y_W + Y_b)
                
                z_W = self.layer_0_z_W(weight)
                z_b = self.layer_0_z_b(bias)
                out_biases.append(z_W + z_b)

            elif i == L-1:
                Y_W = getattr(self, f"layer_{i}_Y_W")(weight)
                out_weights.append(Y_W)
                
                z_b = getattr(self, f"layer_{i}_z_b")(bias)
                z_tau = getattr(self, f"layer_{i}_z_tau")(torch.tensor([1], device=weight.device))
                out_biases.append(z_b + z_tau)
            
            else:
                Y_W = getattr(self, f"layer_{i}_Y_W")(weight)
                out_weights.append(Y_W)
                
                z_b = getattr(self, f"layer_{i}_z_b")(bias)
                out_biases.append(z_b)
                
        return unshape_wsfeat_symmetry(WeightSpaceFeatures(out_weights, out_biases), self.network_spec)
                

class EinsumLayer(nn.Module):
    def __init__(self, equation, weight_shape, fan_in_mask) -> None:
        super().__init__()
        self.weight_shape_list = weight_shape
        self.weight_shape_tensor = torch.tensor(weight_shape, dtype=torch.int)
        self.fan_in_mask = torch.tensor(fan_in_mask).ge(0.5)
        if torch.all(self.fan_in_mask == False):
            self.fan_in = 0
        else:
            self.fan_in = torch.prod(self.weight_shape_tensor[self.fan_in_mask])
        self.fan_out_mask = torch.tensor(fan_in_mask).lt(0.5)
        if torch.all(self.fan_out_mask == False):
            self.fan_out = 0
        else:
            self.fan_out = torch.prod(self.weight_shape_tensor[self.fan_out_mask])

        self.equation = equation
        self.weight = nn.Parameter(torch.empty(self.weight_shape_list))
    
    def forward(self, input):
        return torch.einsum(self.equation, self.weight, input)
    
        
                


def simple_attention(q, k, v, dropout=None):
    # q, k, v: (..., T, d)
    # TODO: consider replacing with F.scaled_dot_product_attention. But probably won't get
    # Flash Attention speedup unless we have A100 @ 16bit: https://github.com/pytorch/pytorch/pull/81434#issuecomment-1451120687.
    attn = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(k.shape[-1]), dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    return attn @ v


class ChannelLinear(nn.Module):
    """Probably equivalent to a 1x1 nn.Conv2d."""
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        self.channels_last = Rearrange("b c ... -> b ... c")
        self.channels_first = Rearrange("b ... c -> b c ...")

    def forward(self, x):
        x = self.channels_last(x)
        x = self.linear(x)
        return self.channels_first(x)


class NPAttention(nn.Module):
    def __init__(
        self, network_spec: NetworkSpec, channels,
        num_heads=8, dropout=0,
        share_projections=True,
        ablate_crossterm=False,
        ablate_diagonalterm=False,
        init_type="pytorch_default"
    ):
        super().__init__()
        assert channels % num_heads == 0, f"channels ({channels}) must be divisible by num_heads ({num_heads})."
        self.network_spec = network_spec
        self.share_projections = share_projections
        if share_projections:
            self.to_qkv = ChannelLinear(channels, 3 * channels)
        else:
            self.weight_to_qkv = nn.ModuleList()  # maps channel dim k * k * c --> c
            self.bias_to_qkv = nn.ModuleList()
            self.unproject_weight = nn.ModuleList()  # maps channel dim c --> k * k * c
            filter_facs = [int(np.prod(spec.shape[2:])) for spec in network_spec.weight_spec]
            for filter_fac in filter_facs:
                self.weight_to_qkv.append(ChannelLinear(filter_fac * channels, 3 * channels))
                self.bias_to_qkv.append(ChannelLinear(channels, 3 * channels))
                self.unproject_weight.append(ChannelLinear(channels, filter_fac * channels))
        self.nh = num_heads
        self.ch = channels // num_heads  # channels per head
        self.dropout = nn.Dropout(dropout)
        self.split_heads = Rearrange("b (nh ch) ... -> b nh ch ...", nh=self.nh, ch=3*self.ch)
        self.combine_nh_nc = Rearrange("b nh t c -> b t (nh c)")
        self.permute_Wi_col = Rearrange("... c n_i n_im1 -> ... n_im1 c n_i")
        self.permute_Wip1 = Rearrange("... c n_ip1 n_i -> ... n_ip1 c n_i")
        self.ablate_crossterm = ablate_crossterm
        self.ablate_diagonalterm = ablate_diagonalterm

    def forward(self, wsfeat: WeightSpaceFeatures) -> WeightSpaceFeatures:
        wsfeat = shape_wsfeat_symmetry(wsfeat, self.network_spec)
        out_weights = [torch.zeros_like(w) for w in wsfeat.weights]
        out_biases = [torch.zeros_like(b) for b in wsfeat.biases]
        if self.share_projections:
            qkv = wsfeat.map(self.to_qkv)
        else:
            qkv_weights, qkv_biases = [], []
            for i in range(len(self.network_spec)):
                qkv_weights.append(self.weight_to_qkv[i](wsfeat.weights[i]))
                qkv_biases.append(self.bias_to_qkv[i](wsfeat.biases[i]))
            qkv = WeightSpaceFeatures(qkv_weights, qkv_biases)
        qkv = qkv.map(self.split_heads)
        rowcol_means = [w.mean(dim=(-2, -1)) for w in qkv.weights]  # (B, nh, c)
        bias_means = [b.mean(dim=-1) for b in qkv.biases]  # (B, nh, c)
        qkv_avgs = torch.stack(rowcol_means + bias_means, dim=-2)  # (B, nh, 2 * n_layers, 3ch)
        q_avg, k_avg, v_avg = qkv_avgs.tensor_split(3, dim=-1)  # (B, nh, 2 * n_layers, ch)
        if not self.ablate_diagonalterm:
            wb_out = self.combine_nh_nc(simple_attention(q_avg, k_avg, v_avg))
            w_out, b_out = wb_out.tensor_split(2, dim=-2)  # (B, n_layers, nh*ch)
            for i in range(len(out_weights)):
                w_out_i = w_out[:, i].unsqueeze(-1).unsqueeze(-1)
                if not self.share_projections: w_out_i = self.unproject_weight[i](w_out_i)
                out_weights[i] += w_out_i
                out_biases[i] += b_out[:, i].unsqueeze(-1)
        if self.ablate_crossterm:
            return unshape_wsfeat_symmetry(WeightSpaceFeatures(tuple(out_weights), tuple(out_biases)), self.network_spec)
        for i in range(-1, len(wsfeat)):
            inp = []
            if i > -1:
                n_i, n_im1 = wsfeat.weights[i].shape[-2], wsfeat.weights[i].shape[-1]
                Wi_cols = self.permute_Wi_col(qkv.weights[i])
                vi = qkv.biases[i].unsqueeze(-3)  # (B, nh, 1, c, n_i)
                inp.extend([Wi_cols, vi])
            if i < len(wsfeat) - 1:
                n_i = wsfeat.weights[i+1].shape[-1]
                inp.append(self.permute_Wip1(qkv.weights[i+1]))
            inp = torch.cat(inp, dim=-3)  # (B, nh, n_im1 + 1 + n_ip1, 3c, n_i)
            q_i, k_i, v_i = inp.tensor_split(3, dim=-2) # (B, nh, n_im1 + 1 + n_ip1, c, n_i)
            # TODO: can this be simplified?
            q_i = torch.flatten(q_i, start_dim=-2)
            k_i = torch.flatten(k_i, start_dim=-2)
            v_i = torch.flatten(v_i, start_dim=-2)
            # (B, nh, n_im1 + 1 + n_ip1, c * n_i)
            out = simple_attention(q_i, k_i, v_i, dropout=self.dropout)
            # ... (ch n_i) -> ... ch n_i
            out = out.view(*out.shape[:-1], self.ch, n_i)
            idx = 0
            if i > -1:
                # b nh n_im1 ch n_i -> b (nh ch) n_i n_im1
                out_weights_i = torch.flatten(out[:, :, :n_im1].permute(0, 1, 3, 4, 2), start_dim=1, end_dim=2)
                if not self.share_projections: out_weights_i = self.unproject_weight[i](out_weights_i)
                out_weights[i] += out_weights_i
                # b nh 1 ch n_i -> b (nh ch) n_i
                out_biases[i] += torch.flatten(out[:, :, n_im1: n_im1 + 1].squeeze(2), start_dim=1, end_dim=2)
                idx = n_im1 + 1
            if i < len(wsfeat) - 1:
                # b nh n_ip1 ch n_i -> b (nh ch) n_ip1 n_i
                out_weights_ip1 = torch.flatten(out[:, :, idx:].transpose(2, 3), start_dim=1, end_dim=2)
                if not self.share_projections: out_weights_ip1 = self.unproject_weight[i+1](out_weights_ip1)
                out_weights[i+1] += out_weights_ip1
        return unshape_wsfeat_symmetry(WeightSpaceFeatures(tuple(out_weights), tuple(out_biases)), self.network_spec)

    def __repr__(self):
        return f"NPAttention(channels={self.ch * self.nh}, num_heads={self.nh}, dropout={self.dropout.p})"