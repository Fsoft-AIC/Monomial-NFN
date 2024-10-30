"""This example creates an NFN to process the weight space of a small CNN and output a scalar,
then verifies that the NFN is permutation invariant. This example doesn't train anything; it
is merely a demonstration of how to use the NFN library.
"""

import torch
from torch import nn
from torch.utils.data.dataloader import default_collate
from nfn.common import state_dict_to_tensors, WeightSpaceFeatures, network_spec_from_wsfeat
from nfn.layers import HNPSLinear, HNPPool, TupleOp, HNPSPool

from examples.basic_cnn.helpers import make_cnn, sample_perm_scale, check_perm_scale_symmetry


def make_nfn(network_spec, nfn_channels=1):
    return nn.Sequential(
        # io_embed: encode the input and output dimensions of the weight space feature
        #HNPSLinear(network_spec, 1, nfn_channels),
        #TupleOp(nn.ReLU()),
        #HNPSLinear(network_spec, nfn_channels, nfn_channels),
        #TupleOp(nn.ReLU()),
        HNPSPool(network_spec,nfn_channels=1),
        nn.Flatten(start_dim=-2),
        nn.Linear(nfn_channels * HNPSPool.get_num_outs(network_spec), 1)
    )


@torch.no_grad()
def main():
    print(f"Sanity check: permuting CNN channels preserves CNN behavior: {check_perm_scale_symmetry()}.")

    # Constructed two feature maps, one that is a permutation of the other.
    wts_and_bs, wts_and_bs_perm_scale = [], []
    for _ in range(10):
        sd = make_cnn().state_dict()
        wts_and_bs.append(state_dict_to_tensors(sd))
        state_dict_tensors_perm_scale = sample_perm_scale(sd)
        wts_and_bs_perm_scale.append(state_dict_to_tensors(state_dict_tensors_perm_scale))

    # Here we manually collate weights and biases (stack into batch dim).
    # When using a dataloader, the collate is done automatically.
    # default_collate output is [2 (weight and bias), num_layer, batch]
    wtfeat = WeightSpaceFeatures(*default_collate(wts_and_bs))
    wtfeat_perm = WeightSpaceFeatures(*default_collate(wts_and_bs_perm_scale))

    network_spec = network_spec_from_wsfeat(wtfeat)
    nfn = make_nfn(network_spec)
    print(nfn)

    out = nfn(wtfeat)
    out_of_perm = nfn(wtfeat_perm)
        
    print(f"NFN is invariant: {torch.allclose(out, out_of_perm)}.")


if __name__ == "__main__":
    main()