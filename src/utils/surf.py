from typing import Union, Iterable

import torch
from torch import Tensor
from torch_geometric.data import Data, Batch
from torch_scatter import scatter


def get_surf_bin_features(
    data: Union[Data, Batch],
    cov_rad: Tensor,
    adsorb_atomic_numbers: Iterable,
    cov_coeff: int,
):
    """
    :example:
    >>> import pandas as pd

    >>> features = pd.read_csv("/workspaces/ocp/ocp/ocpmodels/models/features/features.csv")
    >>> cov_rad = features["Cov_radius_cordero"].values
    >>> adsorb_atomic_numbers = (1, 6, 7, 8)

    >>> get_surf_bin_features(
            data=ds[0],
            cov_rad=cov_rad,
            adsorb_atomic_numbers=adsorb_atomic_numbers,
            cov_coeff=1.5
        )
    tensor([[0., 0., 0., 1.],
           [0., 1., 0., 1.],
           [1., 0., 1., 0.],
           ...,
           [0., 1., 1., 1.],
           [1., 1., 0., 1.],
           [0., 1., 0., 0.]])

    """
    atomic_numbers = data.atomic_numbers.long()
    cov_rad = torch.tensor(cov_rad[atomic_numbers.cpu() - 1]).type_as(data.pos)

    node_bin_features = torch.zeros(
        (data.atomic_numbers.size(0), len(adsorb_atomic_numbers))
    ).type_as(data.distances)
    cat2absorb_mask = (data.tags[data.edge_index[0]] == 2) & (
        data.tags[data.edge_index[1]] == 1
    )

    for i, adsorb_number in enumerate(adsorb_atomic_numbers):
        cat2absorb_mask_atom = cat2absorb_mask & (
            atomic_numbers[data.edge_index[0]] == adsorb_number
        )
        cat2absorb_index = data.edge_index[:, cat2absorb_mask_atom]
        cat2absorb_dist = data.distances[cat2absorb_mask_atom]

        bin_features = (
            cat2absorb_dist
            / (cov_rad[cat2absorb_index[0]] + cov_rad[cat2absorb_index[1]])
            <= cov_coeff
        ).type_as(data.distances)
        edge_bin_features = torch.zeros_like(data.distances)
        edge_bin_features[cat2absorb_mask_atom] = bin_features
        node_bin_features[:, i] = scatter(
            edge_bin_features, data.edge_index[1], dim_size=atomic_numbers.size(0)
        )

    return node_bin_features
