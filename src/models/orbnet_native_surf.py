from math import sqrt
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import conditional_grad, get_pbc_distances, radius_graph_pbc
from ocpmodels.models.gemnet.layers.radial_basis import RadialBasis
from ocpmodels.models.orbnet import DenseLayer, OutputBlock, ResidualLayer
from src.utils import get_surf_bin_features
from torch import Tensor, nn
from torch.utils import data
from torch_geometric.nn import radius_graph
from torch_geometric.nn.inits import glorot_orthogonal
from torch_geometric.utils import softmax
from torch_scatter import scatter


class InteractionBlock(nn.Module):
    def __init__(
        self,
        emb_size_atom: int = 256,
        emb_size_edge: int = 64,
        num_heads: int = 4,
        act: nn.Module = nn.SiLU(),
        attention_act=nn.Tanhshrink(),
    ) -> None:
        super(InteractionBlock, self).__init__()
        self.mha = MultiHeadAttention(
            emb_size_atom=emb_size_atom,
            emb_size_edge=emb_size_edge,
            num_heads=num_heads,
            act=act,
            attention_act=attention_act,
        )
        self.h_down = DenseLayer(emb_size_atom, emb_size_edge, act=act)

        self.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_attr: Tensor,
        edge_index: Tensor,
        edge_aux: Tensor,
    ) -> Tensor:

        x_down = self.h_down(x)
        out, edge_attr = self.mha(
            edge_index, x=x_down, edge_attr=edge_attr, edge_aux=edge_aux
        )

        return x + out, edge_attr

    def reset_parameters(self) -> None:
        self.mha.reset_parameters()


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        emb_size_atom: int = 256,
        emb_size_edge: int = 64,
        num_heads: int = 4,
        attention_act: nn.Module = nn.Tanhshrink(),
        act: nn.Module = nn.SiLU(),
    ):
        super(MultiHeadAttention, self).__init__()

        self.attentions_lins = nn.ModuleList(
            [
                nn.Linear(
                    emb_size_edge,
                    emb_size_edge,
                    bias=False,
                )
                for _ in range(num_heads)
            ]
        )
        self.attention_act = attention_act

        self.m_dense = DenseLayer(
            emb_size_edge,
            emb_size_edge,
            bias=True,
            batch_norm=True,
            act=act,
        )
        self.dense_h = DenseLayer(
            emb_size_edge * num_heads,
            emb_size_atom,
            bias=True,
            batch_norm=True,
            act=act,
        )
        self.dense_e = DenseLayer(
            emb_size_edge,
            emb_size_edge,
            bias=True,
            batch_norm=True,
            act=act,
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.m_dense.reset_parameters()
        self.dense_e.reset_parameters()
        self.dense_h.reset_parameters()
        for att in self.attentions_lins:
            glorot_orthogonal(att.weight, scale=2.0)

    def forward(
        self,
        edge_index: Tensor,
        x: Tensor,
        edge_attr: Tensor,
        edge_aux: Tensor,
    ):
        x_j = x[edge_index[0]]  # Source node features [num_edges, num_features]
        x_i = x[edge_index[1]]  # Target node features [num_edges, num_features]

        m = self.m_dense(x_i * x_j * edge_attr)

        alpha = self.attention_act(
            (
                self.attentions_lins[0](x_i)
                * self.attentions_lins[0](x_j)
                * edge_attr
                * edge_aux
            ).mean(dim=-1, keepdim=True)
        )
        # alpha = softmax(alpha, edge_index[1], num_nodes=x.size(0))
        m_att = alpha * m

        m_att = scatter(m_att, edge_index[1], dim=0, dim_size=x.size(0), reduce="add")

        for attention_lin in self.attentions_lins[1:]:
            alpha = self.attention_act(
                (attention_lin(x_i) * attention_lin(x_j) * edge_attr * edge_aux).mean(
                    dim=-1, keepdim=True
                )
            )
            # alpha = softmax(alpha, edge_index[1], num_nodes=x.size(0))
            att = alpha * m

            att = scatter(att, edge_index[1], dim=0, dim_size=x.size(0), reduce="add")
            m_att = torch.cat(
                [
                    m_att,
                    att,
                ],
                dim=-1,
            )

        out = self.dense_h(m_att)

        e = self.dense_e(m)

        return out, e


class EmbeddingBlock(nn.Module):
    def __init__(
        self,
        emb_size_atom: int = 256,
        emb_size_edge: int = 64,
        num_radial: int = 8,
        num_residual: int = 3,
        cutoff: int = 6.0,
        rbf: dict = {"name": "gaussian"},
        envelope: dict = {"name": "polynomial", "exponent": 5},
        act: nn.Module = nn.SiLU(),
    ) -> None:
        super(EmbeddingBlock, self).__init__()
        self.act = act

        self.embedding_period = nn.Embedding(8, emb_size_atom // 2)
        self.embedding_group = nn.Embedding(19, emb_size_atom // 2)

        self.rbf = RadialBasis(
            num_radial=num_radial,
            cutoff=cutoff,
            rbf=rbf,
            envelope=envelope,
        )
        self.lin_aux = nn.Linear(num_radial, emb_size_edge, bias=False)
        self.enc_e = nn.Sequential(
            DenseLayer(num_radial, emb_size_edge),
            ResidualLayer(emb_size_edge, num_layers=num_residual, act=act),
        )
        self.enc_h = nn.Sequential(
            DenseLayer(emb_size_atom + 4, emb_size_atom),
            ResidualLayer(emb_size_atom, num_layers=num_residual, act=act),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.embedding_period.weight.data.uniform_(-sqrt(3), sqrt(3))
        self.embedding_group.weight.data.uniform_(-sqrt(3), sqrt(3))
        glorot_orthogonal(self.lin_aux.weight, scale=2.0)
        self.enc_e[0].reset_parameters()
        self.enc_e[1].reset_parameters()
        self.enc_h[0].reset_parameters()
        self.enc_h[1].reset_parameters()

    def forward(
        self,
        period: Tensor,
        group: Tensor,
        node_bin_features: Tensor,
        edge_attr: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        period_emb = self.embedding_period(period)
        group_emb = self.embedding_group(group)
        h = torch.cat([period_emb, group_emb, node_bin_features], dim=-1)

        e_rbf = self.rbf(edge_attr)

        e_aux = self.lin_aux(e_rbf)

        h_enc = self.enc_h(h)
        e_enc = self.enc_e(e_rbf)

        return h_enc, e_enc, e_aux


class OrbNet(nn.Module):
    def __init__(
        self,
        num_radial: int = 8,
        emb_size_atom: int = 256,
        emb_size_edge: int = 64,
        num_heads: int = 4,
        num_residual: int = 3,
        cutoff: int = 6.0,
        rbf: dict = {"name": "gaussian"},
        envelope: dict = {"name": "polynomial", "exponent": 5},
        act: nn.Module = nn.SiLU(),
        attention_act: nn.Module = nn.Tanhshrink(),
        num_interactions=3,
    ) -> None:
        super().__init__()

        self.emb = EmbeddingBlock(
            num_radial=num_radial,
            emb_size_atom=emb_size_atom,
            emb_size_edge=emb_size_edge,
            num_residual=num_residual,
            cutoff=cutoff,
            rbf=rbf,
            envelope=envelope,
            act=act,
        )

        self.interaction_blocks = nn.ModuleList(
            [
                InteractionBlock(
                    emb_size_atom, emb_size_edge, num_heads, act, attention_act
                )
                for _ in range(num_interactions)
            ]
        )

        self.output_blocks = nn.ModuleList(
            [OutputBlock(emb_size_atom, act) for _ in range(num_interactions + 1)]
        )

        self.reset_parameters()

    def forward(
        self,
        period: Tensor,
        group: Tensor,
        node_bin_features: Tensor,
        edge_attr: Tensor,
        edge_index: Tensor,
        batch: Tensor = None,
    ):
        # Embedding block.
        h, e, e_aux = self.emb(
            period=period,
            group=group,
            node_bin_features=node_bin_features,
            edge_attr=edge_attr,
        )
        P = self.output_blocks[0](h)

        # Interaction blocks.
        for interaction_block, output_block in zip(
            self.interaction_blocks, self.output_blocks[1:]
        ):
            h, e = interaction_block(h, e, edge_index, e_aux)
            P += output_block(h)

        energy = P.sum(dim=0) if batch is None else scatter(P, batch, dim=0)

        return energy

    def reset_parameters(self):
        self.emb.reset_parameters()
        for input_block in self.interaction_blocks:
            input_block.reset_parameters()
        for output_block in self.output_blocks:
            output_block.reset_parameters()


@registry.register_model("orbnet_native_surf")
class OrbNetWrap(OrbNet):
    def __init__(
        self,
        num_atoms,  # not used
        bond_feat_dim,  # not used
        num_targets,
        features_csv: str,
        cov_coeff: float = 1.5,
        use_pbc=True,
        regress_forces=False,
        otf_graph=False,
        num_heads: int = 4,
        num_radial: int = 8,
        emb_size_atom: int = 256,
        emb_size_edge: int = 64,
        num_residual: int = 3,
        num_interactions=3,
        cutoff: int = 6.0,
        rbf: dict = {"name": "gaussian"},
        envelope: dict = {"name": "polynomial", "exponent": 5},
        act: nn.Module = nn.SiLU(),
        attention_act: nn.Module() = nn.Tanhshrink(),
    ):
        self.num_targets = num_targets
        self.regress_forces = regress_forces
        self.use_pbc = use_pbc
        self.cutoff = cutoff
        self.otf_graph = otf_graph

        features = pd.read_csv(features_csv)

        self.periods = features["Period"].values
        self.groups = features["Group"].values
        self.cov_rad = features["Cov_radius_cordero"].values
        self.adsorb_atomic_numbers = (1, 6, 7, 8)
        self.cov_coeff = cov_coeff

        super(OrbNetWrap, self).__init__(
            num_heads=num_heads,
            num_radial=num_radial,
            emb_size_atom=emb_size_atom,
            num_residual=num_residual,
            emb_size_edge=emb_size_edge,
            cutoff=cutoff,
            rbf=rbf,
            envelope=envelope,
            act=act,
            num_interactions=num_interactions,
            attention_act=attention_act,
        )

    @conditional_grad(torch.enable_grad())
    def _forward(self, data):
        self.device = data.pos.device

        atomic_numbers = data.atomic_numbers.long()
        pos = data.pos

        if self.otf_graph:
            edge_index, cell_offsets, neighbors = radius_graph_pbc(
                data, self.cutoff, 100, data.pos.device
            )
            data.edge_index = edge_index
            data.cell_offsets = cell_offsets
            data.neighbors = neighbors

        if self.use_pbc:
            assert atomic_numbers.dim() == 1 and atomic_numbers.dtype == torch.long

            out = get_pbc_distances(
                pos,
                data.edge_index,
                data.cell,
                data.cell_offsets,
                data.neighbors,
            )

            data.edge_index = out["edge_index"]
            data.distances = out["distances"]

        else:
            data.edge_index = radius_graph(pos, r=self.cutoff, batch=data.batch)
            j, i = data.edge_index
            data.distances = (pos[j] - pos[i]).norm(dim=-1)

        period = torch.tensor(self.periods[atomic_numbers.cpu() - 1]).type_as(
            atomic_numbers
        )
        group = torch.tensor(self.groups[atomic_numbers.cpu() - 1]).type_as(
            atomic_numbers
        )
        node_bin_features = get_surf_bin_features(
            data, self.cov_rad, self.adsorb_atomic_numbers, self.cov_coeff
        )
        energy = super().forward(
            period,
            group,
            node_bin_features,
            data.distances,
            data.edge_index,
            data.batch,
        )

        return energy

    def forward(self, data):
        if self.regress_forces:
            data.pos.requires_grad_(True)
        energy = self._forward(data)

        if self.regress_forces:
            forces = -1 * (
                torch.autograd.grad(
                    energy,
                    data.pos,
                    grad_outputs=torch.ones_like(energy),
                    create_graph=True,
                )[0]
            )
            return energy, forces
        else:
            return energy

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
