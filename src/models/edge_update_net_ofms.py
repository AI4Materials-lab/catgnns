import pandas as pd
import torch
from ocpmodels.models.gemnet.layers.radial_basis import RadialBasis
from torch import nn
from torch_geometric.nn import MessagePassing, radius_graph
from torch_geometric.nn.inits import glorot_orthogonal
from torch_geometric.nn.models.schnet import SchNet, ShiftedSoftplus
from torch_scatter import scatter


class EdgeUpdateBlock(nn.Module):
    def __init__(self, in_features, C):
        super(EdgeUpdateBlock, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 2 * C),
            ShiftedSoftplus(),
            nn.Linear(2 * C, C),
        )

    def forward(self, h, edge_attr, edge_index):
        h1 = h[edge_index[0, :]]
        h2 = h[edge_index[1, :]]
        concat = torch.cat((h1, h2, edge_attr), dim=1)
        x = self.mlp(concat)
        return x

    def reset_parameters(self):
        # glorot_orthogonal(self.mlp[0].weight, scale=2.0)
        self.mlp[0].bias.data.fill_(0)
        # glorot_orthogonal(self.mlp[2].weight, scale=2.0)
        self.mlp[2].bias.data.fill_(0)


class InteractionBlock(MessagePassing):
    def __init__(self, C, C_edge):
        super(InteractionBlock, self).__init__()
        self.fc1 = nn.Linear(C, C_edge)
        self.mlp = nn.Sequential(
            nn.Linear(C_edge, C_edge),
            ShiftedSoftplus(),
            nn.Linear(C_edge, C_edge),
            ShiftedSoftplus(),
        )
        self.mlp1 = nn.Sequential(
            nn.Linear(C_edge, C_edge),
            ShiftedSoftplus(),
            nn.Linear(C_edge, C),
        )

    def forward(self, x, edge_attr, edge_index):
        h = self.fc1(x)
        edge = self.mlp(edge_attr)
        m = self.propagate(edge_index, x=h, edge_attr=edge)
        m = self.mlp1(m)

        return x + m

    def message(self, x_j, edge_attr):
        return x_j * edge_attr

    def reset_parameters(self):
        # glorot_orthogonal(self.fc1.weight, scale=2.0)
        self.fc1.bias.data.fill_(0)
        # glorot_orthogonal(self.mlp[0].weight, scale=2.0)
        self.mlp[0].bias.data.fill_(0)
        # glorot_orthogonal(self.mlp[2].weight, scale=2.0)
        self.mlp[2].bias.data.fill_(0)
        # glorot_orthogonal(self.mlp1[0].weight, scale=2.0)
        self.mlp1[0].bias.data.fill_(0)
        # glorot_orthogonal(self.mlp1[2].weight, scale=2.0)
        self.mlp1[2].bias.data.fill_(0)


class GaussianSmearing(torch.nn.Module):
    def __init__(self, num_gaussians=150, coeff=0.1, m_min=0):
        super(GaussianSmearing, self).__init__()
        offset = torch.arange(0, num_gaussians)
        self.coeff = coeff
        self.inv_coeff = self.coeff**0.5
        self.m_min = m_min
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - (-self.m_min + self.coeff * self.offset.view(1, -1))
        return torch.exp(-self.inv_coeff * torch.pow(dist, 2))


class EdgeUpdateNet(nn.Module):
    def __init__(
        self,
        C=64,
        num_interactions=3,
        num_gaussians=128,
        cutoff=10.0,
        readout="add",
    ):
        super(EdgeUpdateNet, self).__init__()

        self.readout = readout
        self.cutoff = cutoff

        self.embedding = nn.Embedding(95, C)
        self.distance_expansion = RadialBasis(
            num_gaussians,
            cutoff=cutoff,
            rbf={"name": "gaussian"},
            envelope={"name": "polynomial", "exponent": 5},
        )

        self.interactions = nn.ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(C + 1024, C)
            self.interactions.append(block)

        self.edge_updates = nn.ModuleList()
        block = EdgeUpdateBlock(num_gaussians + 2 * (C + 1024), C)
        self.edge_updates.append(block)

        for _ in range(num_interactions - 1):
            block = EdgeUpdateBlock(C * 3 + 2 * 1024, C)
            self.edge_updates.append(block)

        self.mlp = nn.Sequential(
            nn.Linear(C + 1024, C // 2),
            ShiftedSoftplus(),
            nn.Linear(C // 2, 1),
        )
        # TODO fix none
        self.reset_parameters()

    def forward(self, z, edge_index, distances, ofms, batch=None):
        assert z.dtype == torch.long

        batch = torch.zeros_like(z) if batch is None else batch

        h = self.embedding(z)
        h = torch.cat([h, ofms], dim=-1)
        edge_attr = self.distance_expansion(distances)

        for edge_update, interaction in zip(self.edge_updates, self.interactions):
            edge_attr = edge_update(h, edge_attr, edge_index)
            h = interaction(h, edge_attr, edge_index)
        h = self.mlp(h)

        out = scatter(h, batch, dim=0, reduce=self.readout)

        return out

    def reset_parameters(self):
        # self.embedding.reset_parameters()
        for interaction in self.interactions:
            interaction.reset_parameters()
        for edge_update in self.edge_updates:
            edge_update.reset_parameters()
        # glorot_orthogonal(self.mlp[0].weight, scale=2.0)
        self.mlp[0].bias.data.fill_(0)
        self.mlp[2].weight.data.fill_(0)
        self.mlp[2].bias.data.fill_(0)


from ocpmodels.common.registry import registry
from ocpmodels.common.utils import conditional_grad, get_pbc_distances, radius_graph_pbc


@registry.register_model("edgeupdatenet_ofms")
class EdgeUpdateNetWrap(EdgeUpdateNet):
    def __init__(
        self,
        num_atoms,  # not used
        bond_feat_dim,  # not used
        num_targets,
        use_pbc=True,
        otf_graph=False,
        regress_forces=True,
        hidden_channels=128,
        num_interactions=6,
        num_gaussians=150,
        cutoff=10.0,
        readout="add",
    ):
        self.num_targets = num_targets
        self.regress_forces = regress_forces
        self.use_pbc = use_pbc
        self.cutoff = cutoff
        self.otf_graph = otf_graph

        super(EdgeUpdateNetWrap, self).__init__(
            C=hidden_channels,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            readout=readout,
        )

    @conditional_grad(torch.enable_grad())
    def _forward(self, data):
        z = data.atomic_numbers.long()
        pos = data.pos
        batch = data.batch

        if self.otf_graph:
            edge_index, cell_offsets, neighbors = radius_graph_pbc(
                data, self.cutoff, 50, data.pos.device
            )
            data.edge_index = edge_index
            data.cell_offsets = cell_offsets
            data.neighbors = neighbors

        if self.use_pbc:
            out = get_pbc_distances(
                pos,
                data.edge_index,
                data.cell,
                data.cell_offsets,
                data.neighbors,
            )

            edge_index = out["edge_index"]
            distances = out["distances"]
        else:
            edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
            row, col = edge_index
            distances = (pos[row] - pos[col]).norm(dim=-1)

        ofms = data.ofms.type_as(data.pos)

        energy = super(EdgeUpdateNetWrap, self).forward(
            z, edge_index, distances, ofms, batch
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
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
