# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, GraphConv, TransformerConv


class BaseGNN(torch.nn.Module):
    def __init__(self, dropout, num_layers):
        super(BaseGNN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.dropout = dropout
        self.num_layers = num_layers

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        if self.num_layers == 1:
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class SAGE(BaseGNN):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(SAGE, self).__init__(dropout, num_layers)
        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.convs.append(SAGEConv(first_channels, second_channels))


class GCN(BaseGNN):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(GCN, self).__init__(dropout, num_layers)
        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.convs.append(GCNConv(first_channels, second_channels, normalize=False))


class WSAGE(BaseGNN):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(WSAGE, self).__init__(dropout, num_layers)
        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.convs.append(GraphConv(first_channels, second_channels))


class Transformer(BaseGNN):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(Transformer, self).__init__(dropout, num_layers)
        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.convs.append(TransformerConv(first_channels, second_channels))


# =======================================================================
# [CIRURGIA ABLAÇÃO: ESPACIAL] Difusão Dinâmica (Módulo Inception)
# =======================================================================
class InceptionGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout):
        super(InceptionGNN, self).__init__()
        self.dropout = dropout
        
        # Canal A: 1 Salto
        self.branch1_conv = SAGEConv(in_channels, hidden_channels)
        self.branch1_norm = torch.nn.LayerNorm(hidden_channels)
        
        # Canal B: 2 Saltos
        self.branch2_conv1 = SAGEConv(in_channels, hidden_channels)
        self.branch2_norm1 = torch.nn.LayerNorm(hidden_channels)
        self.branch2_conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.branch2_norm2 = torch.nn.LayerNorm(hidden_channels)
        
        # Canal C: 3 Saltos
        self.branch3_conv1 = SAGEConv(in_channels, hidden_channels)
        self.branch3_norm1 = torch.nn.LayerNorm(hidden_channels)
        self.branch3_conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.branch3_norm2 = torch.nn.LayerNorm(hidden_channels)
        self.branch3_conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.branch3_norm3 = torch.nn.LayerNorm(hidden_channels)
        
        # Fuso de Concatenamento
        self.project = torch.nn.Linear(3 * hidden_channels, hidden_channels)
        self.project_norm = torch.nn.LayerNorm(hidden_channels)

    def reset_parameters(self):
        self.branch1_conv.reset_parameters()
        self.branch1_norm.reset_parameters()
        
        self.branch2_conv1.reset_parameters()
        self.branch2_norm1.reset_parameters()
        self.branch2_conv2.reset_parameters()
        self.branch2_norm2.reset_parameters()
        
        self.branch3_conv1.reset_parameters()
        self.branch3_norm1.reset_parameters()
        self.branch3_conv2.reset_parameters()
        self.branch3_norm2.reset_parameters()
        self.branch3_conv3.reset_parameters()
        self.branch3_norm3.reset_parameters()
        
        self.project.reset_parameters()
        self.project_norm.reset_parameters()

    def forward(self, x, adj_t):
        # Fluxo Paralelo A: 1-hop
        h1 = self.branch1_conv(x, adj_t)
        h1 = self.branch1_norm(h1)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        
        # Fluxo Paralelo B: 2-hops
        h2 = self.branch2_conv1(x, adj_t)
        h2 = self.branch2_norm1(h2)
        h2 = F.relu(h2)
        h2 = F.dropout(h2, p=self.dropout, training=self.training)
        
        h2 = self.branch2_conv2(h2, adj_t)
        h2 = self.branch2_norm2(h2)
        h2 = F.relu(h2)
        h2 = F.dropout(h2, p=self.dropout, training=self.training)
        
        # Fluxo Paralelo C: 3-hops
        h3 = self.branch3_conv1(x, adj_t)
        h3 = self.branch3_norm1(h3)
        h3 = F.relu(h3)
        h3 = F.dropout(h3, p=self.dropout, training=self.training)
        
        h3 = self.branch3_conv2(h3, adj_t)
        h3 = self.branch3_norm2(h3)
        h3 = F.relu(h3)
        h3 = F.dropout(h3, p=self.dropout, training=self.training)
        
        h3 = self.branch3_conv3(h3, adj_t)
        h3 = self.branch3_norm3(h3)
        h3 = F.relu(h3)
        h3 = F.dropout(h3, p=self.dropout, training=self.training)
        
        # A Mágica do Inception
        out = torch.cat([h1, h2, h3], dim=-1)
        out = self.project(out)
        out = self.project_norm(out) # Normaliza a fusão
        out = F.relu(out)            # Aplica ativação final
        
        return out
# =======================================================================

class MLPPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(MLPPredictor, self).__init__()
        self.lins = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList() # <- Normalizadores
        
        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.lins.append(torch.nn.Linear(first_channels, second_channels))
            
            if i < num_layers - 1:
                self.norms.append(torch.nn.LayerNorm(second_channels)) # <- Normalizadores
                
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.norms[i](x) # <- Normaliza antes do ReLU
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


class MLPSinglePredictor(torch.nn.Module):
    """MLP predictor that accepts a single input tensor (for concatenated features)."""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(MLPSinglePredictor, self).__init__()
        self.lins = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList() # <- Adicionamos os normalizadores aqui
        
        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.lins.append(torch.nn.Linear(first_channels, second_channels))
            
            # Adiciona LayerNorm para todas as camadas exceto a última
            if i < num_layers - 1:
                self.norms.append(torch.nn.LayerNorm(second_channels))
                
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()

    def forward(self, x):
        # x is a single tensor [batch, in_channels]
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.norms[i](x) # <- Normaliza antes do ReLU
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # A última camada NUNCA leva ReLU ou LayerNorm, ela cospe a probabilidade crua para a Loss
        x = self.lins[-1](x)
        return x


class MLPCatPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(MLPCatPredictor, self).__init__()
        self.lins = torch.nn.ModuleList()
        in_channels = 2 * in_channels
        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.lins.append(torch.nn.Linear(first_channels, second_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x1 = torch.cat([x_i, x_j], dim=-1)
        x2 = torch.cat([x_j, x_i], dim=-1)
        for lin in self.lins[:-1]:
            x1, x2 = lin(x1), lin(x2)
            x1, x2 = F.relu(x1), F.relu(x2)
            x1 = F.dropout(x1, p=self.dropout, training=self.training)
            x2 = F.dropout(x2, p=self.dropout, training=self.training)
        x1 = self.lins[-1](x1)
        x2 = self.lins[-1](x2)
        x = (x1 + x2)/2
        return x


class MLPDotPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout):
        super(MLPDotPredictor, self).__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        for lin in self.lins:
            x_i, x_j = lin(x_i), lin(x_j)
            x_i, x_j = F.relu(x_i), F.relu(x_j)
            x_i, x_j = F.dropout(x_i, p=self.dropout, training=self.training), \
                F.dropout(x_j, p=self.dropout, training=self.training)
        x = torch.sum(x_i * x_j, dim=-1)
        return x


class MLPBilPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout):
        super(MLPBilPredictor, self).__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.bilin = torch.nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        self.bilin.reset_parameters()

    def forward(self, x_i, x_j):
        for lin in self.lins:
            x_i, x_j = lin(x_i), lin(x_j)
            x_i, x_j = F.relu(x_i), F.relu(x_j)
            x_i, x_j = F.dropout(x_i, p=self.dropout, training=self.training), \
                F.dropout(x_j, p=self.dropout, training=self.training)
        x = torch.sum(self.bilin(x_i) * x_j, dim=-1)
        return x


class DotPredictor(torch.nn.Module):
    def __init__(self):
        super(DotPredictor, self).__init__()

    def reset_parameters(self):
        return

    def forward(self, x_i, x_j):
        x = torch.sum(x_i * x_j, dim=-1)
        return x


class BilinearPredictor(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(BilinearPredictor, self).__init__()
        self.bilin = torch.nn.Linear(hidden_channels, hidden_channels, bias=False)

    def reset_parameters(self):
        self.bilin.reset_parameters()

    def forward(self, x_i, x_j):
        x = torch.sum(self.bilin(x_i) * x_j, dim=-1)
        return x


