# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from plnlp.layer import *
from plnlp.loss import *
from plnlp.utils import *
from torch_geometric.utils import dropout_adj
from torch_sparse import SparseTensor

# =======================================================================
# [CIRURGIA ABLAÇÃO: TEMPORAL] Codificador de Bochner (Tempo Contínuo)
# =======================================================================
class TimeEncoder(nn.Module):
    def __init__(self, time_dim):
        super(TimeEncoder, self).__init__()
        self.time_dim = time_dim
        # Frequências treináveis (Bochner)
        self.w = nn.Linear(1, time_dim // 2)

    def forward(self, t):
        # t shape: [batch_size, 1]
        t = t.float()
        freqs = self.w(t)
        # Teorema de Bochner: concatenação de senos e cossenos
        time_emb = torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)
        return time_emb

# =======================================================================
# [CIRURGIA ABLAÇÃO: PREDITOR] Preditor Personalizado (Hadamard + Tempo + AA-DC)
# =======================================================================
class AblationPredictor(nn.Module):
    def __init__(self, hidden_channels, time_dim, num_layers, dropout, use_temporal, use_heuristic):
        super(AblationPredictor, self).__init__()
        self.use_temporal = use_temporal
        self.use_heuristic = use_heuristic
        
        # Calcula a dimensão de entrada do MLP baseado nas flags ativadas
        input_dim = hidden_channels  
        
        if use_temporal:
            self.time_encoder = TimeEncoder(time_dim)
            input_dim += time_dim
            
        if use_heuristic:
            input_dim += 1 # +1 para o valor escalar do AA-DC
            
        # MLP de Decisão Final (usa um MLP que aceita um tensor concatenado)
        self.mlp = MLPSinglePredictor(input_dim, hidden_channels, 1, num_layers, dropout)

    def forward(self, z_u, z_v, edge_dt=None, edge_aadc=None):
        # 1. Sinal Estrutural (Hadamard)
        features = [z_u * z_v]
        
        # 2. Sinal Temporal (Bochner)
        if self.use_temporal and edge_dt is not None:
            time_emb = self.time_encoder(edge_dt)
            features.append(time_emb)
            
        # 3. Sinal Heurístico (AA-DC)
        if self.use_heuristic and edge_aadc is not None:
            features.append(edge_aadc.unsqueeze(-1)) # Transforma um número solto num vetor [batch, 1]
            
        # Concatena os sinais ativados e toma a decisão
        z_pred = torch.cat(features, dim=-1)
        return self.mlp(z_pred)


class BaseModel(object):
    def __init__(self, lr, dropout, grad_clip_norm, gnn_num_layers, mlp_num_layers, emb_hidden_channels,
                 gnn_hidden_channels, mlp_hidden_channels, num_nodes, num_node_feats, gnn_encoder_name,
                 predictor_name, loss_func, optimizer_name, device, use_node_feats, train_node_emb,
                 pretrain_emb=None, 
                 # [CIRURGIA ABLAÇÃO: FLAGS INJETADAS AQUI]
                 spatial_mode='base', use_temporal=False, use_heuristic=False):
        
        self.loss_func_name = loss_func
        self.num_nodes = num_nodes
        self.num_node_feats = num_node_feats
        self.use_node_feats = use_node_feats
        self.train_node_emb = train_node_emb
        self.clip_norm = grad_clip_norm
        self.device = device
        
        # Salva o estado da ablação
        self.spatial_mode = spatial_mode.lower()
        self.use_temporal = use_temporal
        self.use_heuristic = use_heuristic

        # Input Layer
        self.input_channels, self.emb = create_input_layer(num_nodes=num_nodes,
                                                           num_node_feats=num_node_feats,
                                                           hidden_channels=emb_hidden_channels,
                                                           use_node_feats=use_node_feats,
                                                           train_node_emb=train_node_emb,
                                                           pretrain_emb=pretrain_emb)
        if self.emb is not None:
            self.emb = self.emb.to(device)

        # GNN Layer
        # O encoder_name vai ditar se é SAGE, GCN ou o novo INCEPTION
        self.encoder = create_gnn_layer(input_channels=self.input_channels,
                                        hidden_channels=gnn_hidden_channels,
                                        num_layers=gnn_num_layers,
                                        dropout=dropout,
                                        encoder_name=gnn_encoder_name).to(device)

        # [CIRURGIA ABLAÇÃO: PREDITOR] Substitui o preditor padrão pelo nosso Preditor de Ablação
        if self.use_temporal or self.use_heuristic:
            self.predictor = AblationPredictor(hidden_channels=mlp_hidden_channels, 
                                               time_dim=128, # Configuração fixa de dimensão de tempo
                                               num_layers=mlp_num_layers, 
                                               dropout=dropout,
                                               use_temporal=self.use_temporal,
                                               use_heuristic=self.use_heuristic).to(device)
        else:
            self.predictor = create_predictor_layer(input_channels=gnn_hidden_channels,
                                                    hidden_channels=mlp_hidden_channels,
                                                    num_layers=mlp_num_layers,
                                                    dropout=dropout,
                                                    predictor_name=predictor_name).to(device)

        # Parameters and Optimizer
        self.para_list = list(self.encoder.parameters()) + list(self.predictor.parameters())
        if self.emb is not None:
            self.para_list += list(self.emb.parameters())

        if optimizer_name == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.para_list, lr=lr)
        elif optimizer_name == 'SGD':
            self.optimizer = torch.optim.SGD(self.para_list, lr=lr, momentum=0.9, weight_decay=1e-5, nesterov=True)
        else:
            self.optimizer = torch.optim.Adam(self.para_list, lr=lr)

    def param_init(self):
        self.encoder.reset_parameters()
        if hasattr(self.predictor, 'mlp'): # Caso seja o AblationPredictor
            self.predictor.mlp.reset_parameters()
        else:
            self.predictor.reset_parameters()
            
        if self.emb is not None:
            torch.nn.init.xavier_uniform_(self.emb.weight)

    def create_input_feat(self, data):
        if self.use_node_feats:
            input_feat = data.x.to(self.device)
            if self.train_node_emb:
                input_feat = torch.cat([self.emb.weight, input_feat], dim=-1)
        else:
            input_feat = self.emb.weight
        return input_feat

    def calculate_loss(self, pos_out, neg_out, num_neg, margin=None):
        if self.loss_func_name == 'CE':
            loss = ce_loss(pos_out, neg_out)
        elif self.loss_func_name == 'InfoNCE':
            loss = info_nce_loss(pos_out, neg_out, num_neg)
        elif self.loss_func_name == 'LogRank':
            loss = log_rank_loss(pos_out, neg_out, num_neg)
        elif self.loss_func_name == 'HingeAUC':
            loss = hinge_auc_loss(pos_out, neg_out, num_neg)
        elif self.loss_func_name == 'AdaAUC' and margin is not None:
            loss = adaptive_auc_loss(pos_out, neg_out, num_neg, margin)
        elif self.loss_func_name == 'WeightedAUC' and margin is not None:
            loss = weighted_auc_loss(pos_out, neg_out, num_neg, margin)
        elif self.loss_func_name == 'AdaHingeAUC' and margin is not None:
            loss = adaptive_hinge_auc_loss(pos_out, neg_out, num_neg, margin)
        elif self.loss_func_name == 'WeightedHingeAUC' and margin is not None:
            loss = weighted_hinge_auc_loss(pos_out, neg_out, num_neg, margin)
        else:
            loss = auc_loss(pos_out, neg_out, num_neg)
        return loss

    def train(self, data, split_edge, batch_size, neg_sampler_name, num_neg):
        self.encoder.train()
        self.predictor.train()

        # Captura a melhor representação disponível para alimentar o amostrador adversarial
        if self.use_node_feats:
            feats_adv = data.x.to(self.device)
        elif self.train_node_emb and self.emb is not None:
            # Usa os próprios embeddings topológicos se as features textuais estiverem desligadas
            feats_adv = self.emb.weight.to(self.device)
        else:
            feats_adv = None

        pos_train_edge, neg_train_edge = get_pos_neg_edges('train', split_edge,
                                                           edge_index=data.edge_index,
                                                           num_nodes=self.num_nodes,
                                                           neg_sampler_name=neg_sampler_name,
                                                           num_neg=num_neg,
                                                           node_feats=feats_adv) # <-- Passamos a representação capturada

        pos_train_edge, neg_train_edge = pos_train_edge.to(self.device), neg_train_edge.to(self.device)
        
        if 'weight' in split_edge['train']:
            edge_weight_margin = split_edge['train']['weight'].to(self.device)
        else:
            edge_weight_margin = None

        total_loss = total_examples = 0

        # ==============================================================
        # [CURA DO OVERFITTING: EDGE DROPOUT] 
        # Removemos 15% das arestas do grafo para forçar a GNN a generalizar
        # ==============================================================
        edge_index_dropped, _ = dropout_adj(data.edge_index, p=0.15, force_undirected=True, training=self.encoder.training)
        row, col = edge_index_dropped
        adj_t_dropped = SparseTensor(row=col, col=row, sparse_sizes=(self.num_nodes, self.num_nodes)).to(self.device)
        # ==============================================================

        for perm in DataLoader(range(pos_train_edge.size(0)), batch_size, shuffle=True):
            self.optimizer.zero_grad()

            input_feat = self.create_input_feat(data)
            
            # ATENÇÃO AQUI: Passamos o adj_t_dropped (grafo furado) em vez do data.adj_t completo
            h = self.encoder(input_feat, adj_t_dropped)
            
            pos_edge = pos_train_edge[perm].t()
            neg_edge = torch.reshape(neg_train_edge[perm], (-1, 2)).t()

            # ==============================================================
            # O MOTOR DINÂMICO: Extração Justa de Dados para o Lote Atual
            # ==============================================================
            
            # 1. A Heurística AA-DC (Calculada ao vivo para Verdadeiros e Falsos)
            if self.use_heuristic:
                # Utilizamos a função get_batch_aadc (que criámos no utils.py)
                pos_aadc = get_batch_aadc(data.aadc_matrix, pos_edge).to(self.device)
                neg_aadc = get_batch_aadc(data.aadc_matrix, neg_edge).to(self.device)
            else:
                pos_aadc = None
                neg_aadc = None

            # 2. O Tempo Contínuo (Bochner)
            if self.use_temporal:
                ano_base = 2019 # O limite do OGB Collab
                
                # [CORREÇÃO ABLAÇÃO: ALINHAMENTO DE DIMENSÕES]
                train_years = split_edge['train']['year']
                
                # Se o DataLoader tem mais arestas do que anos disponíveis, significa que 
                # a flag 'use_valedges_as_input=True' juntou os grafos no plnlp_sign.py.
                if train_years.size(0) < pos_train_edge.size(0):
                    if 'year' in split_edge['valid']:
                        # Replicamos a exata ordem de concatenação feita no script principal: [valid, train]
                        valid_years = split_edge['valid']['year'].to(train_years.device)
                        train_years = torch.cat([valid_years, train_years], dim=0)
                
                # Delta-T para as colaborações reais (Treino) usando o array alinhado
                pos_anos = train_years[perm].to(self.device)
                pos_dt = (ano_base - pos_anos).view(-1, 1).float()
                
                # Sorteia anos aleatórios do conjunto alinhado para os negativos
                todos_anos = train_years.to(self.device)
                idx_aleatorios = torch.randint(0, todos_anos.size(0), (neg_edge.size(1),), device=self.device)
                neg_anos_sorteados = todos_anos[idx_aleatorios]
                
                neg_dt = (ano_base - neg_anos_sorteados).view(-1, 1).float()
            else:
                pos_dt = None
                neg_dt = None
                
            # ==============================================================
            # A CHAMADA AO PREDITOR (Sem leakage de dados)
            # ==============================================================
            if self.use_temporal or self.use_heuristic:
                # Os dados entram no forward da classe AblationPredictor aqui!
                pos_out = self.predictor(h[pos_edge[0]], h[pos_edge[1]], edge_dt=pos_dt, edge_aadc=pos_aadc)
                neg_out = self.predictor(h[neg_edge[0]], h[neg_edge[1]], edge_dt=neg_dt, edge_aadc=neg_aadc)
            else:
                pos_out = self.predictor(h[pos_edge[0]], h[pos_edge[1]])
                neg_out = self.predictor(h[neg_edge[0]], h[neg_edge[1]])

            # (O resto da função continua igual: cálculo da loss, backward, step)
            weight_margin = edge_weight_margin[perm] if edge_weight_margin is not None else None

            loss = self.calculate_loss(pos_out, neg_out, num_neg, margin=weight_margin)
            loss.backward()

            if self.clip_norm >= 0:
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip_norm)
                torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), self.clip_norm)

            self.optimizer.step()

            num_examples = pos_out.size(0)
            total_loss += loss.item() * num_examples
            total_examples += num_examples

        return total_loss / total_examples
    @torch.no_grad()
    # [CIRURGIA ABLAÇÃO: ARGS DO BATCH_PREDICT] 
    def batch_predict(self, h, edges, batch_size, edge_times=None, edge_aadcs=None):
        preds = []
        for perm in DataLoader(range(edges.size(0)), batch_size):
            edge = edges[perm].t()
            
            dt_batch = edge_times[perm].to(self.device) if edge_times is not None else None
            aadc_batch = edge_aadcs[perm].to(self.device) if edge_aadcs is not None else None
            
            if self.use_temporal or self.use_heuristic:
                preds += [self.predictor(h[edge[0]], h[edge[1]], edge_dt=dt_batch, edge_aadc=aadc_batch).squeeze().cpu()]
            else:
                preds += [self.predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
                
        pred = torch.cat(preds, dim=0)
        return pred

    @torch.no_grad()
    def test(self, data, split_edge, batch_size, evaluator, eval_metric):
        self.encoder.eval()
        self.predictor.eval()

        input_feat = self.create_input_feat(data)
        h = self.encoder(input_feat, data.adj_t)
        
        mean_h = torch.mean(h, dim=0, keepdim=True)
        h = torch.cat([h, mean_h], dim=0)

        pos_valid_edge = split_edge['valid']['edge']
        neg_valid_edge = split_edge['valid']['edge_neg']
        
        pos_test_edge = split_edge['test']['edge']
        neg_test_edge = split_edge['test']['edge_neg']
        
        pos_valid_edge, neg_valid_edge = pos_valid_edge.to(self.device), neg_valid_edge.to(self.device)
        pos_test_edge, neg_test_edge = pos_test_edge.to(self.device), neg_test_edge.to(self.device)

        # [CIRURGIA ABLAÇÃO: TESTE DINÂMICO] Busca os dados de tempo e heurística em tempo real!
        ano_base = 2019

        # --- TEMPO (Bochner) ---
        # --- TEMPO (Bochner) NO TESTE ---
        # --- TEMPO (Bochner) NO TESTE ---
        if self.use_temporal:
            # ==========================================
            # VALIDAÇÃO
            # ==========================================
            if 'year' in split_edge['valid']:
                valid_years = split_edge['valid']['year'].to(self.device)
                v_pos_dt = (ano_base - valid_years).view(-1, 1).float()
                
                # Sorteia anos reais da validação para camuflar os negativos
                idx_v = torch.randint(0, valid_years.size(0), (neg_valid_edge.size(0),), device=self.device)
                v_neg_dt = (ano_base - valid_years[idx_v]).view(-1, 1).float()
            else:
                v_pos_dt = torch.zeros(pos_valid_edge.size(0), 1, device=self.device).float()
                v_neg_dt = torch.zeros(neg_valid_edge.size(0), 1, device=self.device).float()

            # ==========================================
            # TESTE
            # ==========================================
            if 'year' in split_edge['test']:
                test_years = split_edge['test']['year'].to(self.device)
                t_pos_dt = (ano_base - test_years).view(-1, 1).float()
                
                # Sorteia anos reais do teste para camuflar os negativos
                idx_t = torch.randint(0, test_years.size(0), (neg_test_edge.size(0),), device=self.device)
                t_neg_dt = (ano_base - test_years[idx_t]).view(-1, 1).float()
            else:
                t_pos_dt = torch.zeros(pos_test_edge.size(0), 1, device=self.device).float()
                t_neg_dt = torch.zeros(neg_test_edge.size(0), 1, device=self.device).float()
        else:
            v_pos_dt = v_neg_dt = t_pos_dt = t_neg_dt = None

        # --- HEURÍSTICA (AA-DC) ---
        if self.use_heuristic:
            # pos_valid_edge.t() converte de [N, 2] para [2, N] que o nosso get_batch_aadc precisa
            v_pos_aadc = get_batch_aadc(data.aadc_matrix, pos_valid_edge.t())
            v_neg_aadc = get_batch_aadc(data.aadc_matrix, neg_valid_edge.t())
            t_pos_aadc = get_batch_aadc(data.aadc_matrix, pos_test_edge.t())
            t_neg_aadc = get_batch_aadc(data.aadc_matrix, neg_test_edge.t())
        else:
            v_pos_aadc = v_neg_aadc = t_pos_aadc = t_neg_aadc = None

        pos_valid_pred = self.batch_predict(h, pos_valid_edge, batch_size, edge_times=v_pos_dt, edge_aadcs=v_pos_aadc)
        neg_valid_pred = self.batch_predict(h, neg_valid_edge, batch_size, edge_times=v_neg_dt, edge_aadcs=v_neg_aadc)

        # Re-encode para o teste (padrão do OGB)
        h = self.encoder(input_feat, data.adj_t)
        mean_h = torch.mean(h, dim=0, keepdim=True)
        h = torch.cat([h, mean_h], dim=0)

        pos_test_pred = self.batch_predict(h, pos_test_edge, batch_size, edge_times=t_pos_dt, edge_aadcs=t_pos_aadc)
        neg_test_pred = self.batch_predict(h, neg_test_edge, batch_size, edge_times=t_neg_dt, edge_aadcs=t_neg_aadc)

        if eval_metric == 'hits':
            results = evaluate_hits(evaluator, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)
        else:
            results = evaluate_mrr(evaluator, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)

        return results


def adjust_lr(optimizer, progress, base_lr):
    """Adjust learning rate based on progress (0.0 -> 1.0).
    Simple linear decay: new_lr = base_lr * (1 - progress).
    Updates optimizer in-place and returns new_lr.
    """
    progress = max(0.0, min(1.0, float(progress)))
    new_lr = base_lr * (1.0 - progress)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return new_lr


def create_input_layer(num_nodes, num_node_feats, hidden_channels, use_node_feats=True,
                       train_node_emb=False, pretrain_emb=None):
    emb = None
    if use_node_feats:
        input_dim = num_node_feats
        if train_node_emb:
            emb = torch.nn.Embedding(num_nodes, hidden_channels)
            input_dim += hidden_channels
        elif pretrain_emb is not None and pretrain_emb != '':
            weight = torch.load(pretrain_emb)
            emb = torch.nn.Embedding.from_pretrained(weight)
            input_dim += emb.weight.size(1)
    else:
        if pretrain_emb is not None and pretrain_emb != '':
            weight = torch.load(pretrain_emb)
            emb = torch.nn.Embedding.from_pretrained(weight)
            input_dim = emb.weight.size(1)
        else:
            emb = torch.nn.Embedding(num_nodes, hidden_channels)
            input_dim = hidden_channels
    return input_dim, emb


def create_gnn_layer(input_channels, hidden_channels, num_layers, dropout=0, encoder_name='SAGE'):
    # [CIRURGIA ABLAÇÃO: ESPACIAL] Adicionado o suporte ao módulo INCEPTION
    if encoder_name.upper() == 'GCN':
        return GCN(input_channels, hidden_channels, hidden_channels, num_layers, dropout)
    elif encoder_name.upper() == 'WSAGE':
        return WSAGE(input_channels, hidden_channels, hidden_channels, num_layers, dropout)
    elif encoder_name.upper() == 'TRANSFORMER':
        return Transformer(input_channels, hidden_channels, hidden_channels, num_layers, dropout)
    elif encoder_name.upper() == 'INCEPTION':
        return InceptionGNN(input_channels, hidden_channels, num_layers, dropout) # <- Vamos criar no layer.py!
    else:
        return SAGE(input_channels, hidden_channels, hidden_channels, num_layers, dropout)


def create_predictor_layer(input_channels, hidden_channels, num_layers, dropout=0, predictor_name='MLP'):
    predictor_name = predictor_name.upper()
    if predictor_name == 'DOT':
        return DotPredictor()
    elif predictor_name == 'BIL':
        return BilinearPredictor(hidden_channels)
    elif predictor_name == 'MLP':
        return MLPPredictor(input_channels, hidden_channels, 1, num_layers, dropout)
    elif predictor_name == 'MLPDOT':
        return MLPDotPredictor(input_channels, 1, num_layers, dropout)
    elif predictor_name == 'MLPBIL':
        return MLPBilPredictor(input_channels, 1, num_layers, dropout)
    elif predictor_name == 'MLPCAT':
        return MLPCatPredictor(input_channels, hidden_channels, 1, num_layers, dropout)