# -*- coding: utf-8 -*-
import torch
import numpy as np
from plnlp.negative_sample import global_neg_sample, global_perm_neg_sample, local_neg_sample
from torch_sparse import SparseTensor


def get_pos_neg_edges(split, split_edge, edge_index=None, num_nodes=None, neg_sampler_name=None, num_neg=None):
    if 'edge' in split_edge['train']:
        pos_edge = split_edge[split]['edge']
    elif 'source_node' in split_edge['train']:
        source = split_edge[split]['source_node']
        target = split_edge[split]['target_node']
        pos_edge = torch.stack([source, target]).t()

    if split == 'train':
        if neg_sampler_name == 'local':
            neg_edge = local_neg_sample(
                pos_edge,
                num_nodes=num_nodes,
                num_neg=num_neg)
        elif neg_sampler_name == 'global':
            neg_edge = global_neg_sample(
                edge_index,
                num_nodes=num_nodes,
                num_samples=pos_edge.size(0),
                num_neg=num_neg)
            
        # [CIRURGIA ABLAÇÃO: CORRIGIDO]
        elif neg_sampler_name == 'adversarial':
            from plnlp.negative_sample import adversarial_neg_sample
            neg_edge = adversarial_neg_sample(
                edge_index, 
                num_nodes=num_nodes, 
                num_samples=pos_edge.size(0), # Correção: mapeado dinamicamente
                num_neg=num_neg
            )
            
        else:
            neg_edge = global_perm_neg_sample(
                edge_index,
                num_nodes=num_nodes,
                num_samples=pos_edge.size(0),
                num_neg=num_neg)
   
    else:
        if 'edge' in split_edge['train']:
            neg_edge = split_edge[split]['edge_neg']
        elif 'source_node' in split_edge['train']:
            target_neg = split_edge[split]['target_node_neg']
            neg_per_target = target_neg.size(1)
            neg_edge = torch.stack([source.repeat_interleave(neg_per_target),
                                    target_neg.view(-1)]).t()
    return pos_edge, neg_edge


def evaluate_hits(evaluator, pos_val_pred, neg_val_pred,
                  pos_test_pred, neg_test_pred):
    results = {}
    for K in [20, 50, 100]:
        evaluator.K = K
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_val_pred,
            'y_pred_neg': neg_val_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (valid_hits, test_hits)

    return results


def evaluate_mrr(evaluator, pos_val_pred, neg_val_pred,
                 pos_test_pred, neg_test_pred):
    neg_val_pred = neg_val_pred.view(pos_val_pred.shape[0], -1)
    neg_test_pred = neg_test_pred.view(pos_test_pred.shape[0], -1)
    results = {}
    valid_mrr = evaluator.eval({
        'y_pred_pos': pos_val_pred,
        'y_pred_neg': neg_val_pred,
    })['mrr_list'].mean().item()

    test_mrr = evaluator.eval({
        'y_pred_pos': pos_test_pred,
        'y_pred_neg': neg_test_pred,
    })['mrr_list'].mean().item()

    results['MRR'] = (valid_mrr, test_mrr)

    return results


def gcn_normalization(adj_t):
    adj_t = adj_t.set_diag()
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
    return adj_t


def adj_normalization(adj_t):
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-1)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t
    return adj_t


def generate_neg_dist_table(num_nodes, adj_t, power=0.75, table_size=1e8):
    table_size = int(table_size)
    adj_t = adj_t.set_diag()
    node_degree = adj_t.sum(dim=1).to(torch.float)
    node_degree = node_degree.pow(power)

    norm = float((node_degree).sum())  # float is faster than tensor when visited
    node_degree = node_degree.tolist()  # list has fastest visit speed
    sample_table = np.zeros(table_size, dtype=np.int32)
    p = 0
    i = 0
    for j in range(num_nodes):
        p += node_degree[j] / norm
        while i < table_size and float(i) / float(table_size) < p:
            sample_table[i] = j
            i += 1
    sample_table = torch.from_numpy(sample_table)
    return sample_table

# =======================================================================
# [MÓDULO DE CÁLCULO DINÂMICO PARA ABLAÇÃO] (Adicione no final de utils.py)
# =======================================================================
def precompute_aadc_matrix(adj_t):
    """
    Pré-calcula a Matriz Esparsa de Adamic-Adar + Degree Centrality.
    Isso permite que você pegue o AA-DC de qualquer par falso gerado em O(1).
    """
    row, col, _ = adj_t.coo()
    deg = adj_t.sum(dim=1).to(torch.float)
    
    # Lógica Adamic-Adar clássica: 1 / log(grau)
    deg_log = torch.log(deg)
    deg_log[deg_log < 1e-10] = 1.0 # Prevenção de divisão por zero
    weight = 1.0 / deg_log
    weight[deg <= 1] = 0.0 # Nós com 0 ou 1 ligação não são pontes de coautoria válidas
    
    # Cria uma matriz normalizada pelo AA
    adj_w = SparseTensor(row=row, col=col, value=weight[col], sparse_sizes=adj_t.sparse_sizes())
    
    # A matriz AA é literalmente A multiplicada pela A ponderada
    aadc_matrix = adj_t.matmul(adj_w.t())
    return aadc_matrix

def get_batch_aadc(aadc_matrix, edge_index):
    """ Extrai rapidamente os valores da matriz esparsa para o lote atual (Verdadeiros ou Falsos) """
    row, col = edge_index[0], edge_index[1]
    # No PyTorch Geometric/Sparse, extrair valores discretos pode ser feito mapeando:
    # Se aadc_matrix for muito densa, essa operação requer cuidado, 
    # mas para o collab, é bem esparso.
    
    # Obter os scores via SparseTensor pode exigir converter para COO momentaneamente
    # ou usar a função get_value se disponível. Uma abordagem segura e rápida:
    out = []
    try:
        # Tenta extrair via COO da SparseTensor
        a_row, a_col, a_val = aadc_matrix.coo()
        # Criar mapeamento (u,v) -> valor para busca rápida no batch (memória costeável, mas simples)
        mapping = {(int(r), int(c)): float(v) for r, c, v in zip(a_row.tolist(), a_col.tolist(), a_val.tolist())}
        for u, v in zip(row.tolist(), col.tolist()):
            out.append(mapping.get((int(u), int(v)), 0.0))
        return torch.tensor(out, dtype=torch.float)
    except Exception:
        # Fallback robusto: tentativa de indexação direta (algumas versões oferecem __getitem__)
        out = []
        for u, v in zip(row.tolist(), col.tolist()):
            try:
                val = aadc_matrix[int(u), int(v)]
                out.append(float(val.item()) if hasattr(val, 'item') else float(val))
            except Exception:
                out.append(0.0)
        return torch.tensor(out, dtype=torch.float)