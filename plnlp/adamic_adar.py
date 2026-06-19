def precompute_aadc_matrix(adj_t):
    """
    Pré-calcula a Matriz Esparsa de Adamic-Adar
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
    try:
        a_row, a_col, a_val = aadc_matrix.coo()
        aadc_matrix._aadc_mapping = {(int(r), int(c)): float(v) for r, c, v in zip(a_row.tolist(), a_col.tolist(), a_val.tolist())}
    except Exception:
        aadc_matrix._aadc_mapping = None
    return aadc_matrix

def get_batch_aadc(aadc_matrix, edge_index):
    """ Extrai rapidamente os valores da matriz esparsa para o lote atual (Verdadeiros ou Falsos) """
    row, col = edge_index[0], edge_index[1]
    # No PyTorch Geometric/Sparse, extrair valores discretos pode ser feito mapeando:
    # Se aadc_matrix for muito densa, essa operação requer cuidado, 
    # mas para o collab, é bem esparso.
    
    # Usa um lookup em memória se já tivermos o dicionário precalculado.
    mapping = getattr(aadc_matrix, '_aadc_mapping', None)
    if mapping is not None:
        return torch.tensor([mapping.get((int(u), int(v)), 0.0) for u, v in zip(row.tolist(), col.tolist())], dtype=torch.float)

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