import os
import time
import subprocess
import sys

# Importações para a Heurística Solo
import torch
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from torch_sparse import SparseTensor
from plnlp.utils import precompute_aadc_matrix

# ==========================================
# 1. PARÂMETROS FIXOS DAS REDES NEURAIS (A Base Igualitária)
# ==========================================
FIXED_PARAMS = (
    "--data_name ogbl-collab "
    "--predictor MLP "
    "--gnn_num_layers 2 "
    "--mlp_num_layers 2 "
    "--dropout 0.4 "           # Ajustado conforme sua indicação (0.4)
    "--lr 0.0015 "
    "--epochs 500 "            # 500 épocas cravadas para todos
    "--neg_sampler adversarial "
    "--loss_func WeightedHingeAUC "
    "--use_lr_decay True "
    "--train_on_subgraph False" # Treinando no grafo inteiro sempre
)

# ==========================================
# 2. DEFINIÇÃO DA ABLAÇÃO DAS REDES NEURAIS
# ==========================================
# Aqui aplicamos a sua lógica condicional para Random Walk, Valedges e Spatial Mode
MODELS = {
    # Modelo 1: A referência pura. Usa 'sign' e permite os hacks estruturais originais.
    "1_Baseline_Puro": "--use_heuristic False --use_temporal False --spatial_mode sign --random_walk_augment True --use_valedges_as_input True",
    
    # Modelo 2: Troca o espacial para 'inception' e liga a Heurística. Mantém os hacks já que não há temporalidade.
    "2_Inception_Heuristica": "--use_heuristic True --use_temporal False --spatial_mode inception --random_walk_augment True --use_valedges_as_input True",
    
    # Modelo 3: Liga o Temporal e Inception. DESLIGA OBRIGATORIAMENTE os hacks para não quebrar o Bochner.
    "3_Inception_Temporal": "--use_heuristic False --use_temporal True --spatial_mode inception --random_walk_augment False --use_valedges_as_input False",
    
    # Modelo 4: O Seu Estado da Arte. Tudo ligado, mas os hacks desligados para proteger o temporal.
    "4_Modelo_Completo": "--use_heuristic True --use_temporal True --spatial_mode inception --random_walk_augment False --use_valedges_as_input False"
}

# ==========================================
# 3. FUNÇÃO: EXECUTAR REDES NEURAIS
# ==========================================
def run_neural_networks(base_log_dir):
    print(f"\n[{time.strftime('%H:%M:%S')}] --- INICIANDO BATERIA DE REDES NEURAIS ---")
    
    for model_name, specific_flags in MODELS.items():
        print(f"\n[{time.strftime('%H:%M:%S')}] Iniciando treinamento: {model_name}...")
        checkpoint_dir = f"checkpoints/{model_name}"
        log_file_path = os.path.join(base_log_dir, f"log_{model_name}.txt")
        
        full_command = f"python3 plnlp_sign.py {FIXED_PARAMS} {specific_flags} --save_dir {checkpoint_dir} > {log_file_path} 2>&1"
        
        try:
            subprocess.run(full_command, shell=True, check=True)
            print(f"[{time.strftime('%H:%M:%S')}] Sucesso! {model_name} concluído. Log em: {log_file_path}")
        except subprocess.CalledProcessError:
            print(f"[{time.strftime('%H:%M:%S')}] ERRO no modelo {model_name}. Verifique o log. Pulando para o próximo...")

# ==========================================
# 4. FUNÇÃO: EXECUTAR HEURÍSTICA PURA (Sem Treino)
# ==========================================
def run_heuristic_solo(base_log_dir):
    print(f"\n[{time.strftime('%H:%M:%S')}] --- INICIANDO AVALIAÇÃO DA HEURÍSTICA SOLO ---")
    log_file_path = os.path.join(base_log_dir, "log_5_Heuristica_Pura_Solo.txt")
    
    original_stdout = sys.stdout
    with open(log_file_path, 'w') as f:
        sys.stdout = f
        
        print("=== [EXPERIMENTO] ABLAÇÃO: HEURÍSTICA AA-DC SOLO ===")
        print("Carregando o dataset ogbl-collab...")
        dataset = PygLinkPropPredDataset(name='ogbl-collab', root='dataset')
        data = dataset[0]
        split_edge = dataset.get_edge_split()

        YEAR_THRESHOLD = 2010
        print(f"Filtrando grafo de treino para conexões a partir do ano >= {YEAR_THRESHOLD}...")
        
        train_edges = split_edge['train']['edge']
        train_years = split_edge['train']['year']
        mask = train_years >= YEAR_THRESHOLD
        filtered_edges = train_edges[mask]
        num_nodes = data.num_nodes

        print("Construindo a Matriz de Adjacência Esparsa...")
        weights = torch.ones(filtered_edges.size(0), dtype=torch.float32)
        edge_index = filtered_edges.t()
        
        adj_t = SparseTensor(row=edge_index[0], col=edge_index[1], value=weights, sparse_sizes=(num_nodes, num_nodes))
        adj_t = adj_t.to_symmetric()

        print(">>> Computando a Matriz Global de AA-DC Solo (Aguarde)...")
        aadc_matrix = precompute_aadc_matrix(adj_t)
        print("Matriz calculada com sucesso!")

        def get_scores_batched(edges, sparse_mat, batch_size=4000):
            scores = []
            for i in range(0, edges.size(0), batch_size):
                batch_edges = edges[i:i+batch_size]
                rows = batch_edges[:, 0]
                cols = batch_edges[:, 1]
                sub_mat = sparse_mat.index_select(0, rows)
                dense_block = sub_mat.to_dense()
                batch_scores = dense_block[torch.arange(batch_edges.size(0)), cols]
                scores.append(batch_scores)
            return torch.cat(scores, dim=0)

        evaluator = Evaluator(name='ogbl-collab')

        print("\nExtraindo predições para o conjunto de VALIDAÇÃO...")
        pos_valid_pred = get_scores_batched(split_edge['valid']['edge'], aadc_matrix)
        neg_valid_pred = get_scores_batched(split_edge['valid']['edge_neg'], aadc_matrix)

        print("Extraindo predições para o conjunto de TESTE...")
        pos_test_pred = get_scores_batched(split_edge['test']['edge'], aadc_matrix)
        neg_test_pred = get_scores_batched(split_edge['test']['edge_neg'], aadc_matrix)

        print("\n" + "="*50)
        print("         RESULTADOS DA HEURÍSTICA AA-DC PURA")
        print("="*50)
        
        for K in [20, 50, 100]:
            evaluator.K = K
            valid_results = evaluator.eval({'y_pred_pos': pos_valid_pred, 'y_pred_neg': neg_valid_pred})
            test_results = evaluator.eval({'y_pred_pos': pos_test_pred, 'y_pred_neg': neg_test_pred})
            print(f"Hits@{K} -> Validação: {valid_results[f'hits@{K}']*100:.2f}% | Teste: {test_results[f'hits@{K}']*100:.2f}%")
        print("="*50)

    sys.stdout = original_stdout
    print(f"[{time.strftime('%H:%M:%S')}] Sucesso! Avaliação da Heurística concluída. Log em: {log_file_path}")

# ==========================================
# 5. MOTOR PRINCIPAL
# ==========================================
def main():
    base_log_dir = "resultados_ablacao"
    if not os.path.exists(base_log_dir):
        os.makedirs(base_log_dir)

    print("="*60)
    print(" INICIANDO PIPELINE MESTRE DE ABLAÇÃO - OGBL-COLLAB ")
    print("="*60)
    
    run_neural_networks(base_log_dir)
    run_heuristic_solo(base_log_dir)
    
    print("\n" + "="*60)
    print(f"[{time.strftime('%H:%M:%S')}] PIPELINE TOTALMENTE FINALIZADO!")
    print("Todos os resultados encontram-se na pasta: /resultados_ablacao/")
    print("="*60)

if __name__ == "__main__":
    main()