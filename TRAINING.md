# 📊 Treinamento e Análise de Variantes PLNLP

Este diretório contém scripts para treinar e comparar as 8 variantes de ablação do modelo PLNLP.

## 📋 As 8 Variantes

| # | Nome | Descrição | Flags |
|---|------|-----------|-------|
| 1 | **BASE** | Baseline sem ablações | — |
| 2 | **SIGN** | Propagação estática (SIGN) | `--spatial_mode sign` |
| 3 | **TEMPORAL** | Codificação temporal de Bochner | `--use_temporal true` |
| 4 | **HEURISTIC** | Matriz de similaridade AA-DC | `--use_heuristic true` |
| 5 | **SIGN+TEMPORAL** | Combinação espacial + temporal | `--spatial_mode sign --use_temporal true` |
| 6 | **SIGN+HEURISTIC** | Combinação espacial + heurística | `--spatial_mode sign --use_heuristic true` |
| 7 | **TEMPORAL+HEURISTIC** | Combinação temporal + heurística | `--use_temporal true --use_heuristic true` |
| 8 | **ALL** | Todas as três ablações | `--spatial_mode sign --use_temporal true --use_heuristic true` |

## 🚀 Início Rápido

### 1. Treinar Todas as Variantes (Modo Rápido para Teste)

```bash
python train_variants.py --quick --data_name ogbl-ddi
```

Isso executa:
- **Épocas**: 50 (em vez de 500)
- **Rodadas**: 2 (em vez de 10)
- **Avaliação**: a cada 10 épocas (em vez de 5)

⏱️ **Tempo estimado**: ~30 min (com GPU NVIDIA)

### 2. Treinar com Configuração Completa

```bash
python train_variants.py --data_name ogbl-ddi --epochs 500 --runs 10
```

⏱️ **Tempo estimado**: 4-8 horas (com GPU NVIDIA)

### 3. Analisar Logs Gerados

Depois que o treinamento terminar (ou enquanto está rodando):

```bash
python analyze_logs.py
```

Isso gera:
- Estatísticas por variante (min, max, média)
- Tabela comparativa final
- Extração automática de métricas (Hits@20, Hits@50, Hits@100, MRR)

## 📂 Estrutura de Arquivos

Após executar `train_variants.py`, a estrutura fica assim:

```
Graph_ogb/
├── train_variants.py          # Script principal de treinamento
├── analyze_logs.py            # Script de análise de logs
├── training_summary.txt       # Resumo final (gerado)
│
├── checkpoints/               # Diretório de checkpoints
│   ├── 1_BASE/
│   │   ├── log_ogbl-ddi_*.txt     # Logs de treinamento
│   │   ├── best_model.pt          # Melhor modelo
│   │   └── last_model.pt          # Último modelo
│   ├── 2_SIGN/
│   ├── 3_TEMPORAL/
│   ├── 4_HEURISTIC/
│   ├── 5_SIGN_TEMPORAL/
│   ├── 6_SIGN_HEURISTIC/
│   ├── 7_TEMPORAL_HEURISTIC/
│   └── 8_ALL/
```

## 🔧 Opções Avançadas

### Treinar Uma Variante Específica

```bash
# Apenas a variante SIGN
python plnlp_sign.py --data_name ogbl-ddi --spatial_mode sign --save_dir checkpoints/2_SIGN

# Apenas com temporal + heurística
python plnlp_sign.py --data_name ogbl-ddi --use_temporal true --use_heuristic true --save_dir checkpoints/7_TEMPORAL_HEURISTIC
```

### Usar Dataset Diferente

```bash
# Treinar em ogbl-citation2 (requer mais memória)
python train_variants.py --quick --data_name ogbl-citation2

# Treinar em ogbl-collab
python train_variants.py --quick --data_name ogbl-collab
```

### Customizar Recursos Computacionais

```bash
# Usar GPU 1 em vez de GPU 0
python train_variants.py --device 1 --quick

# Reduzir batch size (para GPUs com pouca VRAM)
python plnlp_sign.py --batch_size 32768 --save_dir checkpoints/custom
```

## 📊 Entendendo os Logs

Cada arquivo de log (`log_ogbl-ddi_*.txt`) contém:

1. **Argumentos do treinamento** (flags usadas)
2. **Número total de parâmetros** do modelo
3. **Progresso por época**: Loss, Learning Rate, Valid%, Test%
4. **Métricas finais**: Hits@20, Hits@50, Hits@100 (para `eval_metric='hits'`)
5. **Melhor checkpoint**: arquivo e métrica

### Exemplo de Métrica em Log

```
Run: 01, Epoch: 10, Loss: 0.3245, Learning Rate: 0.0010, Valid: 45.23%, Test: 42.15%
```

## 🎯 Interpretando Resultados

### Qual Variante é a Melhor?

Depois de rodar `analyze_logs.py`, procure:

1. **Test% mais alto** → melhor desempenho em dados nunca vistos
2. **Hits@50** (para datasets de hits) → métrica principal
3. **MRR** (para datasets de ranking) → métrica de ranking

### Exemplo de Saída da Análise

```
Variante                  | Spatial  | Temporal | Heuristic | Test %  | Hits@50
---------                 | -------  | -------- | --------- | -------  | -------
1_BASE                    |          |          |           | 41.23%  | 0.4123
2_SIGN                    | ✓        |          |           | 43.45%  | 0.4345  ← Melhorou!
3_TEMPORAL                |          | ✓        |           | 42.10%  | 0.4210
4_HEURISTIC               |          |          | ✓         | 41.80%  | 0.4180
5_SIGN_TEMPORAL           | ✓        | ✓        |           | 44.20%  | 0.4420  ← Melhor ainda!
6_SIGN_HEURISTIC          | ✓        |          | ✓         | 42.50%  | 0.4250
7_TEMPORAL_HEURISTIC      |          | ✓        | ✓         | 43.00%  | 0.4300
8_ALL                     | ✓        | ✓        | ✓         | 44.80%  | 0.4480  ← Melhor de todas!
```

## 🐛 Troubleshooting

### Erro: `torch_sparse not found`

Você precisa instalar o PyG com wheels compatíveis com seu CUDA. Veja [INSTALL.md](../INSTALL.md).

### Erro: `CUDA out of memory`

Reduza o batch size:

```bash
python train_variants.py --quick --data_name ogbl-ddi --batch_size 32768
```

Ou use CPU (mais lento):

```bash
python plnlp_sign.py --device cpu --epochs 50 --save_dir checkpoints/custom
```

### Logs não estão sendo salvos

Verifique:

1. Diretório `checkpoints/` existe e tem permissão de escrita
2. Use `--save_dir checkpoints/variante_name` explicitamente
3. Verifique output na tela para mensagens de erro

## 📝 Próximos Passos

Depois de treinar e analisar:

1. **Identifique a variante mais rápida vs. mais precisa**
2. **Execute um treinamento final completo** da melhor variante (epochs=500, runs=10)
3. **Salve os checkpoints** (`best_model.pt`) para uso em produção
4. **Documente os resultados** em um arquivo de relatório

## 📚 Referências

- [PLNLP Model Architecture](../plnlp/model.py)
- [Training Script](../plnlp_sign.py)
- [Original Project](https://github.com/your-repo)

---

**Dúvidas?** Verifique os arquivos de log em `checkpoints/*/log_*.txt`
