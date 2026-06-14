# -*- coding: utf-8 -*-
"""
Script para treinar e comparar as variantes de ablação do PLNLP.

Treina 8 variantes:
  1. BASE: modelo baseline padrão
  2. SIGN: propagação estática (SIGN)
  3. TEMPORAL: codificação de tempo de Bochner
  4. HEURISTIC: matriz de similaridade AA-DC
  5. SIGN+TEMPORAL: combinação espacial + temporal
  6. SIGN+HEURISTIC: combinação espacial + heurística
  7. TEMPORAL+HEURISTIC: combinação temporal + heurística
  8. ALL: todas as três ablações ativadas

Uso:
  python train_variants.py [--data_name ogbl-ddi] [--epochs 50] [--runs 2] [--quick]

Flags:
  --data_name: nome do dataset (padrão: ogbl-ddi)
  --epochs: número de épocas por treinamento (padrão: 500)
  --runs: número de rodadas (padrão: 10)
  --quick: modo rápido (epochs=50, runs=2, eval_steps=10)
  --device: GPU (padrão: 0)

Saída:
  - logs/ → diretório com logs de cada variante
  - checkpoints/ → modelos salvos (best_model.pt, last_model.pt)
  - summary.txt → resumo comparativo final
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_name', type=str, default='ogbl-ddi')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--quick', action='store_true', help='Modo rápido: epochs=50, runs=2')
    return parser.parse_args()

def create_variant_config():
    """Define as 8 variantes de ablação."""
    variants = {
        '1_BASE': {
            'name': 'Baseline (sem ablações)',
            'spatial_mode': 'base',
            'use_temporal': False,
            'use_heuristic': False,
        },
        '2_SIGN': {
            'name': 'SIGN (Propagação Estática)',
            'spatial_mode': 'sign',
            'use_temporal': False,
            'use_heuristic': False,
        },
        '3_TEMPORAL': {
            'name': 'Temporal (Codificação de Bochner)',
            'spatial_mode': 'base',
            'use_temporal': True,
            'use_heuristic': False,
        },
        '4_HEURISTIC': {
            'name': 'Heurística (Matriz AA-DC)',
            'spatial_mode': 'base',
            'use_temporal': False,
            'use_heuristic': True,
        },
        '5_SIGN_TEMPORAL': {
            'name': 'SIGN + Temporal',
            'spatial_mode': 'sign',
            'use_temporal': True,
            'use_heuristic': False,
        },
        '6_SIGN_HEURISTIC': {
            'name': 'SIGN + Heurística',
            'spatial_mode': 'sign',
            'use_temporal': False,
            'use_heuristic': True,
        },
        '7_TEMPORAL_HEURISTIC': {
            'name': 'Temporal + Heurística',
            'spatial_mode': 'base',
            'use_temporal': True,
            'use_heuristic': True,
        },
        '8_ALL': {
            'name': 'SIGN + Temporal + Heurística (Completo)',
            'spatial_mode': 'sign',
            'use_temporal': True,
            'use_heuristic': True,
        },
    }
    return variants

def build_command(variant_key, variant_cfg, args):
    """Constrói o comando python para treinar uma variante."""
    save_dir = f'checkpoints/{variant_key}'
    os.makedirs(save_dir, exist_ok=True)
    
    cmd = [
        'python', 'plnlp_sign.py',
        '--data_name', args.data_name,
        '--epochs', str(args.epochs),
        '--runs', str(args.runs),
        '--eval_steps', str(args.eval_steps),
        '--log_steps', str(args.log_steps),
        '--batch_size', str(args.batch_size),
        '--device', str(args.device),
        '--save_dir', save_dir,
        '--spatial_mode', variant_cfg['spatial_mode'],
        '--use_temporal', str(variant_cfg['use_temporal']),
        '--use_heuristic', str(variant_cfg['use_heuristic']),
    ]
    return cmd, save_dir

def run_variant(variant_key, variant_cfg, args, variant_num, total_variants):
    """Executa o treinamento de uma variante e captura logs."""
    print(f"\n{'='*80}")
    print(f"[{variant_num}/{total_variants}] Treinando: {variant_cfg['name']}")
    print(f"{'='*80}")
    
    cmd, save_dir = build_command(variant_key, variant_cfg, args)
    
    print(f"Config:")
    print(f"  spatial_mode:  {variant_cfg['spatial_mode']}")
    print(f"  use_temporal:  {variant_cfg['use_temporal']}")
    print(f"  use_heuristic: {variant_cfg['use_heuristic']}")
    print(f"  save_dir:      {save_dir}")
    print(f"\nComando: {' '.join(cmd)}\n")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=False)
        elapsed = time.time() - start_time
        status = '✓ SUCESSO' if result.returncode == 0 else '✗ ERRO'
        return {
            'variant_key': variant_key,
            'variant_name': variant_cfg['name'],
            'status': status,
            'elapsed_sec': elapsed,
            'save_dir': save_dir,
            'returncode': result.returncode,
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            'variant_key': variant_key,
            'variant_name': variant_cfg['name'],
            'status': f'✗ EXCEÇÃO: {type(e).__name__}',
            'elapsed_sec': elapsed,
            'save_dir': save_dir,
            'returncode': -1,
        }

def find_logs(save_dir):
    """Procura por arquivos de log no diretório de salvamento."""
    log_files = list(Path(save_dir).glob('log_*.txt'))
    return sorted(log_files, reverse=True)  # Mais recentes primeiro

def extract_final_metrics(log_file):
    """Extrai as métricas finais de um arquivo de log."""
    metrics = {}
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # Procura pelas últimas linhas com "Hits@" ou "MRR"
        for line in reversed(lines):
            if 'Hits@' in line or 'MRR' in line:
                # Tenta extrair valor numérico
                try:
                    parts = line.strip().split()
                    for i, part in enumerate(parts):
                        if '%' in part:
                            metric_name = parts[i-1] if i > 0 else 'unknown'
                            metric_value = float(part.replace('%', ''))
                            metrics[metric_name] = metric_value
                            break
                except (ValueError, IndexError):
                    pass
        
        return metrics if metrics else {'status': 'logs incompletos'}
    except Exception as e:
        return {'error': str(e)}

def print_summary(results):
    """Imprime sumário comparativo das execuções."""
    print(f"\n{'='*80}")
    print("RESUMO FINAL DE TREINAMENTOS")
    print(f"{'='*80}\n")
    
    summary_lines = []
    
    for res in results:
        line = (
            f"{res['variant_key']:20s} | "
            f"{res['status']:15s} | "
            f"{res['elapsed_sec']:8.1f}s"
        )
        summary_lines.append(line)
        print(f"{res['variant_key']:20s} → {res['variant_name']}")
        print(f"  Status: {res['status']}")
        print(f"  Tempo: {res['elapsed_sec']:.1f}s ({res['elapsed_sec']/60:.1f}min)")
        
        # Tenta extrair métricas dos logs
        log_files = find_logs(res['save_dir'])
        if log_files:
            metrics = extract_final_metrics(log_files[0])
            print(f"  Metrics: {metrics}")
            print(f"  Log: {log_files[0].name}")
        else:
            print(f"  Nenhum log encontrado em {res['save_dir']}")
        print()
    
    return summary_lines

def main():
    args = parse_args()
    
    # Modo rápido
    if args.quick:
        args.epochs = 50
        args.runs = 2
        args.eval_steps = 10
        print("[MODO RÁPIDO] epochs=50, runs=2, eval_steps=10\n")
    
    # Cria diretórios
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    variants = create_variant_config()
    print(f"\nTreinando {len(variants)} variantes no dataset '{args.data_name}'...")
    print(f"Config: epochs={args.epochs}, runs={args.runs}, eval_steps={args.eval_steps}")
    
    results = []
    start_total = time.time()
    
    for i, (variant_key, variant_cfg) in enumerate(variants.items(), 1):
        result = run_variant(variant_key, variant_cfg, args, i, len(variants))
        results.append(result)
    
    total_elapsed = time.time() - start_total
    
    # Sumário
    summary_lines = print_summary(results)
    
    print(f"{'='*80}")
    print(f"Tempo Total: {total_elapsed:.1f}s ({total_elapsed/3600:.2f}h)")
    print(f"{'='*80}\n")
    
    # Salva sumário em arquivo
    summary_file = 'training_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("RESUMO DE TREINAMENTOS DE VARIANTES\n")
        f.write("="*80 + "\n\n")
        f.write(f"Dataset: {args.data_name}\n")
        f.write(f"Epochs: {args.epochs}, Runs: {args.runs}\n")
        f.write(f"Tempo Total: {total_elapsed:.1f}s ({total_elapsed/3600:.2f}h)\n\n")
        for line in summary_lines:
            f.write(line + "\n")
    
    print(f"Sumário salvo em: {summary_file}")

if __name__ == '__main__':
    main()
