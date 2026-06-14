# -*- coding: utf-8 -*-
"""
Script para analisar e comparar logs de treinamento das variantes.

Lê os arquivos de log gerados por train_variants.py e extrai:
  - Métricas finais (Hits@20, Hits@50, Hits@100, MRR)
  - Tempo de treinamento
  - Melhores checkpoints
  - Comparação entre variantes

Uso:
  python analyze_logs.py [--variant_dir checkpoints]
"""

import os
import re
from pathlib import Path
from collections import defaultdict

def find_all_logs(variant_dir='checkpoints'):
    """Procura por todos os logs em diretórios de variantes."""
    logs_by_variant = defaultdict(list)
    
    variant_path = Path(variant_dir)
    if not variant_path.exists():
        print(f"Diretório {variant_dir} não encontrado!")
        return logs_by_variant
    
    for variant_folder in sorted(variant_path.iterdir()):
        if variant_folder.is_dir():
            log_files = sorted(variant_folder.glob('log_*.txt'), reverse=True)
            if log_files:
                logs_by_variant[variant_folder.name] = log_files
    
    return logs_by_variant

def parse_log_file(log_path):
    """Extrai informações de um arquivo de log."""
    info = {
        'file': str(log_path),
        'hits20': [],
        'hits50': [],
        'hits100': [],
        'mrr': [],
        'valid_results': [],
        'test_results': [],
    }
    
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Extrai argumentos
        if '--spatial_mode' in content:
            spatial = re.search(r"--spatial_mode\s+(\w+)", content)
            temporal = re.search(r"--use_temporal\s+(\w+)", content)
            heuristic = re.search(r"--use_heuristic\s+(\w+)", content)
            
            info['config'] = {
                'spatial_mode': spatial.group(1) if spatial else '?',
                'use_temporal': temporal.group(1) if temporal else '?',
                'use_heuristic': heuristic.group(1) if heuristic else '?',
            }
        
        # Extrai métricas (padrão: "Valid: 45.23%, Test: 42.15%")
        valid_pattern = r'Valid:\s+([\d.]+)%'
        test_pattern = r'Test:\s+([\d.]+)%'
        
        for match in re.finditer(valid_pattern, content):
            info['valid_results'].append(float(match.group(1)))
        
        for match in re.finditer(test_pattern, content):
            info['test_results'].append(float(match.group(1)))
        
        # Métricas finais (padrão: "Hits@50" seguido de números)
        if 'Hits@20' in content:
            for match in re.finditer(r'Hits@20.*?(\d+\.\d+)', content):
                info['hits20'].append(float(match.group(1)))
        
        if 'Hits@50' in content:
            for match in re.finditer(r'Hits@50.*?(\d+\.\d+)', content):
                info['hits50'].append(float(match.group(1)))
        
        if 'Hits@100' in content:
            for match in re.finditer(r'Hits@100.*?(\d+\.\d+)', content):
                info['hits100'].append(float(match.group(1)))
        
        if 'MRR' in content:
            for match in re.finditer(r'MRR.*?(\d+\.\d+)', content):
                info['mrr'].append(float(match.group(1)))
    
    except Exception as e:
        info['error'] = str(e)
    
    return info

def compute_stats(values):
    """Calcula estatísticas para uma lista de valores."""
    if not values:
        return {}
    
    values = sorted(values)
    return {
        'min': min(values),
        'max': max(values),
        'mean': sum(values) / len(values),
        'median': values[len(values)//2],
        'count': len(values),
    }

def print_detailed_analysis(logs_by_variant):
    """Imprime análise detalhada por variante."""
    print("\n" + "="*100)
    print("ANÁLISE DETALHADA POR VARIANTE")
    print("="*100 + "\n")
    
    results_summary = {}
    
    for variant_name in sorted(logs_by_variant.keys()):
        log_files = logs_by_variant[variant_name]
        print(f"\n📁 {variant_name}")
        print(f"   {len(log_files)} arquivo(s) de log encontrado(s)")
        
        all_parsed = []
        for log_path in log_files:
            parsed = parse_log_file(log_path)
            all_parsed.append(parsed)
            
            print(f"\n   📄 {log_path.name}")
            
            if 'config' in parsed:
                cfg = parsed['config']
                print(f"      Config: spatial={cfg['spatial_mode']}, temporal={cfg['use_temporal']}, heuristic={cfg['use_heuristic']}")
            
            if 'error' in parsed:
                print(f"      ❌ Erro ao ler: {parsed['error']}")
            else:
                # Métricas encontradas
                if parsed['valid_results']:
                    v_stats = compute_stats(parsed['valid_results'])
                    print(f"      Valid: mean={v_stats.get('mean', 0):.2f}% (min={v_stats.get('min', 0):.2f}%, max={v_stats.get('max', 0):.2f}%)")
                
                if parsed['test_results']:
                    t_stats = compute_stats(parsed['test_results'])
                    print(f"      Test:  mean={t_stats.get('mean', 0):.2f}% (min={t_stats.get('min', 0):.2f}%, max={t_stats.get('max', 0):.2f}%)")
                
                # Hits
                for metric_name, values in [('Hits@20', parsed['hits20']), 
                                             ('Hits@50', parsed['hits50']), 
                                             ('Hits@100', parsed['hits100'])]:
                    if values:
                        stats = compute_stats(values)
                        print(f"      {metric_name}: {stats.get('mean', 0):.4f} ± {(stats.get('max', 0) - stats.get('min', 0))/2:.4f}")
                
                # MRR
                if parsed['mrr']:
                    stats = compute_stats(parsed['mrr'])
                    print(f"      MRR:   {stats.get('mean', 0):.4f} ± {(stats.get('max', 0) - stats.get('min', 0))/2:.4f}")
        
        # Resumo da variante
        if all_parsed:
            results_summary[variant_name] = all_parsed[0]

    return results_summary

def print_comparison_table(results_summary):
    """Imprime tabela comparativa das variantes."""
    print("\n" + "="*100)
    print("TABELA COMPARATIVA")
    print("="*100 + "\n")
    
    print(f"{'Variante':<25} | {'Spatial':<8} | {'Temporal':<8} | {'Heuristic':<10} | {'Test %':<8} | {'Hits@50':<8}")
    print("-" * 100)
    
    for variant_name in sorted(results_summary.keys()):
        info = results_summary[variant_name]
        
        cfg_str = ""
        if 'config' in info:
            cfg = info['config']
            spatial = "✓" if cfg['spatial_mode'] == 'sign' else " "
            temporal = "✓" if cfg['use_temporal'].lower() == 'true' else " "
            heuristic = "✓" if cfg['use_heuristic'].lower() == 'true' else " "
        else:
            spatial = temporal = heuristic = "?"
        
        test_val = compute_stats(info['test_results']).get('mean', 0) if info['test_results'] else 0
        hits50_val = compute_stats(info['hits50']).get('mean', 0) if info['hits50'] else 0
        
        print(f"{variant_name:<25} | {spatial:^8} | {temporal:^8} | {heuristic:^10} | {test_val:>7.2f}% | {hits50_val:>7.4f}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant_dir', type=str, default='checkpoints')
    args = parser.parse_args()
    
    print(f"Procurando logs em: {args.variant_dir}")
    logs_by_variant = find_all_logs(args.variant_dir)
    
    if not logs_by_variant:
        print(f"❌ Nenhum log encontrado em {args.variant_dir}")
        return
    
    print(f"✓ Encontrados {sum(len(v) for v in logs_by_variant.values())} arquivo(s) de log em {len(logs_by_variant)} variante(s)\n")
    
    results_summary = print_detailed_analysis(logs_by_variant)
    
    if results_summary:
        print_comparison_table(results_summary)
    
    print("\n" + "="*100)
    print("✓ Análise concluída!")
    print("="*100 + "\n")

if __name__ == '__main__':
    main()
