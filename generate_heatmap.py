"""
热力图生成脚本 - 生成 HPWL 和运行时间热力图

用法:
    python generate_heatmap.py --base-path ./data/hp --sim-start 10 --sim-end 100 --sim-steps 10 --batch-start 10 --batch-end 100 --batch-steps 10 --seed 78 --output ./heatmaps
"""

import argparse
import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json


def run_mcts(base_path: str, sims: int, batch: int, seed: int) -> tuple:
    """
    运行 MCTS 并返回 HPWL 和运行时间
    
    Returns:
        (hpwl, runtime) 或 (None, None) 如果失败
    """
    cmd = [
        "python", "main.py",
        "--base-path", base_path,
        "--sims", str(sims),
        "--batch", str(batch),
        "--seed", str(seed),
        "--no-tree",
        "--no-viz"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5分钟超时
        )
        output = result.stdout + result.stderr
        
        # 解析 HPWL (Best terminal reward) - 支持负数
        hpwl_match = re.search(r'Best terminal reward:\s*(-?[\d.]+)', output)
        # 解析运行时间
        time_match = re.search(r'MCTS运行时间:\s*([\d.]+)', output)
        
        # 检查是否找到了有效的终端状态
        if 'No terminal states found' in output:
            print(f"  [Warning] No terminal states found for sims={sims}, batch={batch}")
            return None, None
        
        if hpwl_match and time_match:
            hpwl = float(hpwl_match.group(1))
            runtime = float(time_match.group(1))
            return hpwl, runtime
        else:
            print(f"  [Warning] Could not parse output for sims={sims}, batch={batch}")
            print(f"  Output: {output[:500]}")
            return None, None
            
    except subprocess.TimeoutExpired:
        print(f"  [Timeout] sims={sims}, batch={batch}")
        return None, None
    except Exception as e:
        print(f"  [Error] sims={sims}, batch={batch}: {e}")
        return None, None


def generate_heatmaps(
    base_path: str,
    sim_range: list,
    batch_range: list,
    seed: int,
    output_dir: str,
    benchmark_name: str
):
    """生成 HPWL 和时间热力图"""
    
    n_sims = len(sim_range)
    n_batches = len(batch_range)
    
    # 初始化数据矩阵
    hpwl_matrix = np.full((n_batches, n_sims), np.nan)
    time_matrix = np.full((n_batches, n_sims), np.nan)
    
    total_runs = n_sims * n_batches
    current_run = 0
    
    print(f"\n{'='*60}")
    print(f"Generating heatmaps for {benchmark_name}")
    print(f"Simulations: {sim_range}")
    print(f"Batch sizes: {batch_range}")
    print(f"Total runs: {total_runs}")
    print(f"{'='*60}\n")
    
    # 运行所有组合
    for i, batch in enumerate(batch_range):
        for j, sims in enumerate(sim_range):
            current_run += 1
            print(f"[{current_run}/{total_runs}] Running sims={sims}, batch={batch}...", end=" ")
            
            hpwl, runtime = run_mcts(base_path, sims, batch, seed)
            
            if hpwl is not None:
                hpwl_matrix[i, j] = hpwl
                time_matrix[i, j] = runtime
                print(f"HPWL={hpwl:.2f}, Time={runtime:.2f}s")
            else:
                print("Failed")
    
    # 保存原始数据
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 将 numpy 类型转换为 Python 原生类型
    data = {
        "benchmark": benchmark_name,
        "sim_range": [int(x) for x in sim_range],
        "batch_range": [int(x) for x in batch_range],
        "seed": int(seed),
        "hpwl_matrix": [[float(x) if not np.isnan(x) else None for x in row] for row in hpwl_matrix],
        "time_matrix": [[float(x) if not np.isnan(x) else None for x in row] for row in time_matrix]
    }
    
    with open(output_path / f"{benchmark_name}_heatmap_data.json", "w") as f:
        json.dump(data, f, indent=2)
    
    # 生成 HPWL 热力图
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        hpwl_matrix,
        xticklabels=sim_range,
        yticklabels=batch_range,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        cbar_kws={'label': 'HPWL'}
    )
    plt.xlabel("Number of Simulations", fontsize=12)
    plt.ylabel("Batch Size", fontsize=12)
    plt.title(f"HPWL Heatmap - {benchmark_name} Benchmark", fontsize=14)
    plt.tight_layout()
    
    hpwl_path = output_path / f"fig_4_2_{benchmark_name}_hpwl_heatmap.png"
    plt.savefig(hpwl_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nHPWL heatmap saved to: {hpwl_path}")
    
    # 生成时间热力图
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        time_matrix,
        xticklabels=sim_range,
        yticklabels=batch_range,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        cbar_kws={'label': 'Runtime (seconds)'}
    )
    plt.xlabel("Number of Simulations", fontsize=12)
    plt.ylabel("Batch Size", fontsize=12)
    plt.title(f"Runtime Heatmap - {benchmark_name} Benchmark", fontsize=14)
    plt.tight_layout()
    
    time_path = output_path / f"fig_4_3_{benchmark_name}_time_heatmap.png"
    plt.savefig(time_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Time heatmap saved to: {time_path}")
    
    return hpwl_matrix, time_matrix


def main():
    parser = argparse.ArgumentParser(description='生成 MCTS 布局的 HPWL 和时间热力图')
    
    parser.add_argument('--base-path', type=str, default='./data/hp',
                        help='基准测试数据路径')
    parser.add_argument('--sim-start', type=int, default=200,
                        help='模拟次数起始值 (建议 >= 200)')
    parser.add_argument('--sim-end', type=int, default=1000,
                        help='模拟次数结束值')
    parser.add_argument('--sim-steps', type=int, default=5,
                        help='模拟次数的步数')
    parser.add_argument('--batch-start', type=int, default=20,
                        help='批处理大小起始值')
    parser.add_argument('--batch-end', type=int, default=100,
                        help='批处理大小结束值')
    parser.add_argument('--batch-steps', type=int, default=5,
                        help='批处理大小的步数')
    parser.add_argument('--seed', type=int, default=78,
                        help='随机种子')
    parser.add_argument('--output', type=str, default='./heatmaps',
                        help='输出目录')
    
    args = parser.parse_args()
    
    # 生成范围
    sim_range = list(np.linspace(args.sim_start, args.sim_end, args.sim_steps, dtype=int))
    batch_range = list(np.linspace(args.batch_start, args.batch_end, args.batch_steps, dtype=int))
    
    # 获取基准测试名称
    benchmark_name = Path(args.base_path).name
    
    # 生成热力图
    generate_heatmaps(
        base_path=args.base_path,
        sim_range=sim_range,
        batch_range=batch_range,
        seed=args.seed,
        output_dir=args.output,
        benchmark_name=benchmark_name
    )
    
    print(f"\n{'='*60}")
    print("Heatmap generation complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

