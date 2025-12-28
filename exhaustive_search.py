"""
穷举搜索程序 - 找到小规模布局问题的最优解

对于小规模问题（如 4 个模块），穷举所有可能的序列对和方向组合，
找到全局最优的 HPWL 解。

用法:
    python exhaustive_search.py --base-path ./data/testcase_4die_small_a1
"""

import argparse
import itertools
import time
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from data_loader import BookshelfLoader
from placement_solver import PlacementSolver


def exhaustive_search(base_path: str, output_dir: str = "./exhaustive_results"):
    """
    穷举搜索所有可能的布局组合
    
    Args:
        base_path: 基准测试数据路径
        output_dir: 输出目录
    """
    # 加载数据
    print(f"Loading benchmark from: {base_path}")
    bench = BookshelfLoader.load_bookshelf_from_base_path(base_path)
    
    # 识别可移动模块
    movable_mask = bench.is_terminal == 0
    movable_indices = jnp.where(movable_mask)[0]
    num_movable = len(movable_indices)
    
    print(f"\n{'='*60}")
    print(f"Exhaustive Search for Optimal Placement")
    print(f"{'='*60}")
    print(f"Benchmark: {Path(base_path).name}")
    print(f"Movable modules: {num_movable}")
    print(f"Terminals: {jnp.sum(bench.is_terminal)}")
    print(f"Nets: {len(bench.nets_ptr) - 1}")
    
    # 计算搜索空间大小
    import math
    num_permutations = math.factorial(num_movable) ** 2  # 序列对排列
    num_orientations = 4 ** num_movable  # 方向组合
    total_combinations = num_permutations * num_orientations
    
    print(f"\nSearch space:")
    print(f"  Sequence pair permutations: {num_permutations:,}")
    print(f"  Orientation combinations: {num_orientations:,}")
    print(f"  Total combinations: {total_combinations:,}")
    print(f"{'='*60}\n")
    
    if total_combinations > 10_000_000:
        print(f"WARNING: Search space too large ({total_combinations:,})! Consider using MCTS instead.")
        return None
    
    # 创建布局求解器
    solver = PlacementSolver(bench, movable_indices)
    
    # 生成模块索引（按面积排序）
    areas = bench.widths[movable_indices] * bench.heights[movable_indices]
    sorted_order = jnp.argsort(-areas)
    module_order = movable_indices[sorted_order]
    module_list = list(range(num_movable))
    
    # 初始化最优解
    best_hpwl = float('inf')
    best_s1 = None
    best_s2 = None
    best_orientations = None
    best_x = None
    best_y = None
    
    # 统计
    valid_count = 0
    invalid_count = 0
    
    # JIT 编译 HPWL 计算函数
    @jax.jit
    def compute_hpwl_jit(s1, s2, orientations):
        return solver.compute_hpwl(s1, s2, orientations)
    
    # 预热 JIT
    dummy_s1 = jnp.arange(num_movable, dtype=jnp.int32)
    dummy_s2 = jnp.arange(num_movable, dtype=jnp.int32)
    dummy_orient = jnp.zeros(num_movable, dtype=jnp.int32)
    _ = compute_hpwl_jit(dummy_s1, dummy_s2, dummy_orient)
    
    print("Starting exhaustive search...")
    start_time = time.time()
    
    # 生成所有序列对排列
    all_permutations = list(itertools.permutations(module_list))
    
    # 生成所有方向组合
    all_orientations = list(itertools.product([0, 1, 2, 3], repeat=num_movable))
    
    # 使用 tqdm 显示进度
    total_iter = len(all_permutations) ** 2 * len(all_orientations)
    
    with tqdm(total=total_iter, desc="Searching", unit="combo") as pbar:
        for s1_perm in all_permutations:
            for s2_perm in all_permutations:
                s1 = jnp.array(s1_perm, dtype=jnp.int32)
                s2 = jnp.array(s2_perm, dtype=jnp.int32)
                
                for orient in all_orientations:
                    orientations = jnp.array(orient, dtype=jnp.int32)
                    
                    try:
                        hpwl = float(compute_hpwl_jit(s1, s2, orientations))
                        
                        if hpwl > 0 and not np.isnan(hpwl) and not np.isinf(hpwl):
                            valid_count += 1
                            
                            if hpwl < best_hpwl:
                                best_hpwl = hpwl
                                best_s1 = s1_perm
                                best_s2 = s2_perm
                                best_orientations = orient
                                
                                # 计算最佳布局的坐标
                                x, y, w, h, _, _ = solver.compute_final_positions(
                                    jnp.array(best_s1, dtype=jnp.int32),
                                    jnp.array(best_s2, dtype=jnp.int32),
                                    jnp.array(best_orientations, dtype=jnp.int32)
                                )
                                best_x = x
                                best_y = y
                        else:
                            invalid_count += 1
                    except Exception as e:
                        invalid_count += 1
                    
                    pbar.update(1)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    # 输出结果
    print(f"\n{'='*60}")
    print(f"EXHAUSTIVE SEARCH COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {elapsed:.2f} seconds")
    print(f"Valid placements: {valid_count:,}")
    print(f"Invalid placements: {invalid_count:,}")
    print(f"\n{'='*60}")
    print(f"OPTIMAL SOLUTION FOUND")
    print(f"{'='*60}")
    print(f"Best HPWL: {best_hpwl:.4f}")
    print(f"Best S+: {list(best_s1)}")
    print(f"Best S-: {list(best_s2)}")
    print(f"Best Orientations: {list(best_orientations)}")
    orient_names = ['N', 'E', 'S', 'W']
    print(f"  (Orientations: {[orient_names[o] for o in best_orientations]})")
    print(f"{'='*60}")
    
    # 保存结果
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    benchmark_name = Path(base_path).name
    
    result = {
        "benchmark": benchmark_name,
        "num_movable": int(num_movable),
        "total_combinations": int(total_combinations),
        "valid_placements": int(valid_count),
        "invalid_placements": int(invalid_count),
        "elapsed_time_sec": float(elapsed),
        "optimal_hpwl": float(best_hpwl),
        "optimal_s1": list(best_s1),
        "optimal_s2": list(best_s2),
        "optimal_orientations": list(best_orientations),
        "optimal_orientations_names": [orient_names[o] for o in best_orientations]
    }
    
    result_file = output_path / f"{benchmark_name}_optimal.json"
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to: {result_file}")
    
    # 可视化最佳布局
    try:
        from visualizer import PlacementVisualizer
        
        x, y, w, h, pins_dx, pins_dy = solver.compute_final_positions(
            jnp.array(best_s1, dtype=jnp.int32),
            jnp.array(best_s2, dtype=jnp.int32),
            jnp.array(best_orientations, dtype=jnp.int32)
        )
        
        viz_file = output_path / f"{benchmark_name}_optimal_placement.png"
        PlacementVisualizer.plot_placement(
            bench, 
            jnp.array(x), 
            jnp.array(y),
            jnp.array(w),
            jnp.array(h),
            jnp.array(pins_dx),
            jnp.array(pins_dy),
            movable_indices,
            str(viz_file)
        )
        print(f"Visualization saved to: {viz_file}")
    except Exception as e:
        print(f"Could not generate visualization: {e}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description='穷举搜索最优布局')
    parser.add_argument('--base-path', type=str, default='./data/testcase_4die_small_a1',
                        help='基准测试数据路径')
    parser.add_argument('--output', type=str, default='./exhaustive_results',
                        help='输出目录')
    
    args = parser.parse_args()
    
    result = exhaustive_search(args.base_path, args.output)
    
    if result:
        print(f"\n✓ Optimal HPWL for {result['benchmark']}: {result['optimal_hpwl']:.4f}")


if __name__ == '__main__':
    main()

