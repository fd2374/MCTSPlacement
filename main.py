"""
主程序模块 - MCTS布局算法主程序
"""
from __future__ import annotations

import argparse
import time
import jax
import jax.numpy as jnp
import mctx
import functools

from data_loader import BookshelfLoader
from placement_state import StateManager, PlacementState
from mcts_placer import MCTSPlacer
from visualizer import PlacementVisualizer
from config import PlacementConfig
from post_optimizer import PostOptimizer


class PlacementRunner:
    """布局运行器
    
    负责协调整个MCTS布局算法的执行流程，包括数据加载、算法执行和结果输出。
    """
    
    def __init__(self, config: PlacementConfig):
        """初始化布局运行器
        
        Args:
            config: 布局算法配置
        """
        self.config = config
        self.bench = None
        self.movable_indices = None
        self.num_movable = 0
        self.placer = None  # MCTS布局器
        
    def load_benchmark(self) -> None:
        """加载基准测试数据并计算interposer边界"""
        self.bench = BookshelfLoader.load_bookshelf_from_base_path(self.config.base_path)
        
        # 识别可移动模块
        movable_mask = self.bench.is_terminal == 0
        self.movable_indices = jnp.where(movable_mask)[0]
        self.num_movable = len(self.movable_indices)
        
        # 计算interposer边界：优先使用config中手动指定的值，否则从terminal自动计算
        if self.config.boundary_width is not None and self.config.boundary_height is not None:
            self.boundary_width = self.config.boundary_width
            self.boundary_height = self.config.boundary_height
        else:
            self.boundary_width, self.boundary_height = self._calc_boundary_from_terminals()
        
        print(f"总节点数: {len(self.bench.names)}")
        print(f"可移动模块: {self.num_movable}")
        print(f"终端/固定节点: {jnp.sum(self.bench.is_terminal)}")
        print(f"网络数: {len(self.bench.nets_ptr) - 1}")
        print(f"Interposer边界: {self.boundary_width:.2f} x {self.boundary_height:.2f}")
    
    def run_mcts(self) -> tuple:
        """运行MCTS算法"""
        print(f"\n运行MCTS，{self.config.num_simulations}次模拟...")
        
        # 按面积排序模块（降序）
        ordered_modules = jax.random.permutation(jax.random.PRNGKey(self.config.seed), self.movable_indices)
        
        # 创建MCTS布局器
        self.placer = MCTSPlacer(self.bench, jnp.array(self.movable_indices), ordered_modules)
        
        # 运行MCTS
        rng_key = jax.random.PRNGKey(self.config.seed)
        rng_key, subkey = jax.random.split(rng_key)
        
        initial_state = StateManager.create_initial_state(self.num_movable)
        recurrent_fn = jax.vmap(self.placer.create_recurrent_fn(), (None, None, 0, 0))
        root = jax.vmap(self.placer.root_fn, (None, None, 0))(
            initial_state, self.placer.max_actions, jax.random.split(subkey, self.config.batch_size)
        )
        
        policy_output = mctx.gumbel_muzero_policy(
            params=None,
            rng_key=rng_key,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=self.config.num_simulations,
            max_depth=3 * self.num_movable,
            gumbel_scale=self.config.gumbel_scale,
            qtransform=functools.partial(mctx.qtransform_completed_by_mix_value)
        )
        
        # 提取最佳终端状态
        best_state, best_reward = self._extract_best_terminal_state(policy_output.search_tree)
        
        print(f"  最佳奖励: {float(best_reward):.2f}")
        print(f"  s1={best_state.s1}, s2={best_state.s2}")
        print(f"  orientations={best_state.orientations}")
        
        return policy_output, best_state
    
    def _extract_best_terminal_state(self, tree):
        """从搜索树中提取最佳终端状态"""
        target_step = 3 * self.num_movable
        terminal_mask = tree.embeddings.step == target_step
        masked_values = jnp.where(terminal_mask, tree.node_values, -jnp.inf)
        
        flat_values = masked_values.reshape(-1)
        best_idx = int(jnp.argmax(flat_values))
        best_value = float(flat_values[best_idx])
        
        if best_value == float('-inf'):
            print("警告: 未找到终端状态，建议增加 --sims")
            return StateManager.create_initial_state(self.num_movable), 0.0
        
        num_nodes = tree.node_values.shape[1]
        batch_idx, node_idx = best_idx // num_nodes, best_idx % num_nodes
        
        best_state = PlacementState(
            s1=tree.embeddings.s1[batch_idx, node_idx],
            s2=tree.embeddings.s2[batch_idx, node_idx],
            orientations=tree.embeddings.orientations[batch_idx, node_idx],
            step=tree.embeddings.step[batch_idx, node_idx]
        )
        return best_state, -best_value
    
    def _extract_top_k_states(self, tree, k):
        """从搜索树中提取每个 batch 的最佳终端状态，返回 top-K 个（向量化，O(1) 同步）"""
        target_step = 3 * self.num_movable
        terminal_mask = tree.embeddings.step == target_step
        masked_values = jnp.where(terminal_mask, tree.node_values, -jnp.inf)
        
        best_node_per_batch = jnp.argmax(masked_values, axis=1)
        best_value_per_batch = jnp.max(masked_values, axis=1)
        
        batch_indices = jnp.arange(masked_values.shape[0])
        all_states = PlacementState(
            s1=tree.embeddings.s1[batch_indices, best_node_per_batch],
            s2=tree.embeddings.s2[batch_indices, best_node_per_batch],
            orientations=tree.embeddings.orientations[batch_indices, best_node_per_batch],
            step=tree.embeddings.step[batch_indices, best_node_per_batch],
        )
        all_hpwls = -best_value_per_batch
        
        top_k_indices = jnp.argsort(all_hpwls)[:k]
        top_states = PlacementState(
            s1=all_states.s1[top_k_indices],
            s2=all_states.s2[top_k_indices],
            orientations=all_states.orientations[top_k_indices],
            step=all_states.step[top_k_indices],
        )
        top_hpwls = all_hpwls[top_k_indices]
        return top_states, top_hpwls, int(k)
    
    def get_coords(self, best_state):
        """获取布局坐标"""
        return self.placer.placement_solver.compute_final_positions(
            best_state.s1, best_state.s2, best_state.orientations
        )
    
    def _calc_boundary_from_terminals(self):
        """从terminal节点计算interposer边界"""
        terminal_mask = self.bench.is_terminal == 1
        tx = jnp.where(terminal_mask, self.bench.x_fixed, 0)
        ty = jnp.where(terminal_mask, self.bench.y_fixed, 0)
        tw = jnp.where(terminal_mask, self.bench.widths, 0)
        th = jnp.where(terminal_mask, self.bench.heights, 0)
        return float(jnp.max(tx + tw)), float(jnp.max(ty + th))
    
    def post_optimize(self, x, y, w, h, pins_dx, pins_dy):
        """后处理优化（参数从config读取）"""
        optimizer = PostOptimizer(self.bench, self.movable_indices)
        return optimizer.optimize_with_annealing(
            x, y, w, h, pins_dx, pins_dy,
            boundary_width=self.boundary_width,
            boundary_height=self.boundary_height,
            max_iterations=self.config.annealing_phases,
            search_points=self.config.search_points,
        )
    
    def plot(self, x, y, w, h, pins_dx, pins_dy, filename, title=None):
        """绘制布局图（受 --no-viz 控制）"""
        hpwl = float(PostOptimizer._compute_hpwl_direct(
            x, y, w, h, self.bench.nets_ptr, self.bench.pins_nodes, pins_dx, pins_dy
        ))
        if self.config.save_visualization:
            path = f"{self.config.output_dir}/{filename}"
            PlacementVisualizer.plot_placement(
                self.bench, jnp.array(x), jnp.array(y), jnp.array(w), jnp.array(h),
                jnp.array(pins_dx), jnp.array(pins_dy), self.movable_indices, path, draw_connections=True
            )
            print(f"  {title or filename}: HPWL={hpwl:.2f} -> {path}")
        else:
            print(f"  {title or filename}: HPWL={hpwl:.2f}")
        return hpwl
    
    def save_tree(self, policy_output):
        """保存搜索树"""
        if self.config.save_tree:
            graph = PlacementVisualizer.convert_tree_to_graph(policy_output.search_tree)
            path = f"{self.config.output_dir}/search_tree.png"
            graph.draw(path, prog="dot")
            print(f"搜索树已保存到: {path}")


def create_config_from_args() -> PlacementConfig:
    """从YAML配置文件 + 命令行参数创建配置
    
    优先级：命令行参数 > YAML文件 > 默认值
    """
    parser = argparse.ArgumentParser(description='MCTS序列对布局器')
    
    # 配置文件
    parser.add_argument('-c', '--config', type=str, default=None,
                        help='YAML配置文件路径（命令行参数可覆盖）')
    
    # 所有参数默认None，只有显式指定时才覆盖YAML
    parser.add_argument('--base-path', default=None, help='数据路径')
    parser.add_argument('--sims', type=int, default=None, help='MCTS模拟次数')
    parser.add_argument('--seed', type=int, default=None, help='随机种子')
    parser.add_argument('--batch', type=int, default=None, help='批处理大小')
    parser.add_argument('--output', default=None, help='输出目录')
    parser.add_argument('--gumbel-scale', type=float, default=None, help='Gumbel缩放因子')
    parser.add_argument('--width', type=float, default=None, help='Interposer宽度')
    parser.add_argument('--height', type=float, default=None, help='Interposer高度')
    parser.add_argument('--search-points', type=int, default=None, help='搜索点数')
    parser.add_argument('--annealing-phases', type=int, default=None, help='退火阶段数')
    parser.add_argument('--no-tree', action='store_true', help='不保存搜索树图')
    parser.add_argument('--no-viz', action='store_true', help='不保存可视化')
    
    args = parser.parse_args()
    
    # 1. 从YAML加载 或 使用默认值
    if args.config:
        config = PlacementConfig.from_yaml(args.config)
        print(f"已加载配置文件: {args.config}")
    else:
        config = PlacementConfig()
    
    # 2. 命令行参数覆盖（只覆盖显式指定的）
    cli = {k.replace('-', '_'): v for k, v in vars(args).items() if k != 'config'}
    config.merge_cli(cli)
    
    config.validate()
    return config


def main():
    """主函数"""
    # 创建配置
    config = create_config_from_args()
    
    # 创建布局运行器
    runner = PlacementRunner(config)
    
    # 加载基准测试（边界在此计算并存储到runner中）
    runner.load_benchmark()
    
    # 运行MCTS
    start = time.time()
    policy_output, best_state = runner.run_mcts()
    print(f"MCTS运行时间: {time.time() - start:.2f}秒")
    
    runner.save_tree(policy_output)
    
    # 提取所有 batch 的最佳终端状态，计算坐标，过滤超出边界的解
    all_states, all_mcts_hpwls, total = runner._extract_top_k_states(
        policy_output.search_tree, config.batch_size
    )

    all_x, all_y, all_w, all_h, all_pdx, all_pdy = jax.vmap(
        lambda s1, s2, ori: runner.placer.placement_solver.compute_final_positions(s1, s2, ori)
    )(all_states.s1, all_states.s2, all_states.orientations)

    mi = jnp.array(runner.movable_indices)
    in_bounds = (
        jnp.all(all_x[:, mi] >= 0, axis=1) &
        jnp.all(all_y[:, mi] >= 0, axis=1) &
        jnp.all(all_x[:, mi] + all_w[:, mi] <= runner.boundary_width, axis=1) &
        jnp.all(all_y[:, mi] + all_h[:, mi] <= runner.boundary_height, axis=1)
    )
    valid_mask = in_bounds & (all_mcts_hpwls > 0) & (all_mcts_hpwls < jnp.inf)
    num_valid = int(jnp.sum(valid_mask))
    num_oob = total - num_valid
    print(f"  有效解: {num_valid}/{total}（删除 {num_oob} 个超出边界/无效解）")

    sorted_hpwls = jnp.where(valid_mask, all_mcts_hpwls, jnp.inf)
    top_k = min(config.batch_size // 5 + 1, num_valid)
    top_indices = jnp.argsort(sorted_hpwls)[:top_k]

    all_x, all_y = all_x[top_indices], all_y[top_indices]
    all_w, all_h = all_w[top_indices], all_h[top_indices]
    all_pdx, all_pdy = all_pdx[top_indices], all_pdy[top_indices]
    top_mcts_hpwls = all_mcts_hpwls[top_indices]
    k = top_k

    # 保留 MCTS 原始坐标用于最终对比可视化
    mcts_raw_x, mcts_raw_y = all_x.copy(), all_y.copy()
    mcts_raw_w, mcts_raw_h = all_w.copy(), all_h.copy()
    mcts_raw_pdx, mcts_raw_pdy = all_pdx.copy(), all_pdy.copy()

    # Phase 1: 批量后处理位置优化
    print(f"\n开始后处理优化（{k} 个候选方案，GPU 并行）...")
    start = time.time()

    optimizer = PostOptimizer(runner.bench, runner.movable_indices)
    all_opt_x, all_opt_y, all_hpwls = optimizer.optimize_batch(
        all_x, all_y, all_w, all_h, all_pdx, all_pdy,
        boundary_width=runner.boundary_width,
        boundary_height=runner.boundary_height,
        max_iterations=config.annealing_phases,
        search_points=config.search_points)

    print(f"后处理优化时间: {time.time() - start:.2f}秒")

    print(f"\n候选结果（按 MCTS HPWL 升序）:")
    running_best_hpwl = float('inf')
    running_best_idx = -1
    for i in range(k):
        mcts_h = float(top_mcts_hpwls[i])
        opt_h = float(all_hpwls[i])
        if opt_h < running_best_hpwl:
            running_best_hpwl = opt_h
            running_best_idx = i
            print(f"  候选 {i+1}/{k}: MCTS={mcts_h:.0f} → PostOpt={opt_h:.0f} ← 最优")
        else:
            print(f"  候选 {i+1}/{k}: MCTS={mcts_h:.0f} → PostOpt={opt_h:.0f}")

    best_idx = running_best_idx

    # Phase 2: 方向优化（对 top-10 方案贪心翻转每个模块方向 + 重新退火）
    top10_count = min(10, k)
    top10_indices = jnp.argsort(all_hpwls)[:top10_count]

    print(f"\n开始方向优化（top {top10_count} 候选）...")
    start = time.time()

    ori_x, ori_y, ori_w, ori_h, ori_pdx, ori_pdy, ori_hpwls = optimizer.optimize_orientations(
        all_opt_x[top10_indices], all_opt_y[top10_indices],
        all_w[top10_indices], all_h[top10_indices],
        all_pdx[top10_indices], all_pdy[top10_indices],
        boundary_width=runner.boundary_width,
        boundary_height=runner.boundary_height,
        search_points=config.search_points,
        annealing_phases=config.annealing_phases)

    print(f"方向优化时间: {time.time() - start:.2f}秒")

    print(f"\n方向优化结果:")
    ori_best_idx = -1
    ori_best_hpwl = float('inf')
    for i in range(top10_count):
        before_h = float(all_hpwls[int(top10_indices[i])])
        after_h = float(ori_hpwls[i])
        if after_h < ori_best_hpwl:
            ori_best_hpwl = after_h
            ori_best_idx = i
            print(f"  方向优化 {i+1}/{top10_count}: PostOpt={before_h:.0f} → OriOpt={after_h:.0f} ← 最优")
        else:
            print(f"  方向优化 {i+1}/{top10_count}: PostOpt={before_h:.0f} → OriOpt={after_h:.0f}")

    # 选全局最优 & 画出各阶段的图
    if ori_best_hpwl < running_best_hpwl:
        trace_idx = int(top10_indices[ori_best_idx])
        print(f"\n方向优化改善: {running_best_hpwl:.0f} → {ori_best_hpwl:.0f}")

        runner.plot(mcts_raw_x[trace_idx], mcts_raw_y[trace_idx],
                    mcts_raw_w[trace_idx], mcts_raw_h[trace_idx],
                    mcts_raw_pdx[trace_idx], mcts_raw_pdy[trace_idx],
                    "stage1_mcts.png", "Stage 1: MCTS")
        runner.plot(all_opt_x[trace_idx], all_opt_y[trace_idx],
                    all_w[trace_idx], all_h[trace_idx],
                    all_pdx[trace_idx], all_pdy[trace_idx],
                    "stage2_postopt.png", "Stage 2: PostOpt")
        runner.plot(ori_x[ori_best_idx], ori_y[ori_best_idx],
                    ori_w[ori_best_idx], ori_h[ori_best_idx],
                    ori_pdx[ori_best_idx], ori_pdy[ori_best_idx],
                    "stage3_oriopt.png", "Stage 3: OriOpt (Final)")
    else:
        runner.plot(mcts_raw_x[best_idx], mcts_raw_y[best_idx],
                    mcts_raw_w[best_idx], mcts_raw_h[best_idx],
                    mcts_raw_pdx[best_idx], mcts_raw_pdy[best_idx],
                    "stage1_mcts.png", "Stage 1: MCTS")
        runner.plot(all_opt_x[best_idx], all_opt_y[best_idx],
                    all_w[best_idx], all_h[best_idx],
                    all_pdx[best_idx], all_pdy[best_idx],
                    "stage2_postopt.png", "Stage 2: PostOpt (Final)")

    print("\n" + "="*60)
    print("完成！")
    print("="*60)


if __name__ == '__main__':
    main()