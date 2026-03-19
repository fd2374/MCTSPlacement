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
from nn_model import init_params as nn_init_params, load_params as nn_load_params


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
    
    def run_mcts(self, use_nn: bool = False) -> tuple:
        """运行MCTS算法"""
        print(f"\n运行MCTS，{self.config.num_simulations}次模拟"
              f"{'（神经网络模式）' if use_nn else ''}...")
        
        areas = self.bench.widths[self.movable_indices] * self.bench.heights[self.movable_indices]
        ordered_modules = self.movable_indices[jnp.argsort(-areas)]
        
        self.placer = MCTSPlacer(
            self.bench, jnp.array(self.movable_indices), ordered_modules,
            use_nn=use_nn,
            boundary_width=self.boundary_width,
            boundary_height=self.boundary_height,
        )
        
        rng_key = jax.random.PRNGKey(self.config.seed)
        rng_key, subkey, nn_key = jax.random.split(rng_key, 3)
        
        if use_nn and self.config.nn_weights:
            nn_params = nn_load_params(self.config.nn_weights)
            print(f"  已加载网络权重: {self.config.nn_weights}")
        elif use_nn:
            nn_params = nn_init_params(self.num_movable, nn_key)
            print(f"  使用随机初始化网络权重")
        else:
            nn_params = None
        
        initial_state = StateManager.create_initial_state(self.num_movable)
        recurrent_fn = jax.vmap(self.placer.create_recurrent_fn(), (None, None, 0, 0))
        root = jax.vmap(self.placer.root_fn, (None, None, None, 0))(
            nn_params, initial_state, self.placer.max_actions,
            jax.random.split(subkey, self.config.batch_size)
        )
        
        policy_output = mctx.gumbel_muzero_policy(
            params=nn_params,
            rng_key=rng_key,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=self.config.num_simulations,
            max_depth=3 * self.num_movable,
            gumbel_scale=self.config.gumbel_scale,
            qtransform=functools.partial(mctx.qtransform_completed_by_mix_value)
        )
        
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
            initial_step=self.config.initial_step,
            final_step=self.config.final_step,
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
    parser.add_argument('--initial-step', type=float, default=None, help='后处理初始步长')
    parser.add_argument('--final-step', type=float, default=None, help='后处理最终步长')
    parser.add_argument('--search-points', type=int, default=None, help='搜索点数')
    parser.add_argument('--annealing-phases', type=int, default=None, help='退火阶段数')
    parser.add_argument('--no-tree', action='store_true', help='不保存搜索树图')
    parser.add_argument('--no-viz', action='store_true', help='不保存可视化')
    parser.add_argument('--use-nn', action='store_true', help='使用神经网络替代rollout')
    parser.add_argument('--nn-weights', type=str, default=None, help='训练好的网络权重路径(JSON)')
    
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
    policy_output, best_state = runner.run_mcts(use_nn=config.use_nn)
    print(f"MCTS运行时间: {time.time() - start:.2f}秒")
    
    runner.save_tree(policy_output)
    
    # 提取 top-K 候选方案（每个 batch 的最佳终端状态）
    top_k = min(config.batch_size, 1000)
    top_states, top_mcts_hpwls, k = runner._extract_top_k_states(
        policy_output.search_tree, top_k
    )
    
    # 显示 MCTS 最佳结果（优化前）
    x, y, w, h, pins_dx, pins_dy = runner.get_coords(best_state)
    
    # 对所有候选方案运行后处理优化，取最优
    print(f"\n开始后处理优化（{k} 个候选方案）...")
    start = time.time()
    
    best_hpwl = float('inf')
    best_result = None
    
    for i in range(k):
        mcts_hpwl = float(top_mcts_hpwls[i])
        if jnp.isinf(top_mcts_hpwls[i]) or jnp.isnan(top_mcts_hpwls[i]):
            continue
        
        state_i = PlacementState(
            s1=top_states.s1[i], s2=top_states.s2[i],
            orientations=top_states.orientations[i], step=top_states.step[i],
        )
        xi, yi, wi, hi, pdx, pdy = runner.get_coords(state_i)
        opt_x, opt_y, hpwl = runner.post_optimize(xi, yi, wi, hi, pdx, pdy)
        
        tag = ""
        if hpwl < best_hpwl:
            best_hpwl = hpwl
            best_result = (opt_x, opt_y, wi, hi, pdx, pdy)
            tag = " ← 最优"
        print(f"  候选 {i+1}/{k}: MCTS={mcts_hpwl:.0f} → PostOpt={hpwl:.0f}{tag}")
    
    print(f"后处理优化时间: {time.time() - start:.2f}秒")
    
    if best_result is None:
        print("警告: 没有有效候选，使用全局最佳 MCTS 结果进行后处理")
        x, y, w, h, pins_dx, pins_dy = runner.get_coords(best_state)
        opt_x, opt_y, hpwl = runner.post_optimize(x, y, w, h, pins_dx, pins_dy)
        best_result = (opt_x, opt_y, w, h, pins_dx, pins_dy)
    
    opt_x, opt_y, w, h, pins_dx, pins_dy = best_result
    runner.plot(opt_x, opt_y, w, h, pins_dx, pins_dy, "best_placement.png", "优化后")
    
    print("\n" + "="*60)
    print("完成！")
    print("="*60)


if __name__ == '__main__':
    main()