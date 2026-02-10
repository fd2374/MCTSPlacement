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
        areas = self.bench.widths[self.movable_indices] * self.bench.heights[self.movable_indices]
        ordered_modules = self.movable_indices[jnp.argsort(-areas)]
        
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
        """绘制布局图"""
        hpwl = float(PostOptimizer._compute_hpwl_direct(
            x, y, w, h, self.bench.nets_ptr, self.bench.pins_nodes, pins_dx, pins_dy
        ))
        path = f"{self.config.output_dir}/{filename}"
        PlacementVisualizer.plot_placement(
            self.bench, jnp.array(x), jnp.array(y), jnp.array(w), jnp.array(h),
            jnp.array(pins_dx), jnp.array(pins_dy), self.movable_indices, path, draw_connections=True
        )
        print(f"  {title or filename}: HPWL={hpwl:.2f} -> {path}")
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
    
    # 获取坐标
    x, y, w, h, pins_dx, pins_dy = runner.get_coords(best_state)
    
    print("\n" + "="*50)
    print("布局结果")
    print("="*50)
    
    # 画优化前
    runner.plot(x, y, w, h, pins_dx, pins_dy, "before_opt.png", "优化前")
    
    # 后处理优化
    print("\n开始后处理优化...")
    start = time.time()
    opt_x, opt_y, _ = runner.post_optimize(x, y, w, h, pins_dx, pins_dy)
    print(f"后处理优化时间: {time.time() - start:.2f}秒")
    
    # 画优化后
    runner.plot(opt_x, opt_y, w, h, pins_dx, pins_dy, "after_opt.png", "优化后")
    
    print("\n" + "="*60)
    print("完成！")
    print("="*60)


if __name__ == '__main__':
    main()
