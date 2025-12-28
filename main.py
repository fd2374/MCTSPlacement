"""
主程序模块 - 重构后的MCTS布局算法主程序

该模块提供了完整的MCTS布局算法流程，包括数据加载、算法执行和结果可视化。
"""
from __future__ import annotations

import argparse
import json
import jax
import jax.numpy as jnp
import mctx
import functools

from data_loader import BookshelfLoader
from placement_state import StateManager, PlacementState
from mcts_placer import MCTSPlacer
from placement_solver import PlacementSolver
from visualizer import PlacementVisualizer
from config import PlacementConfig


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
        """加载基准测试数据"""
        self.bench = BookshelfLoader.load_bookshelf_from_base_path(self.config.base_path)
        
        # 识别可移动模块
        movable_mask = self.bench.is_terminal == 0
        self.movable_indices = jnp.where(movable_mask)[0]
        self.num_movable = len(self.movable_indices)
        
        print(f"总节点数: {len(self.bench.names)}")
        print(f"可移动模块: {self.num_movable}")
        print(f"终端/固定节点: {jnp.sum(self.bench.is_terminal)}")
        print(f"网络数: {len(self.bench.nets_ptr) - 1}")
        
    
    def prepare_modules(self):
        """准备模块排序"""
        # 按面积排序可移动模块（降序）
        areas = self.bench.widths[self.movable_indices] * self.bench.heights[self.movable_indices]
        sorted_order = jnp.argsort(-areas)  # 降序
        ordered_modules = self.movable_indices[sorted_order]
        return jnp.array(ordered_modules)
    
    functools.partial(jax.jit, static_argnums=(0,))
    def run_mcts(self) -> tuple:
        """运行MCTS算法
        
        Returns:
            tuple: (policy_output, action) 策略输出和选择的动作
        """
        print(f"\n运行MCTS，{self.config.num_simulations}次模拟...")
        
        # 准备模块
        ordered_modules = self.prepare_modules()
        
        # 创建初始状态
        initial_state = StateManager.create_initial_state(self.num_movable)
        
        # 创建MCTS布局器
        placer = MCTSPlacer(
            self.bench,
            jnp.array(self.movable_indices),
            ordered_modules
        )
        
        # 运行MCTS
        rng_key = jax.random.PRNGKey(self.config.seed)
        rng_key, subkey = jax.random.split(rng_key)
        
        recurrent_fn = jax.vmap(placer.create_recurrent_fn(), (None, None, 0, 0))
        root = jax.vmap(placer.root_fn, (None, None, 0))(
            initial_state, placer.max_actions, jax.random.split(subkey, self.config.batch_size)
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
        
        # 保存placer作为实例属性，以便后续使用
        self.placer = placer
        
        # 从搜索树中提取所有终端状态和对应的奖励
        best_terminal_state, best_reward = self._extract_best_terminal_state(
            policy_output.search_tree, placer
        )
        
        print(f"  Best terminal reward: {float(best_reward):.4f}")
        print(f"  Best terminal state: s1={best_terminal_state.s1}, s2={best_terminal_state.s2}")
        print(f"  Best terminal state: orientations={best_terminal_state.orientations}")
        
        return policy_output, best_terminal_state, placer
    
    def _extract_best_terminal_state(self, search_tree, placer):
        """从搜索树中提取奖励最高的终端状态"""
        # 遍历搜索树找到所有终端节点
        best_reward = float('-inf')
        best_state = None
        
        # 获取搜索树信息
        tree = search_tree
        num_nodes = tree.num_simulations
        
        # 遍历所有节点，寻找终端状态
        for batch_idx in range(self.config.batch_size):
            for node_idx in range(num_nodes):
                # 检查是否为终端节点（没有子节点或访问次数为0）
                if tree.embeddings.step[batch_idx, node_idx] != 3 * self.num_movable:
                    continue
                    
                # 获取节点值作为奖励
                node_reward = float(tree.node_values[batch_idx, node_idx])
                
                # 如果这个节点的奖励更高，记录它
                if node_reward > best_reward:
                    best_reward = node_reward
                    # 这里我们需要从节点索引重构状态
                    # 由于MCTS树结构复杂，我们使用一个简化的方法
                    # 实际应用中可能需要更复杂的状态重构逻辑
                    best_state = PlacementState(
                        s1=tree.embeddings.s1[batch_idx, node_idx],
                        s2=tree.embeddings.s2[batch_idx, node_idx],
                        orientations=tree.embeddings.orientations[batch_idx, node_idx],
                        step=tree.embeddings.step[batch_idx, node_idx]
                    )
        
        # 如果没有找到终端状态，使用默认状态
        if best_state is None:
            print("Warning: No terminal states found, using default state")
            best_state = placer.state_manager.create_initial_state(self.num_movable)
            best_reward = 0.0
            
        return best_state, -best_reward
    
    def visualize_results(self, policy_output, best_state) -> None:
        """可视化结果"""
        if self.config.save_tree:
            # 保存搜索树图
            graph = PlacementVisualizer.convert_tree_to_graph(
                policy_output.search_tree, 
            )
            tree_path = f"{self.config.output_dir}/search_tree.png"
            print(f"Saving search tree diagram to: {tree_path}")
            graph.draw(tree_path, prog="dot")
        
        if self.config.save_visualization:
            # 从最佳状态计算最终布局
            self._visualize_best_placement(best_state)
    
    def _visualize_best_placement(self, best_state):
        """可视化最佳布局结果"""
        print("\n" + "="*50)
        print("最佳布局结果")
        print("="*50)

        x_coords, y_coords, w_final, h_final, pins_dx, pins_dy = self.placer.placement_solver.compute_final_positions(
            best_state.s1, best_state.s2, best_state.orientations
        )

        final_hpwl = self.placer.placement_solver.compute_hpwl(
            best_state.s1, best_state.s2, best_state.orientations
        )

        # compute_final_positions 已经返回了包含所有模块（包括固定终端）的完整坐标数组
        full_x_coords = x_coords
        full_y_coords = y_coords
        
        # 绘制布局图
        PlacementVisualizer.plot_placement(
            self.bench, 
            jnp.array(full_x_coords), 
            jnp.array(full_y_coords),
            jnp.array(w_final),
            jnp.array(h_final),
            jnp.array(pins_dx),
            jnp.array(pins_dy),
            self.movable_indices,
            f"{self.config.output_dir}/best_placement.png",
            draw_connections=True
        )
        
        print(f"最佳布局图已保存到: {self.config.output_dir}/best_placement.png")


def create_config_from_args() -> PlacementConfig:
    """从命令行参数创建配置"""
    parser = argparse.ArgumentParser(description='重构的MCTS序列对布局器（终端奖励 = -HPWL）')
    parser.add_argument('--base-path', default="./data/apte", help='基础路径（会自动添加.blocks, .nets, .pl后缀）')
    parser.add_argument('--sims', type=int, default=100, help='MCTS模拟次数')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--batch', type=int, default=1, help='并行根节点的批处理大小')
    parser.add_argument('--output', default=".", help='输出目录')
    parser.add_argument('--gumbel-scale', type=float, default=1.0, help='Gumbel缩放因子')
    parser.add_argument('--no-tree', action='store_true', help='不保存搜索树图')
    parser.add_argument('--no-viz', action='store_true', help='不保存可视化')

    args = parser.parse_args()
    
    config = PlacementConfig(
        base_path=args.base_path,
        num_simulations=args.sims,
        seed=args.seed,
        batch_size=args.batch,
        output_dir=args.output,
        gumbel_scale=args.gumbel_scale,
        save_tree=not args.no_tree,
        save_visualization=not args.no_viz
    )
    
    config.validate()
    return config


def main():
    """主函数"""
    # 创建配置
    config = create_config_from_args()
    
    # 创建布局运行器
    runner = PlacementRunner(config)
    
    # 加载基准测试
    runner.load_benchmark()
    
    # with jax.profiler.trace("./tmp/jax-mcts-trace", create_perfetto_link=True):
        # 运行MCTS
    import time
    start_time = time.time()
    policy_output, best_state, placer = runner.run_mcts()
    end_time = time.time()
    print(f"MCTS运行时间: {end_time - start_time}秒")
    # 可视化结果
    runner.visualize_results(policy_output, best_state)
    
    print("\n" + "=" * 60)
    print("MCTS布局完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
