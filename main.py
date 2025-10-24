"""
主程序模块 - 重构后的MCTS布局算法主程序

该模块提供了完整的MCTS布局算法流程，包括数据加载、算法执行和结果可视化。
"""
from __future__ import annotations

import argparse
import json
import numpy as np
import jax
import jax.numpy as jnp
import mctx

from data_loader import BookshelfLoader
from placement_state import StateManager
from mcts_placer import MCTSPlacer
from hpwl_calculator import HPWLCalculator
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
        
    def load_benchmark(self) -> None:
        """加载基准测试数据"""
        print(f"加载基准测试: {self.config.blocks_path}")
        self.bench = BookshelfLoader.load_bookshelf(
            self.config.blocks_path, 
            self.config.nets_path, 
            self.config.pl_path
        )
        
        # 识别可移动模块
        movable_mask = self.bench.is_terminal == 0
        self.movable_indices = np.where(movable_mask)[0]
        self.num_movable = len(self.movable_indices)
        
        print(f"总节点数: {len(self.bench.names)}")
        print(f"可移动模块: {self.num_movable}")
        print(f"终端/固定节点: {np.sum(self.bench.is_terminal)}")
        print(f"网络数: {len(self.bench.nets_ptr) - 1}")
        
    
    def prepare_modules(self):
        """准备模块排序"""
        # 按面积排序可移动模块（降序）
        areas = self.bench.widths[self.movable_indices] * self.bench.heights[self.movable_indices]
        sorted_order = np.argsort(-areas)  # 降序
        ordered_modules = self.movable_indices[sorted_order]
        return jnp.array(ordered_modules)
    
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
            jnp.array(self.bench.widths),
            jnp.array(self.bench.heights),
            jnp.array(self.bench.nets_ptr),
            jnp.array(self.bench.pins_nodes),
            jnp.array(self.bench.pins_dx),
            jnp.array(self.bench.pins_dy),
            self.num_movable,
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
            gumbel_scale=self.config.gumbel_scale
        )
        
        # 选择访问次数最高的动作
        action = jnp.argmax(policy_output.action_weights)
        print(f"  Selected action: {int(action)}")
        
        return policy_output, action, placer
    
    def visualize_results(self, policy_output) -> None:
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
            # 注意：由于代码重构，完整的布局计算和可视化需要进一步实现
            # 这里只是保存了搜索树图
            print("Note: Complete layout visualization needs further implementation")


def create_config_from_args() -> PlacementConfig:
    """从命令行参数创建配置"""
    parser = argparse.ArgumentParser(description='重构的MCTS序列对布局器（终端奖励 = -HPWL）')
    parser.add_argument('--blocks', default="./data/apte.blocks", help='.blocks文件路径')
    parser.add_argument('--nets', default="./data/apte.nets", help='.nets文件路径')
    parser.add_argument('--pl', default="./data/apte.pl", help='.pl文件路径（用于终端）')
    parser.add_argument('--sims', type=int, default=100, help='MCTS模拟次数')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--batch', type=int, default=1, help='并行根节点的批处理大小')
    parser.add_argument('--output', default=".", help='输出目录')
    parser.add_argument('--gumbel-scale', type=float, default=1.0, help='Gumbel缩放因子')
    parser.add_argument('--no-tree', action='store_true', help='不保存搜索树图')
    parser.add_argument('--no-viz', action='store_true', help='不保存可视化')

    args = parser.parse_args()
    
    config = PlacementConfig(
        blocks_path=args.blocks,
        nets_path=args.nets,
        pl_path=args.pl,
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
    
    # 计算初始HPWL
    print("计算初始HPWL...")
    # 初始布局：终端在固定位置，可移动模块在(0,0)
    movable_mask = runner.bench.is_terminal == 0
    initial_hpwl = HPWLCalculator.calculate_hpwl(
        jnp.where(movable_mask, 0.0, runner.bench.x_fixed),
        jnp.where(movable_mask, 0.0, runner.bench.y_fixed),
        jnp.array(runner.bench.widths),
        jnp.array(runner.bench.heights),
        jnp.array(runner.bench.nets_ptr),
        jnp.array(runner.bench.pins_nodes),
        jnp.array(runner.bench.pins_dx),
        jnp.array(runner.bench.pins_dy)
    )
    print(f"初始HPWL: {float(initial_hpwl):.2f}")
    
    # 运行MCTS
    policy_output, action, placer = runner.run_mcts()
    
    # 可视化结果
    runner.visualize_results(policy_output)
    
    print("\n" + "=" * 60)
    print("MCTS布局完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
