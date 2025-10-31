"""
可视化模块 - 处理布局结果的可视化
"""
from __future__ import annotations

import numpy as np
import pygraphviz
import mctx
import chex
import jax.numpy as jnp
import jax
from typing import Optional, Sequence

from data_loader import BookshelfData


class PlacementVisualizer:
    """布局可视化器"""
    
    @staticmethod
    def plot_placement(bench: BookshelfData, x: np.ndarray, y: np.ndarray, widths: np.ndarray, heights: np.ndarray, pins_dx: np.ndarray, pins_dy: np.ndarray,
                      movable_indices: np.ndarray, output_path: str = "output_placement.png", draw_connections: bool = False):
        """绘制最终布局，包含模块和网络"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
        except ImportError:
            print("Matplotlib未安装。跳过可视化。")
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        
        # 绘制模块
        for i, (name, w, h, is_term) in enumerate(zip(bench.names, widths, 
                                                      heights, bench.is_terminal)):
            xi, yi = x[i], y[i]
            if np.isnan(xi) or np.isnan(yi):
                continue
            
            # 可移动模块和终端的不同颜色
            color = 'lightblue' if not is_term else 'lightgray'
            edgecolor = 'blue' if not is_term else 'gray'
            
            rect = patches.Rectangle((xi, yi), w, h, 
                                   linewidth=1.5, 
                                   edgecolor=edgecolor, 
                                   facecolor=color,
                                   alpha=0.7)
            ax.add_patch(rect)
        
        # 绘制网络连接
        centers_x = x + 0.5 * widths
        centers_y = y + 0.5 * heights
        
        num_nets = len(bench.nets_ptr) - 1
        for net_idx in range(num_nets):
            start = bench.nets_ptr[net_idx]
            end = bench.nets_ptr[net_idx + 1]
            
            # 获取此网络的所有引脚位置
            pin_x = []
            pin_y = []
            for pin_idx in range(start, end):
                node_idx = bench.pins_nodes[pin_idx]
                dx_pct = pins_dx[pin_idx]
                dy_pct = pins_dy[pin_idx]
                
                # 计算带偏移的引脚位置
                px = centers_x[node_idx] + (dx_pct / 100.0) * widths[node_idx]
                py = centers_y[node_idx] + (dy_pct / 100.0) * heights[node_idx]
                
                if not np.isnan(px) and not np.isnan(py):
                    pin_x.append(px)
                    pin_y.append(py)
            
            # 绘制连接此网络中所有引脚的线（连接到物理重心）
            if len(pin_x) > 1 and draw_connections:
                # 计算物理重心
                center_x = np.mean(pin_x)
                center_y = np.mean(pin_y)
                
                # 绘制从每个引脚到重心的连接线
                for i in range(len(pin_x)):
                    ax.plot([pin_x[i], center_x], [pin_y[i], center_y], 
                           'orange', alpha=0.4, linewidth=0.8)
                
                # 绘制重心点
                ax.scatter(center_x, center_y, c='red', s=20, zorder=6, alpha=0.8, marker='x')
                
                # 绘制引脚点
            ax.scatter(pin_x, pin_y, c='orange', s=10, zorder=5, alpha=0.6)
        
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_title('Final Placement', fontsize=14)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n布局可视化已保存到: {output_path}")
        plt.close()

    @staticmethod
    def convert_tree_to_graph(tree: mctx.Tree,
                            action_labels: Optional[Sequence[str]] = None,
                            batch_index: int = 0) -> pygraphviz.AGraph:
        """将搜索树转换为Graphviz图"""
        chex.assert_rank(tree.node_values, 2)
        batch_size = tree.node_values.shape[0]
        
        if action_labels is None:
            action_labels = range(tree.num_actions)
        elif len(action_labels) != tree.num_actions:
            raise ValueError(
                f"action_labels {action_labels} has wrong number of actions "
                f"({len(action_labels)}). Expected {tree.num_actions}.")

        def node_to_str(node_i, reward=0, discount=1):
            base_str = (f"Node {node_i}\n"
                       f"Reward: {reward:.2f}\n"
                       f"Discount: {discount:.2f}\n"
                       f"Value: {tree.node_values[batch_index, node_i]:.2f}\n"
                       f"Visits: {tree.node_visits[batch_index, node_i]}\n")
            

            s1_list = tree.embeddings.s1[batch_index, node_i]
            s2_list = tree.embeddings.s2[batch_index, node_i]
            orient_list = tree.embeddings.orientations[batch_index, node_i]
            step = tree.embeddings.step[batch_index, node_i]
            
            # 格式化数组显示，限制显示长度
            s1_str = "[" + ",".join([str(x) for x in s1_list[:8]]) + ("..." if len(s1_list) > 8 else "") + "]"
            s2_str = "[" + ",".join([str(x) for x in s2_list[:8]]) + ("..." if len(s2_list) > 8 else "") + "]"
            orient_str = "[" + ",".join([str(x) for x in orient_list[:8]]) + ("..." if len(orient_list) > 8 else "") + "]"
            
            base_str += f"S1: {s1_str}\n"
            base_str += f"S2: {s2_str}\n"
            base_str += f"Orient: {orient_str}\n"
            base_str += f"Step: {int(step)}\n"
            
            return base_str

        def edge_to_str(node_i, a_i):
            node_index = jnp.full([batch_size], node_i)
            probs = jax.nn.softmax(tree.children_prior_logits[batch_index, node_i])
            return (f"Action {action_labels[a_i]}\n"
                    f"Q: {tree.qvalues(node_index)[batch_index, a_i]:.2f}\n"
                    f"p: {probs[a_i]:.2f}\n")

        graph = pygraphviz.AGraph(directed=True)

        # 添加根节点
        graph.add_node(0, label=node_to_str(node_i=0), color="green")
        
        # 添加所有其他节点并连接它们
        for node_i in range(tree.num_simulations):
            for a_i in range(tree.num_actions):
                # 子节点索引，如果未展开则为-1
                children_i = tree.children_index[batch_index, node_i, a_i]
                if children_i >= 0:
                    graph.add_node(
                        children_i,
                        label=node_to_str(
                            node_i=children_i,
                            reward=tree.children_rewards[batch_index, node_i, a_i],
                            discount=tree.children_discounts[batch_index, node_i, a_i]),
                        color="red")
                    graph.add_edge(node_i, children_i, label=edge_to_str(node_i, a_i))

        return graph
