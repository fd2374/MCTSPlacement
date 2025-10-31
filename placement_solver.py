"""
布局求解器模块 - 将状态转换为最终坐标并计算HPWL
"""
from __future__ import annotations

import jax.numpy as jnp
import jax
import functools
from typing import Tuple

from sequence_pair import SequencePairSolver


class PlacementSolver:
    """布局求解器 - 将状态转换为最终坐标并计算HPWL"""
    
    def __init__(self, bench, movable_indices: jnp.ndarray):
        """
        初始化布局求解器
        
        Args:
            bench: BookshelfData对象
            movable_indices: 可移动模块的索引
        """
        self.bench = bench
        self.movable_indices = movable_indices
        self.num_movable = len(movable_indices)
        
        # 预计算可移动模块的宽度和高度
        self.movable_widths = jnp.array(bench.widths)[movable_indices]
        self.movable_heights = jnp.array(bench.heights)[movable_indices]
        
        # 预计算固定终端数据
        self.fixed_x = jnp.array(bench.x_fixed)
        self.fixed_y = jnp.array(bench.y_fixed)
        self.is_terminal = jnp.array(bench.is_terminal)
    
    @staticmethod
    @jax.jit
    def _calculate_hpwl_core(x: jnp.ndarray, y: jnp.ndarray,
                           widths: jnp.ndarray, heights: jnp.ndarray,
                           nets_ptr: jnp.ndarray, pins_nodes: jnp.ndarray,
                           pins_dx: jnp.ndarray, pins_dy: jnp.ndarray) -> jnp.ndarray:
        """
        计算HPWL（半周长线长）的核心函数
        
        Args:
            x, y: 模块位置
            widths, heights: 模块尺寸
            nets_ptr: 网络指针
            pins_nodes: 引脚节点索引
            pins_dx, pins_dy: 引脚偏移（百分比）
            
        Returns:
            HPWL值
        """
        centers_x = x + 0.5 * widths
        centers_y = y + 0.5 * heights

        pw = widths[pins_nodes]
        ph = heights[pins_nodes]

        node_x = centers_x[pins_nodes]
        node_y = centers_y[pins_nodes]

        pin_x = node_x + (pins_dx / 100.0) * pw
        pin_y = node_y + (pins_dy / 100.0) * ph

        num_nets = nets_ptr.shape[0] - 1
        counts = nets_ptr[1:] - nets_ptr[:-1]
        seg_ids = jnp.repeat(jnp.arange(num_nets, dtype=jnp.int32), counts, 
                           total_repeat_length=pins_nodes.shape[0])

        maxx = jax.ops.segment_max(pin_x, seg_ids, num_segments=num_nets)
        minx = jax.ops.segment_min(pin_x, seg_ids, num_segments=num_nets)
        maxy = jax.ops.segment_max(pin_y, seg_ids, num_segments=num_nets)
        miny = jax.ops.segment_min(pin_y, seg_ids, num_segments=num_nets)
        
        hpwl = jnp.sum((maxx - minx) + (maxy - miny))
        return hpwl
    
    def _apply_orientations(self, orientations: jnp.ndarray):
        """应用方向到宽度和高度"""
        w = self.movable_widths
        h = self.movable_heights
        
        # 为E/W方向（1, 3）交换宽度/高度
        should_swap = (orientations == 1) | (orientations == 3)
        w_final = jnp.where(should_swap, h, w)
        h_final = jnp.where(should_swap, w, h)
        
        # 对于每个引脚，根据其所属模块的方向决定是否翻转偏移量
        pin_orientations = orientations[self.bench.pins_nodes]
        should_swap_pins_dx = (pin_orientations == 2) | (pin_orientations == 3)
        should_swap_pins_dy = (pin_orientations == 1) | (pin_orientations == 2)
        
        pins_dx = jnp.where(should_swap_pins_dx, -self.bench.pins_dx, self.bench.pins_dx)
        pins_dy = jnp.where(should_swap_pins_dy, self.bench.pins_dy, -self.bench.pins_dy)
        
        return w_final, h_final, pins_dx, pins_dy
    
    def compute_final_positions(self, s1: jnp.ndarray, s2: jnp.ndarray, 
                               orientations: jnp.ndarray):
        """
        计算最终位置坐标
        
        Args:
            s1, s2: 序列对
            orientations: 方向数组
            
        Returns:
            (x, y): 最终坐标
        """
        # 应用方向
        w_final, h_final, pins_dx, pins_dy = self._apply_orientations(orientations)
        
        # 从序列对获取位置
        x_mov, y_mov = SequencePairSolver.seqpair_to_positions(
            s1, s2, w_final, h_final
        )
        
        # 与固定终端合并
        x = jnp.zeros_like(jnp.array(self.bench.widths))
        y = jnp.zeros_like(jnp.array(self.bench.heights))
        
        # 设置可移动模块的坐标
        x = x.at[self.movable_indices].set(x_mov)
        y = y.at[self.movable_indices].set(y_mov)
        
        # 设置固定终端的坐标
        x = jnp.where(self.is_terminal == 1, self.fixed_x, x)
        y = jnp.where(self.is_terminal == 1, self.fixed_y, y)

        w_final_all = self.bench.widths.at[self.movable_indices].set(w_final)
        h_final_all = self.bench.heights.at[self.movable_indices].set(h_final)

        return x, y, w_final_all, h_final_all, pins_dx, pins_dy
    
    def compute_hpwl(self, s1: jnp.ndarray, s2: jnp.ndarray, 
                     orientations: jnp.ndarray) -> jnp.ndarray:
        """
        计算给定状态的HPWL
        
        Args:
            s1, s2: 序列对
            orientations: 方向数组
            
        Returns:
            HPWL值
        """
        x, y, w_final, h_final, pins_dx, pins_dy = self.compute_final_positions(s1, s2, orientations)
        return self._calculate_hpwl_core(x, y, w_final, h_final, self.bench.nets_ptr, 
        self.bench.pins_nodes, pins_dx, pins_dy)
