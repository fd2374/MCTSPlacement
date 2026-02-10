"""
后处理优化模块 - 对MCTS布局结果进行局部优化

在得到初始布局后，逐个调整每个模块的位置，
使其在不超边界、不重叠的约束下最小化总wirelength。
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import Tuple, Optional
import functools


class PostOptimizer:
    """后处理优化器
    
    对布局结果进行局部优化，通过逐个调整模块位置来减小HPWL。
    """
    
    def __init__(self, bench, movable_indices: jnp.ndarray):
        """初始化后处理优化器
        
        Args:
            bench: BookshelfData对象
            movable_indices: 可移动模块的索引
        """
        self.bench = bench
        self.movable_indices = movable_indices
        self.num_movable = len(movable_indices)
        
        # 预计算网络相关数据
        self.nets_ptr = bench.nets_ptr
        self.pins_nodes = bench.pins_nodes
        self.pins_dx = bench.pins_dx
        self.pins_dy = bench.pins_dy
        
        # 固定终端数据
        self.fixed_x = jnp.array(bench.x_fixed)
        self.fixed_y = jnp.array(bench.y_fixed)
        self.is_terminal = jnp.array(bench.is_terminal)
    
    @staticmethod
    @functools.partial(jax.jit, static_argnums=())
    def _compute_hpwl_direct(x: jnp.ndarray, y: jnp.ndarray,
                             widths: jnp.ndarray, heights: jnp.ndarray,
                             nets_ptr: jnp.ndarray, pins_nodes: jnp.ndarray,
                             pins_dx: jnp.ndarray, pins_dy: jnp.ndarray) -> jnp.ndarray:
        """直接从坐标计算HPWL"""
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
    
    def _check_boundary(self, x: float, y: float, w: float, h: float,
                        boundary_width: float, boundary_height: float) -> bool:
        """检查模块是否在边界内"""
        return (x >= 0 and y >= 0 and 
                x + w <= boundary_width and 
                y + h <= boundary_height)
    
    def _check_overlap(self, module_idx: int, new_x: float, new_y: float,
                       new_w: float, new_h: float,
                       all_x: jnp.ndarray, all_y: jnp.ndarray,
                       all_w: jnp.ndarray, all_h: jnp.ndarray) -> bool:
        """检查模块是否与其他模块重叠
        
        Returns:
            True 如果有重叠，False 如果没有重叠
        """
        for i in range(self.num_movable):
            if i == module_idx:
                continue
            
            other_idx = int(self.movable_indices[i])
            ox, oy = float(all_x[other_idx]), float(all_y[other_idx])
            ow, oh = float(all_w[other_idx]), float(all_h[other_idx])
            
            # 检查矩形是否重叠
            if not (new_x + new_w <= ox or ox + ow <= new_x or
                    new_y + new_h <= oy or oy + oh <= new_y):
                return True
        
        return False
    
    def _compute_overlap_area(self, x1: float, y1: float, w1: float, h1: float,
                              x2: float, y2: float, w2: float, h2: float) -> float:
        """计算两个矩形的重叠面积"""
        overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        return overlap_x * overlap_y
    
    def _get_total_overlap(self, x: jnp.ndarray, y: jnp.ndarray,
                           w: jnp.ndarray, h: jnp.ndarray) -> float:
        """计算所有可移动模块之间的总重叠面积"""
        total_overlap = 0.0
        for i in range(self.num_movable):
            idx_i = int(self.movable_indices[i])
            xi, yi = float(x[idx_i]), float(y[idx_i])
            wi, hi = float(w[idx_i]), float(h[idx_i])
            
            for j in range(i + 1, self.num_movable):
                idx_j = int(self.movable_indices[j])
                xj, yj = float(x[idx_j]), float(y[idx_j])
                wj, hj = float(w[idx_j]), float(h[idx_j])
                
                total_overlap += self._compute_overlap_area(xi, yi, wi, hi, xj, yj, wj, hj)
        
        return total_overlap
    
    def separate_overlaps(self, x: jnp.ndarray, y: jnp.ndarray,
                          widths: jnp.ndarray, heights: jnp.ndarray,
                          boundary_width: float, boundary_height: float,
                          max_iterations: int = 50) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """分离重叠的模块
        
        使用力导向方法将重叠的模块推开。
        """
        opt_x = jnp.array(x, dtype=jnp.float32)
        opt_y = jnp.array(y, dtype=jnp.float32)
        
        initial_overlap = self._get_total_overlap(opt_x, opt_y, widths, heights)
        if initial_overlap == 0:
            print("  无重叠，跳过分离步骤")
            return opt_x, opt_y
        
        print(f"  初始重叠面积: {initial_overlap:.2f}")
        
        for iteration in range(max_iterations):
            moved = False
            
            for i in range(self.num_movable):
                idx_i = int(self.movable_indices[i])
                xi, yi = float(opt_x[idx_i]), float(opt_y[idx_i])
                wi, hi = float(widths[idx_i]), float(heights[idx_i])
                
                # 计算与其他模块的推力
                force_x, force_y = 0.0, 0.0
                
                for j in range(self.num_movable):
                    if i == j:
                        continue
                    
                    idx_j = int(self.movable_indices[j])
                    xj, yj = float(opt_x[idx_j]), float(opt_y[idx_j])
                    wj, hj = float(widths[idx_j]), float(heights[idx_j])
                    
                    # 计算重叠
                    overlap_area = self._compute_overlap_area(xi, yi, wi, hi, xj, yj, wj, hj)
                    if overlap_area > 0:
                        # 计算推开方向
                        ci_x, ci_y = xi + wi/2, yi + hi/2
                        cj_x, cj_y = xj + wj/2, yj + hj/2
                        
                        dx = ci_x - cj_x
                        dy = ci_y - cj_y
                        
                        # 归一化并根据重叠面积加权
                        dist = max(1e-6, (dx**2 + dy**2)**0.5)
                        force_x += dx / dist * (overlap_area ** 0.5)
                        force_y += dy / dist * (overlap_area ** 0.5)
                
                # 应用推力（小步移动）
                if abs(force_x) > 1e-6 or abs(force_y) > 1e-6:
                    step_size = min(wi, hi) * 0.3
                    force_mag = (force_x**2 + force_y**2)**0.5
                    
                    new_x = xi + force_x / force_mag * step_size
                    new_y = yi + force_y / force_mag * step_size
                    
                    # 边界约束
                    new_x = max(0, min(new_x, boundary_width - wi))
                    new_y = max(0, min(new_y, boundary_height - hi))
                    
                    opt_x = opt_x.at[idx_i].set(new_x)
                    opt_y = opt_y.at[idx_i].set(new_y)
                    moved = True
            
            current_overlap = self._get_total_overlap(opt_x, opt_y, widths, heights)
            
            if iteration % 10 == 0:
                print(f"    分离迭代 {iteration}: 重叠面积 = {current_overlap:.2f}")
            
            if current_overlap == 0:
                print(f"  分离完成！迭代次数: {iteration + 1}")
                break
            
            if not moved:
                break
        
        final_overlap = self._get_total_overlap(opt_x, opt_y, widths, heights)
        print(f"  分离后重叠面积: {final_overlap:.2f}")
        
        return opt_x, opt_y
    
    def _get_boundary_from_terminals(self) -> Tuple[float, float]:
        """从终端节点计算边界"""
        terminal_mask = self.is_terminal == 1
        terminal_x = jnp.where(terminal_mask, self.fixed_x, 0)
        terminal_y = jnp.where(terminal_mask, self.fixed_y, 0)
        terminal_w = jnp.where(terminal_mask, self.bench.widths, 0)
        terminal_h = jnp.where(terminal_mask, self.bench.heights, 0)
        
        max_x = float(jnp.max(terminal_x + terminal_w))
        max_y = float(jnp.max(terminal_y + terminal_h))
        
        return max_x, max_y
    
    def optimize(self, x: jnp.ndarray, y: jnp.ndarray,
                 widths: jnp.ndarray, heights: jnp.ndarray,
                 pins_dx: jnp.ndarray, pins_dy: jnp.ndarray,
                 boundary_width: Optional[float] = None,
                 boundary_height: Optional[float] = None,
                 max_iterations: int = 10,
                 search_step: float = None,
                 num_search_points: int = 10) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
        """执行后处理优化
        
        Args:
            x, y: 所有模块的初始坐标（包括固定终端）
            widths, heights: 所有模块的尺寸
            pins_dx, pins_dy: 引脚偏移
            boundary_width, boundary_height: 边界尺寸，None则自动计算
            max_iterations: 最大迭代次数
            search_step: 搜索步长，None则自动计算
            num_search_points: 每个方向的搜索点数
            
        Returns:
            (optimized_x, optimized_y, final_hpwl): 优化后的坐标和HPWL
        """
        # 自动计算边界
        if boundary_width is None or boundary_height is None:
            boundary_width, boundary_height = self._get_boundary_from_terminals()
            print(f"自动计算边界: {boundary_width:.2f} x {boundary_height:.2f}")
        
        # 自动计算搜索步长
        if search_step is None:
            avg_module_size = float(jnp.mean(widths[self.movable_indices] + heights[self.movable_indices]) / 2)
            search_step = avg_module_size * 0.5
            print(f"自动计算搜索步长: {search_step:.2f}")
        
        # 转换为可修改的数组
        opt_x = jnp.array(x, dtype=jnp.float32)
        opt_y = jnp.array(y, dtype=jnp.float32)
        
        initial_hpwl = float(self._compute_hpwl_direct(
            opt_x, opt_y, widths, heights, 
            self.nets_ptr, self.pins_nodes, pins_dx, pins_dy
        ))
        initial_overlap = self._get_total_overlap(opt_x, opt_y, widths, heights)
        print(f"初始HPWL: {initial_hpwl:.2f}, 初始重叠面积: {initial_overlap:.2f}")
        
        # 迭代优化
        for iteration in range(max_iterations):
            improved = False
            iteration_improvements = 0
            
            # 对每个可移动模块进行优化
            for i in range(self.num_movable):
                module_idx = int(self.movable_indices[i])
                current_x = float(opt_x[module_idx])
                current_y = float(opt_y[module_idx])
                module_w = float(widths[module_idx])
                module_h = float(heights[module_idx])
                
                best_x, best_y = current_x, current_y
                best_hpwl = float(self._compute_hpwl_direct(
                    opt_x, opt_y, widths, heights,
                    self.nets_ptr, self.pins_nodes, pins_dx, pins_dy
                ))
                
                # 网格搜索最优位置
                for dx_mult in range(-num_search_points, num_search_points + 1):
                    for dy_mult in range(-num_search_points, num_search_points + 1):
                        if dx_mult == 0 and dy_mult == 0:
                            continue
                        
                        new_x = current_x + dx_mult * search_step
                        new_y = current_y + dy_mult * search_step
                        
                        # 检查边界约束
                        if not self._check_boundary(new_x, new_y, module_w, module_h,
                                                   boundary_width, boundary_height):
                            continue
                        
                        # 临时更新坐标
                        temp_x = opt_x.at[module_idx].set(new_x)
                        temp_y = opt_y.at[module_idx].set(new_y)
                        
                        # 检查重叠约束
                        if self._check_overlap(i, new_x, new_y, module_w, module_h,
                                              temp_x, temp_y, widths, heights):
                            continue
                        
                        # 计算新HPWL
                        new_hpwl = float(self._compute_hpwl_direct(
                            temp_x, temp_y, widths, heights,
                            self.nets_ptr, self.pins_nodes, pins_dx, pins_dy
                        ))
                        
                        if new_hpwl < best_hpwl:
                            best_x, best_y = new_x, new_y
                            best_hpwl = new_hpwl
                            improved = True
                
                # 应用最佳位置
                if best_x != current_x or best_y != current_y:
                    opt_x = opt_x.at[module_idx].set(best_x)
                    opt_y = opt_y.at[module_idx].set(best_y)
                    iteration_improvements += 1
            
            current_hpwl = float(self._compute_hpwl_direct(
                opt_x, opt_y, widths, heights,
                self.nets_ptr, self.pins_nodes, pins_dx, pins_dy
            ))
            print(f"  迭代 {iteration + 1}: HPWL = {current_hpwl:.2f}, 改进模块数 = {iteration_improvements}")
            
            # 如果没有改进，停止
            if not improved:
                print(f"  迭代 {iteration + 1} 无改进，停止优化")
                break
        
        final_hpwl = float(self._compute_hpwl_direct(
            opt_x, opt_y, widths, heights,
            self.nets_ptr, self.pins_nodes, pins_dx, pins_dy
        ))
        
        # 验证最终是否有重叠
        final_overlap = self._get_total_overlap(opt_x, opt_y, widths, heights)
        if final_overlap > 0:
            print(f"警告: 最终仍有重叠! 重叠面积 = {final_overlap:.2f}")
        
        improvement = (initial_hpwl - final_hpwl) / initial_hpwl * 100
        print(f"优化完成: {initial_hpwl:.2f} -> {final_hpwl:.2f} (改进 {improvement:.2f}%)")
        
        return opt_x, opt_y, final_hpwl
    
    def optimize_with_annealing(self, x: jnp.ndarray, y: jnp.ndarray,
                                widths: jnp.ndarray, heights: jnp.ndarray,
                                pins_dx: jnp.ndarray, pins_dy: jnp.ndarray,
                                boundary_width: Optional[float] = None,
                                boundary_height: Optional[float] = None,
                                max_iterations: int = 5,
                                initial_step: float = 10,
                                final_step: float = 1,
                                initial_search_points: int = 20,
                                final_search_points: int = 5) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
        """使用退火策略的后处理优化
        
        Args:
            initial_step: 初始搜索步长
            final_step: 最终搜索步长
            initial_search_points: 初始搜索点数
            final_search_points: 最终搜索点数
            
        Returns:
            (optimized_x, optimized_y, final_hpwl)
        """
        print("\n" + "="*50)
        print("后处理优化（退火策略）")
        print("="*50)
        
        # 自动计算边界
        if boundary_width is None or boundary_height is None:
            boundary_width, boundary_height = self._get_boundary_from_terminals()
            print(f"边界: {boundary_width:.2f} x {boundary_height:.2f}")
        
        opt_x, opt_y = x, y
        
        # 第0步：分离重叠的模块
        print("\n步骤 0: 分离重叠模块")
        opt_x, opt_y = self.separate_overlaps(
            opt_x, opt_y, widths, heights, boundary_width, boundary_height
        )
        
        # 退火优化阶段
        for phase in range(max_iterations):
            t = phase / max(1, max_iterations - 1)
            current_search_points = int(initial_search_points * (1 - t) + final_search_points * t)
            current_step = initial_step * (1 - t) + final_step * t
            
            print(f"\n阶段 {phase + 1}/{max_iterations}: 搜索点数={current_search_points}, 步长={current_step:.2f}")
            
            opt_x, opt_y, hpwl = self.optimize(
                opt_x, opt_y, widths, heights, pins_dx, pins_dy,
                boundary_width, boundary_height,
                max_iterations=3,
                search_step=current_step,
                num_search_points=current_search_points
            )
        
        return opt_x, opt_y, hpwl
