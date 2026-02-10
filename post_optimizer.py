"""
后处理优化模块 - 对MCTS布局结果进行局部优化（GPU加速版）

在得到初始布局后，逐个调整每个模块的位置，
使其在不超边界、不重叠的约束下最小化总wirelength。
所有计算均在GPU上批量完成，避免CPU-GPU频繁同步。
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import Tuple, Optional


class PostOptimizer:
    """后处理优化器（GPU加速）"""
    
    def __init__(self, bench, movable_indices: jnp.ndarray):
        self.bench = bench
        self.movable_indices = jnp.array(movable_indices)
        self.num_movable = len(movable_indices)
        
        self.nets_ptr = bench.nets_ptr
        self.pins_nodes = bench.pins_nodes
        self.pins_dx = bench.pins_dx
        self.pins_dy = bench.pins_dy
        
        self.fixed_x = jnp.array(bench.x_fixed)
        self.fixed_y = jnp.array(bench.y_fixed)
        self.is_terminal = jnp.array(bench.is_terminal)
    
    # ======================== 静态JIT核心计算 ========================
    
    @staticmethod
    @jax.jit
    def _compute_hpwl_direct(x, y, widths, heights,
                             nets_ptr, pins_nodes, pins_dx, pins_dy):
        """直接从坐标计算HPWL"""
        centers_x = x + 0.5 * widths
        centers_y = y + 0.5 * heights
        pw = widths[pins_nodes]
        ph = heights[pins_nodes]
        pin_x = centers_x[pins_nodes] + (pins_dx / 100.0) * pw
        pin_y = centers_y[pins_nodes] + (pins_dy / 100.0) * ph
        
        num_nets = nets_ptr.shape[0] - 1
        counts = nets_ptr[1:] - nets_ptr[:-1]
        seg_ids = jnp.repeat(jnp.arange(num_nets, dtype=jnp.int32), counts,
                             total_repeat_length=pins_nodes.shape[0])
        
        maxx = jax.ops.segment_max(pin_x, seg_ids, num_segments=num_nets)
        minx = jax.ops.segment_min(pin_x, seg_ids, num_segments=num_nets)
        maxy = jax.ops.segment_max(pin_y, seg_ids, num_segments=num_nets)
        miny = jax.ops.segment_min(pin_y, seg_ids, num_segments=num_nets)
        return jnp.sum((maxx - minx) + (maxy - miny))
    
    @staticmethod
    @jax.jit
    def _compute_total_overlap(x, y, w, h, movable_indices):
        """GPU加速的总重叠面积计算"""
        mx, my = x[movable_indices], y[movable_indices]
        mw, mh = w[movable_indices], h[movable_indices]
        
        ov_x = jnp.maximum(0, jnp.minimum(mx[:, None] + mw[:, None],
                                           mx[None, :] + mw[None, :]) -
                              jnp.maximum(mx[:, None], mx[None, :]))
        ov_y = jnp.maximum(0, jnp.minimum(my[:, None] + mh[:, None],
                                           my[None, :] + mh[None, :]) -
                              jnp.maximum(my[:, None], my[None, :]))
        
        n = movable_indices.shape[0]
        mask = jnp.triu(jnp.ones((n, n), dtype=bool), k=1)
        return jnp.sum(ov_x * ov_y * mask)
    
    @staticmethod
    @jax.jit
    def _separation_step(opt_x, opt_y, widths, heights,
                         movable_indices, boundary_w, boundary_h):
        """一步分离操作（GPU加速）：计算推力并移动所有模块"""
        mx, my = opt_x[movable_indices], opt_y[movable_indices]
        mw, mh = widths[movable_indices], heights[movable_indices]
        n = movable_indices.shape[0]
        
        # 两两重叠面积 (n, n)
        ov_x = jnp.maximum(0, jnp.minimum(mx[:, None] + mw[:, None],
                                           mx[None, :] + mw[None, :]) -
                              jnp.maximum(mx[:, None], mx[None, :]))
        ov_y = jnp.maximum(0, jnp.minimum(my[:, None] + mh[:, None],
                                           my[None, :] + mh[None, :]) -
                              jnp.maximum(my[:, None], my[None, :]))
        overlap_area = ov_x * ov_y
        
        # 推力方向（从j指向i）
        cx, cy = mx + mw / 2, my + mh / 2
        dx = cx[:, None] - cx[None, :]
        dy = cy[:, None] - cy[None, :]
        dist = jnp.maximum(1e-6, jnp.sqrt(dx**2 + dy**2))
        
        force_mag = jnp.sqrt(overlap_area) * (~jnp.eye(n, dtype=bool))
        force_x = jnp.sum(dx / dist * force_mag, axis=1)
        force_y = jnp.sum(dy / dist * force_mag, axis=1)
        
        # 应用推力
        fmag = jnp.maximum(1e-6, jnp.sqrt(force_x**2 + force_y**2))
        step_size = jnp.minimum(mw, mh) * 0.3
        should_move = fmag > 1e-6
        
        new_mx = mx + jnp.where(should_move, force_x / fmag * step_size, 0.0)
        new_my = my + jnp.where(should_move, force_y / fmag * step_size, 0.0)
        new_mx = jnp.clip(new_mx, 0, boundary_w - mw)
        new_my = jnp.clip(new_my, 0, boundary_h - mh)
        
        opt_x = opt_x.at[movable_indices].set(new_mx)
        opt_y = opt_y.at[movable_indices].set(new_my)
        
        total_overlap = jnp.sum(overlap_area * jnp.triu(jnp.ones((n, n)), k=1))
        return opt_x, opt_y, jnp.any(should_move), total_overlap
    
    @staticmethod
    @jax.jit
    def _batch_find_best(opt_x, opt_y, widths, heights,
                         module_idx, candidate_x, candidate_y,
                         module_w, module_h,
                         boundary_w, boundary_h,
                         movable_indices, module_local_idx,
                         nets_ptr, pins_nodes, pins_dx, pins_dy):
        """批量评估所有候选位置，找到最佳位置（GPU加速核心）
        
        一次GPU调用完成：边界检查 + 重叠检查 + 所有候选HPWL计算
        """
        # 1. 边界检查 (C,)
        valid = ((candidate_x >= 0) & (candidate_y >= 0) &
                 (candidate_x + module_w <= boundary_w) &
                 (candidate_y + module_h <= boundary_h))
        
        # 2. 重叠检查 (C, M) -> (C,)
        other_x = opt_x[movable_indices]
        other_y = opt_y[movable_indices]
        other_w = widths[movable_indices]
        other_h = heights[movable_indices]
        exclude = jnp.arange(movable_indices.shape[0]) == module_local_idx
        
        cx, cy = candidate_x[:, None], candidate_y[:, None]
        ox, oy = other_x[None, :], other_y[None, :]
        ow, oh = other_w[None, :], other_h[None, :]
        
        ov_x = jnp.maximum(0, jnp.minimum(cx + module_w, ox + ow) - jnp.maximum(cx, ox))
        ov_y = jnp.maximum(0, jnp.minimum(cy + module_h, oy + oh) - jnp.maximum(cy, oy))
        has_overlap = jnp.any((ov_x * ov_y) * ~exclude[None, :] > 0, axis=1)
        
        valid = valid & ~has_overlap
        
        # 3. 批量计算HPWL (vmap: 一次GPU调用算完所有候选)
        def single_hpwl(cx_val, cy_val):
            tx = opt_x.at[module_idx].set(cx_val)
            ty = opt_y.at[module_idx].set(cy_val)
            centers_x = tx + 0.5 * widths
            centers_y = ty + 0.5 * heights
            pw = widths[pins_nodes]
            ph = heights[pins_nodes]
            pin_x = centers_x[pins_nodes] + (pins_dx / 100.0) * pw
            pin_y = centers_y[pins_nodes] + (pins_dy / 100.0) * ph
            num_nets = nets_ptr.shape[0] - 1
            counts = nets_ptr[1:] - nets_ptr[:-1]
            seg_ids = jnp.repeat(jnp.arange(num_nets, dtype=jnp.int32), counts,
                                 total_repeat_length=pins_nodes.shape[0])
            return jnp.sum(
                jax.ops.segment_max(pin_x, seg_ids, num_segments=num_nets) -
                jax.ops.segment_min(pin_x, seg_ids, num_segments=num_nets) +
                jax.ops.segment_max(pin_y, seg_ids, num_segments=num_nets) -
                jax.ops.segment_min(pin_y, seg_ids, num_segments=num_nets)
            )
        
        all_hpwl = jax.vmap(single_hpwl)(candidate_x, candidate_y)
        all_hpwl = jnp.where(valid, all_hpwl, jnp.inf)
        
        # 与当前位置比较
        current_hpwl = single_hpwl(opt_x[module_idx], opt_y[module_idx])
        best_idx = jnp.argmin(all_hpwl)
        best_hpwl = all_hpwl[best_idx]
        
        improved = best_hpwl < current_hpwl
        final_x = jnp.where(improved, candidate_x[best_idx], opt_x[module_idx])
        final_y = jnp.where(improved, candidate_y[best_idx], opt_y[module_idx])
        
        return final_x, final_y, improved
    
    # ======================== 公开方法 ========================
    
    def _get_boundary_from_terminals(self) -> Tuple[float, float]:
        """从终端节点计算边界"""
        terminal_mask = self.is_terminal == 1
        terminal_x = jnp.where(terminal_mask, self.fixed_x, 0)
        terminal_y = jnp.where(terminal_mask, self.fixed_y, 0)
        terminal_w = jnp.where(terminal_mask, self.bench.widths, 0)
        terminal_h = jnp.where(terminal_mask, self.bench.heights, 0)
        return float(jnp.max(terminal_x + terminal_w)), float(jnp.max(terminal_y + terminal_h))
    
    def separate_overlaps(self, x, y, widths, heights,
                          boundary_width, boundary_height,
                          max_iterations=50):
        """分离重叠模块（GPU加速力导向法）"""
        opt_x = jnp.array(x, dtype=jnp.float32)
        opt_y = jnp.array(y, dtype=jnp.float32)
        
        initial_overlap = float(self._compute_total_overlap(
            opt_x, opt_y, widths, heights, self.movable_indices))
        if initial_overlap == 0:
            print("  无重叠，跳过分离步骤")
            return opt_x, opt_y
        
        print(f"  初始重叠面积: {initial_overlap:.2f}")
        bw, bh = jnp.float32(boundary_width), jnp.float32(boundary_height)
        
        for iteration in range(max_iterations):
            opt_x, opt_y, moved, overlap = self._separation_step(
                opt_x, opt_y, widths, heights, self.movable_indices, bw, bh
            )
            current_overlap = float(overlap)
            
            if iteration % 10 == 0:
                print(f"    分离迭代 {iteration}: 重叠面积 = {current_overlap:.2f}")
            if current_overlap == 0:
                print(f"  分离完成！迭代次数: {iteration + 1}")
                break
            if not bool(moved):
                break
        
        final_overlap = float(self._compute_total_overlap(
            opt_x, opt_y, widths, heights, self.movable_indices))
        print(f"  分离后重叠面积: {final_overlap:.2f}")
        return opt_x, opt_y
    
    def optimize(self, x, y, widths, heights, pins_dx, pins_dy,
                 boundary_width=None, boundary_height=None,
                 max_iterations=10, search_step=None, num_search_points=10):
        """GPU加速的后处理优化
        
        对每个模块：一次GPU调用批量评估所有候选位置的边界、重叠和HPWL。
        原来每模块需 ~1680 次GPU调用，现在只需 1 次。
        """
        if boundary_width is None or boundary_height is None:
            boundary_width, boundary_height = self._get_boundary_from_terminals()
        
        if search_step is None:
            avg_size = float(jnp.mean(
                widths[self.movable_indices] + heights[self.movable_indices]) / 2)
            search_step = avg_size * 0.5
        
        opt_x = jnp.array(x, dtype=jnp.float32)
        opt_y = jnp.array(y, dtype=jnp.float32)
        
        initial_hpwl = float(self._compute_hpwl_direct(
            opt_x, opt_y, widths, heights,
            self.nets_ptr, self.pins_nodes, pins_dx, pins_dy
        ))
        initial_overlap = float(self._compute_total_overlap(
            opt_x, opt_y, widths, heights, self.movable_indices))
        print(f"初始HPWL: {initial_hpwl:.2f}, 初始重叠面积: {initial_overlap:.2f}")
        
        # 预计算候选偏移（Python端，只算一次）
        offsets = [(dx * search_step, dy * search_step)
                   for dx in range(-num_search_points, num_search_points + 1)
                   for dy in range(-num_search_points, num_search_points + 1)
                   if dx != 0 or dy != 0]
        offsets_x = jnp.array([o[0] for o in offsets], dtype=jnp.float32)
        offsets_y = jnp.array([o[1] for o in offsets], dtype=jnp.float32)
        
        bw, bh = jnp.float32(boundary_width), jnp.float32(boundary_height)
        prev_hpwl = initial_hpwl
        
        for iteration in range(max_iterations):
            for i in range(self.num_movable):
                idx = self.movable_indices[i]
                # 全部在GPU上：索引、加偏移、批量评估、更新
                best_x, best_y, _ = self._batch_find_best(
                    opt_x, opt_y, widths, heights,
                    idx, opt_x[idx] + offsets_x, opt_y[idx] + offsets_y,
                    widths[idx], heights[idx], bw, bh,
                    self.movable_indices, jnp.int32(i),
                    self.nets_ptr, self.pins_nodes, pins_dx, pins_dy
                )
                opt_x = opt_x.at[idx].set(best_x)
                opt_y = opt_y.at[idx].set(best_y)
            
            # 每轮迭代只同步一次（打印HPWL）
            current_hpwl = float(self._compute_hpwl_direct(
                opt_x, opt_y, widths, heights,
                self.nets_ptr, self.pins_nodes, pins_dx, pins_dy
            ))
            print(f"  迭代 {iteration + 1}: HPWL = {current_hpwl:.2f}")
            
            if current_hpwl >= prev_hpwl - 1e-6:
                print(f"  无改进，停止优化")
                break
            prev_hpwl = current_hpwl
        
        final_hpwl = current_hpwl if max_iterations > 0 else initial_hpwl
        final_overlap = float(self._compute_total_overlap(
            opt_x, opt_y, widths, heights, self.movable_indices))
        if final_overlap > 0:
            print(f"警告: 最终仍有重叠! 重叠面积 = {final_overlap:.2f}")
        
        improvement = (initial_hpwl - final_hpwl) / initial_hpwl * 100
        print(f"优化完成: {initial_hpwl:.2f} -> {final_hpwl:.2f} (改进 {improvement:.2f}%)")
        return opt_x, opt_y, final_hpwl
    
    def optimize_with_annealing(self, x, y, widths, heights, pins_dx, pins_dy,
                                boundary_width=None, boundary_height=None,
                                max_iterations=5,
                                initial_step=10, final_step=1,
                                initial_search_points=20, final_search_points=5):
        """退火策略后处理优化"""
        print("\n" + "="*50)
        print("后处理优化（退火策略）")
        print("="*50)
        
        if boundary_width is None or boundary_height is None:
            boundary_width, boundary_height = self._get_boundary_from_terminals()
        print(f"边界: {boundary_width:.2f} x {boundary_height:.2f}")
        
        opt_x, opt_y = jnp.array(x, dtype=jnp.float32), jnp.array(y, dtype=jnp.float32)
        
        print("\n步骤 0: 分离重叠模块")
        opt_x, opt_y = self.separate_overlaps(
            opt_x, opt_y, widths, heights, boundary_width, boundary_height
        )
        
        hpwl = float(self._compute_hpwl_direct(
            opt_x, opt_y, widths, heights,
            self.nets_ptr, self.pins_nodes, pins_dx, pins_dy
        ))
        
        for phase in range(max_iterations):
            t = phase / max(1, max_iterations - 1)
            cur_points = int(initial_search_points * (1 - t) + final_search_points * t)
            cur_step = initial_step * (1 - t) + final_step * t
            
            print(f"\n阶段 {phase + 1}/{max_iterations}: "
                  f"搜索点数={cur_points}, 步长={cur_step:.2f}")
            
            opt_x, opt_y, hpwl = self.optimize(
                opt_x, opt_y, widths, heights, pins_dx, pins_dy,
                boundary_width, boundary_height,
                max_iterations=3, search_step=cur_step,
                num_search_points=cur_points
            )
        
        return opt_x, opt_y, hpwl
