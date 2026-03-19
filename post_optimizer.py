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
    
    @staticmethod
    @jax.jit
    def _sweep_modules(opt_x, opt_y, widths, heights,
                       movable_indices, offsets_x, offsets_y,
                       bw, bh, nets_ptr, pins_nodes, pins_dx, pins_dy):
        """一轮完整的模块扫描优化（全部在GPU上，无Python循环开销）
        
        用 lax.fori_loop 替代 Python for 循环，将整个模块遍历编译为
        单个 XLA 计算图，消除逐模块的 kernel launch 和 CPU-GPU 同步。
        """
        n = movable_indices.shape[0]
        
        def body(i, carry):
            ox, oy = carry
            idx = movable_indices[i]
            bx, by, _ = PostOptimizer._batch_find_best(
                ox, oy, widths, heights,
                idx, ox[idx] + offsets_x, oy[idx] + offsets_y,
                widths[idx], heights[idx], bw, bh,
                movable_indices, i,
                nets_ptr, pins_nodes, pins_dx, pins_dy
            )
            ox = ox.at[idx].set(bx)
            oy = oy.at[idx].set(by)
            return ox, oy
        
        return jax.lax.fori_loop(0, n, body, (opt_x, opt_y))
    
    # ======================== 公开方法 ========================
    
    def _get_boundary_from_terminals(self) -> Tuple[float, float]:
        """从终端节点计算边界"""
        terminal_mask = self.is_terminal == 1
        terminal_x = jnp.where(terminal_mask, self.fixed_x, 0)
        terminal_y = jnp.where(terminal_mask, self.fixed_y, 0)
        terminal_w = jnp.where(terminal_mask, self.bench.widths, 0)
        terminal_h = jnp.where(terminal_mask, self.bench.heights, 0)
        return float(jnp.max(terminal_x + terminal_w)), float(jnp.max(terminal_y + terminal_h))
    
    @staticmethod
    @jax.jit
    def _optimize_iterations(opt_x, opt_y, widths, heights,
                             movable_indices, offsets_x, offsets_y,
                             bw, bh, nets_ptr, pins_nodes, pins_dx, pins_dy,
                             max_iterations):
        """多轮 sweep 迭代，带早停（全部在 GPU 上，零 CPU 同步）"""
        def compute_hpwl(ox, oy):
            centers_x = ox + 0.5 * widths
            centers_y = oy + 0.5 * heights
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
                jax.ops.segment_min(pin_y, seg_ids, num_segments=num_nets))
        
        init_hpwl = compute_hpwl(opt_x, opt_y)
        
        def cond(carry):
            ox, oy, prev_hpwl, i, cont = carry
            return cont & (i < max_iterations)
        
        def body(carry):
            ox, oy, prev_hpwl, i, _ = carry
            ox, oy = PostOptimizer._sweep_modules(
                ox, oy, widths, heights, movable_indices, offsets_x, offsets_y,
                bw, bh, nets_ptr, pins_nodes, pins_dx, pins_dy)
            cur_hpwl = compute_hpwl(ox, oy)
            improved = cur_hpwl < prev_hpwl - 1e-6
            return ox, oy, cur_hpwl, i + 1, improved
        
        ox, oy, final_hpwl, _, _ = jax.lax.while_loop(
            cond, body, (opt_x, opt_y, init_hpwl, jnp.int32(0), jnp.bool_(True)))
        return ox, oy, final_hpwl
    
    @staticmethod
    @jax.jit
    def _full_annealing(opt_x, opt_y, widths, heights,
                        movable_indices, bw, bh,
                        nets_ptr, pins_nodes, pins_dx, pins_dy,
                        num_phases, iters_per_phase, initial_step, final_step,
                        base_offsets_x, base_offsets_y):
        """完整的退火优化流程（全部在 GPU 上，单个 kernel）
        
        将退火阶段循环 + 每阶段的迭代优化全部编译进 XLA，
        整个过程零 CPU-GPU 同步。
        """
        def phase_body(phase, carry):
            ox, oy = carry
            t = phase / jnp.maximum(1, num_phases - 1)
            cur_step = initial_step * (1 - t) + final_step * t
            offsets_x = base_offsets_x * cur_step
            offsets_y = base_offsets_y * cur_step
            ox, oy, _ = PostOptimizer._optimize_iterations(
                ox, oy, widths, heights, movable_indices, offsets_x, offsets_y,
                bw, bh, nets_ptr, pins_nodes, pins_dx, pins_dy, iters_per_phase)
            return ox, oy
        
        opt_x, opt_y = jax.lax.fori_loop(0, num_phases, phase_body, (opt_x, opt_y))
        return opt_x, opt_y
    
    def optimize_with_annealing(self, x, y, widths, heights, pins_dx, pins_dy,
                                boundary_width=None, boundary_height=None,
                                max_iterations=5,
                                initial_step=10, final_step=1,
                                search_points=20):
        """退火策略后处理优化（入口函数）"""
        if boundary_width is None or boundary_height is None:
            boundary_width, boundary_height = self._get_boundary_from_terminals()
        
        opt_x = jnp.array(x, dtype=jnp.float32)
        opt_y = jnp.array(y, dtype=jnp.float32)
        bw, bh = jnp.float32(boundary_width), jnp.float32(boundary_height)
        
        sp = search_points
        offsets_1d = jnp.arange(-sp, sp + 1, dtype=jnp.float32)
        gx, gy = jnp.meshgrid(offsets_1d, offsets_1d)
        gx, gy = gx.ravel(), gy.ravel()
        nonzero = (gx != 0) | (gy != 0)
        base_offsets_x = jnp.where(nonzero, gx, 0.0)
        base_offsets_y = jnp.where(nonzero, gy, 0.0)
        
        opt_x, opt_y = self._full_annealing(
            opt_x, opt_y,
            jnp.array(widths), jnp.array(heights),
            self.movable_indices, bw, bh,
            self.nets_ptr, self.pins_nodes, pins_dx, pins_dy,
            jnp.int32(max_iterations), jnp.int32(3),
            jnp.float32(initial_step), jnp.float32(final_step),
            base_offsets_x, base_offsets_y)
        
        hpwl = float(self._compute_hpwl_direct(
            opt_x, opt_y, jnp.array(widths), jnp.array(heights),
            self.nets_ptr, self.pins_nodes, pins_dx, pins_dy))
        
        return opt_x, opt_y, hpwl
