"""
后处理优化模块 - 对MCTS布局结果进行局部优化（GPU加速版）

合并后的单阶段退火：
- 每个 phase 先做一次整体平移（把整个 movable 集群作为刚体搬到更优位置）
- 然后对每个 movable 做 4 方向 × 本地 offset 网格搜索
- phase 之间 step_size 从 max(bw,bh)/search_points 指数衰减到 1
所有计算均在 JAX/XLA 上批量完成，避免 CPU-GPU 频繁同步。
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import Tuple


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
    def _batch_find_best(opt_x, opt_y, widths, heights,
                         module_idx, candidate_x, candidate_y,
                         module_w, module_h,
                         boundary_w, boundary_h,
                         movable_indices, module_local_idx,
                         nets_ptr, pins_nodes, pins_dx, pins_dy):
        """批量评估所有候选位置，返回 HPWL 最优且合法的位置。

        一次 GPU 调用完成：边界检查 + 重叠检查 + 所有候选 HPWL 计算。
        若所有候选都不合法则 fallback 到当前位置（调用方需自行复核合法性）。
        """
        valid = ((candidate_x >= 0) & (candidate_y >= 0) &
                 (candidate_x + module_w <= boundary_w) &
                 (candidate_y + module_h <= boundary_h))

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

        def single_hpwl(cx_val, cy_val):
            return PostOptimizer._compute_hpwl_direct(
                opt_x.at[module_idx].set(cx_val),
                opt_y.at[module_idx].set(cy_val),
                widths, heights, nets_ptr, pins_nodes, pins_dx, pins_dy)

        all_hpwl = jax.vmap(single_hpwl)(candidate_x, candidate_y)
        all_hpwl = jnp.where(valid, all_hpwl, jnp.inf)

        current_hpwl = single_hpwl(opt_x[module_idx], opt_y[module_idx])
        best_idx = jnp.argmin(all_hpwl)
        best_hpwl = all_hpwl[best_idx]

        improved = best_hpwl < current_hpwl
        final_x = jnp.where(improved, candidate_x[best_idx], opt_x[module_idx])
        final_y = jnp.where(improved, candidate_y[best_idx], opt_y[module_idx])

        return final_x, final_y, improved

    @staticmethod
    @jax.jit
    def _try_global_translation(opt_x, opt_y, widths, heights,
                                 movable_indices, offsets_x, offsets_y,
                                 bw, bh, nets_ptr, pins_nodes, pins_dx, pins_dy):
        """整体平移所有 movable 模块，保留 HPWL 最低的偏移。

        由于所有 movable 同步平移：
          - 相对几何不变，不会产生新的 movable-movable 重叠
          - 只需检查整体包围盒平移后是否仍在 interposer 内
        """
        mi = movable_indices

        mx_min = jnp.min(opt_x[mi])
        my_min = jnp.min(opt_y[mi])
        mx_max = jnp.max(opt_x[mi] + widths[mi])
        my_max = jnp.max(opt_y[mi] + heights[mi])

        cur_hpwl = PostOptimizer._compute_hpwl_direct(
            opt_x, opt_y, widths, heights,
            nets_ptr, pins_nodes, pins_dx, pins_dy)

        def eval_shift(dx, dy):
            in_bounds = ((mx_min + dx >= 0) & (my_min + dy >= 0) &
                         (mx_max + dx <= bw) & (my_max + dy <= bh))
            new_x = opt_x.at[mi].add(dx)
            new_y = opt_y.at[mi].add(dy)
            hpwl = PostOptimizer._compute_hpwl_direct(
                new_x, new_y, widths, heights,
                nets_ptr, pins_nodes, pins_dx, pins_dy)
            return jnp.where(in_bounds, hpwl, jnp.float32(jnp.inf))

        all_hpwl = jax.vmap(eval_shift)(offsets_x, offsets_y)
        best_idx = jnp.argmin(all_hpwl)
        best_hpwl = all_hpwl[best_idx]
        improved = best_hpwl < cur_hpwl
        dx = jnp.where(improved, offsets_x[best_idx], jnp.float32(0.0))
        dy = jnp.where(improved, offsets_y[best_idx], jnp.float32(0.0))
        return opt_x.at[mi].add(dx), opt_y.at[mi].add(dy)

    # ======================== 方向+位置联合 sweep ========================

    @staticmethod
    def _apply_orientation_to_module(widths, heights, pins_dx, pins_dy,
                                      module_idx, ori,
                                      base_widths, base_heights,
                                      base_pins_dx, base_pins_dy,
                                      pins_nodes):
        """对单模块应用方向（N=0, E=1, S=2, W=3），基于原始未旋转值。"""
        should_swap = (ori == 1) | (ori == 3)
        bw_mod = base_widths[module_idx]
        bh_mod = base_heights[module_idx]
        widths = widths.at[module_idx].set(jnp.where(should_swap, bh_mod, bw_mod))
        heights = heights.at[module_idx].set(jnp.where(should_swap, bw_mod, bh_mod))

        is_this = pins_nodes == module_idx
        new_pdx = jnp.where(ori == 1, -base_pins_dy,
                  jnp.where(ori == 2, -base_pins_dx,
                  jnp.where(ori == 3, base_pins_dy, base_pins_dx)))
        new_pdy = jnp.where(ori == 1, base_pins_dx,
                  jnp.where(ori == 2, -base_pins_dy,
                  jnp.where(ori == 3, -base_pins_dx, base_pins_dy)))
        pins_dx = jnp.where(is_this, new_pdx, pins_dx)
        pins_dy = jnp.where(is_this, new_pdy, pins_dy)

        return widths, heights, pins_dx, pins_dy

    @staticmethod
    def _try_orient_move_for_module(opt_x, opt_y, widths, heights, pins_dx, pins_dy,
                                     module_idx, module_local_idx, movable_indices,
                                     offsets_x, offsets_y, bw, bh,
                                     nets_ptr, pins_nodes,
                                     base_widths, base_heights,
                                     base_pins_dx, base_pins_dy):
        """单模块 4 方向 × 本地 offset 网格搜索，选 HPWL 最低（严格改善才接受）。

        ori=当前方向 的分支等价于纯位置 sweep，所以本函数可以同时替代逐模块
        位置优化和方向优化。
        """
        cur_hpwl = PostOptimizer._compute_hpwl_direct(
            opt_x, opt_y, widths, heights,
            nets_ptr, pins_nodes, pins_dx, pins_dy)

        def try_ori(ori, carry):
            best_hpwl, best_ori, best_bx, best_by = carry
            w, h, pdx, pdy = PostOptimizer._apply_orientation_to_module(
                widths, heights, pins_dx, pins_dy,
                module_idx, ori,
                base_widths, base_heights, base_pins_dx, base_pins_dy,
                pins_nodes)
            mw, mh = w[module_idx], h[module_idx]

            cx = opt_x[module_idx] + offsets_x
            cy = opt_y[module_idx] + offsets_y

            bx, by, _ = PostOptimizer._batch_find_best(
                opt_x, opt_y, w, h,
                module_idx, cx, cy, mw, mh, bw, bh,
                movable_indices, module_local_idx,
                nets_ptr, pins_nodes, pdx, pdy)

            tx = opt_x.at[module_idx].set(bx)
            ty = opt_y.at[module_idx].set(by)
            hpwl = PostOptimizer._compute_hpwl_direct(
                tx, ty, w, h, nets_ptr, pins_nodes, pdx, pdy)

            # _batch_find_best 在所有候选均无效时会 fallback 到当前位置；
            # 新方向下 W/H 互换后，fallback 位置可能与其他 movable 重叠或越界。
            # 这里重新校验一次，避免接受看似 HPWL 低但物理非法的解。
            in_bounds = ((bx >= 0) & (by >= 0) &
                         (bx + mw <= bw) & (by + mh <= bh))
            other_x = opt_x[movable_indices]
            other_y = opt_y[movable_indices]
            other_w = w[movable_indices]
            other_h = h[movable_indices]
            excl = jnp.arange(movable_indices.shape[0]) == module_local_idx
            ovx = jnp.maximum(0, jnp.minimum(bx + mw, other_x + other_w)
                                  - jnp.maximum(bx, other_x))
            ovy = jnp.maximum(0, jnp.minimum(by + mh, other_y + other_h)
                                  - jnp.maximum(by, other_y))
            has_ov = jnp.any((ovx * ovy) * ~excl > 0)
            valid = in_bounds & ~has_ov
            hpwl = jnp.where(valid, hpwl, jnp.float32(jnp.inf))

            improved = hpwl < best_hpwl
            return (jnp.where(improved, hpwl, best_hpwl),
                    jnp.where(improved, ori, best_ori),
                    jnp.where(improved, bx, best_bx),
                    jnp.where(improved, by, best_by))

        init = (cur_hpwl, jnp.int32(-1),
                opt_x[module_idx], opt_y[module_idx])
        _, best_ori, best_x, best_y = jax.lax.fori_loop(
            0, 4, try_ori, init)

        has_change = best_ori >= 0
        safe_ori = jnp.maximum(best_ori, 0)
        new_w, new_h, new_pdx, new_pdy = PostOptimizer._apply_orientation_to_module(
            widths, heights, pins_dx, pins_dy,
            module_idx, safe_ori,
            base_widths, base_heights, base_pins_dx, base_pins_dy,
            pins_nodes)
        widths = jnp.where(has_change, new_w, widths)
        heights = jnp.where(has_change, new_h, heights)
        pins_dx = jnp.where(has_change, new_pdx, pins_dx)
        pins_dy = jnp.where(has_change, new_pdy, pins_dy)
        opt_x = opt_x.at[module_idx].set(best_x)
        opt_y = opt_y.at[module_idx].set(best_y)
        return opt_x, opt_y, widths, heights, pins_dx, pins_dy

    @staticmethod
    @jax.jit
    def _sweep_orient_and_move(opt_x, opt_y, widths, heights, pins_dx, pins_dy,
                                movable_indices, offsets_x, offsets_y, bw, bh,
                                nets_ptr, pins_nodes,
                                base_widths, base_heights,
                                base_pins_dx, base_pins_dy):
        """按 movable_indices 的顺序，对每个模块做 4 方向 + 本地位置搜索。"""
        n = movable_indices.shape[0]

        def body(i, carry):
            ox, oy, w, h, pdx, pdy = carry
            idx = movable_indices[i]
            return PostOptimizer._try_orient_move_for_module(
                ox, oy, w, h, pdx, pdy,
                idx, i, movable_indices,
                offsets_x, offsets_y, bw, bh,
                nets_ptr, pins_nodes,
                base_widths, base_heights, base_pins_dx, base_pins_dy)

        return jax.lax.fori_loop(
            0, n, body, (opt_x, opt_y, widths, heights, pins_dx, pins_dy))

    # ======================== 单 phase 与整轮退火 ========================

    @staticmethod
    @jax.jit
    def _phase_step_merged(opt_x, opt_y, widths, heights, pins_dx, pins_dy,
                           movable_indices, offsets_x, offsets_y, bw, bh,
                           nets_ptr, pins_nodes,
                           base_widths, base_heights,
                           base_pins_dx, base_pins_dy):
        """单个 phase：整体平移 -> 方向·位置联合 sweep。

        同时返回平移后、sweep 后两个中间态，供动画逐帧展示。
        """
        tx, ty = PostOptimizer._try_global_translation(
            opt_x, opt_y, widths, heights, movable_indices,
            offsets_x, offsets_y, bw, bh,
            nets_ptr, pins_nodes, pins_dx, pins_dy)

        nx, ny, nw, nh, npdx, npdy = PostOptimizer._sweep_orient_and_move(
            tx, ty, widths, heights, pins_dx, pins_dy,
            movable_indices, offsets_x, offsets_y, bw, bh,
            nets_ptr, pins_nodes,
            base_widths, base_heights, base_pins_dx, base_pins_dy)
        return tx, ty, nx, ny, nw, nh, npdx, npdy

    @staticmethod
    @jax.jit
    def _full_annealing_merged(opt_x, opt_y, widths, heights, pins_dx, pins_dy,
                                movable_indices, bw, bh,
                                nets_ptr, pins_nodes,
                                num_phases, initial_step, final_step,
                                base_offsets_x, base_offsets_y,
                                base_widths, base_heights,
                                base_pins_dx, base_pins_dy):
        """整轮合并退火：每个 phase 执行 [整体平移 + 方向·位置联合 sweep]。

        - 第 1 个 phase 的 offset 网格几乎覆盖整张 interposer，起到全图搜索作用；
        - 后续 phase 随 cur_step 指数收缩，自动过渡到精细调整；
        - ori=当前方向 的分支等价于纯位置 sweep，所以无需再单独 sweep。
        """
        def phase_body(phase, carry):
            ox, oy, w, h, pdx, pdy = carry
            t = phase / jnp.maximum(1, num_phases - 1)
            ratio = final_step / jnp.maximum(initial_step, final_step)
            cur_step = initial_step * ratio ** t
            offsets_x = base_offsets_x * cur_step
            offsets_y = base_offsets_y * cur_step

            _, _, ox, oy, w, h, pdx, pdy = PostOptimizer._phase_step_merged(
                ox, oy, w, h, pdx, pdy,
                movable_indices, offsets_x, offsets_y, bw, bh,
                nets_ptr, pins_nodes,
                base_widths, base_heights, base_pins_dx, base_pins_dy)
            return ox, oy, w, h, pdx, pdy

        return jax.lax.fori_loop(
            0, num_phases, phase_body,
            (opt_x, opt_y, widths, heights, pins_dx, pins_dy))

    @staticmethod
    @jax.jit
    def _vmap_annealing_merged(batch_x, batch_y, batch_w, batch_h, batch_pdx, batch_pdy,
                                movable_indices, bw, bh, nets_ptr, pins_nodes,
                                num_phases, initial_step, final_step,
                                base_offsets_x, base_offsets_y,
                                base_widths, base_heights,
                                base_pins_dx, base_pins_dy):
        """vmap 并行合并版退火：一次 GPU 调用处理整个 chunk。"""
        def single(args):
            x, y, w, h, pdx, pdy = args
            ox, oy, nw, nh, npdx, npdy = PostOptimizer._full_annealing_merged(
                x, y, w, h, pdx, pdy, movable_indices, bw, bh,
                nets_ptr, pins_nodes,
                num_phases, initial_step, final_step,
                base_offsets_x, base_offsets_y,
                base_widths, base_heights, base_pins_dx, base_pins_dy)
            hpwl = PostOptimizer._compute_hpwl_direct(
                ox, oy, nw, nh, nets_ptr, pins_nodes, npdx, npdy)
            return ox, oy, nw, nh, npdx, npdy, hpwl
        return jax.vmap(single)((batch_x, batch_y, batch_w, batch_h,
                                 batch_pdx, batch_pdy))

    # ======================== 公开方法 ========================

    def _get_boundary_from_terminals(self) -> Tuple[float, float]:
        """从终端节点计算 interposer 边界。"""
        terminal_mask = self.is_terminal == 1
        terminal_x = jnp.where(terminal_mask, self.fixed_x, 0)
        terminal_y = jnp.where(terminal_mask, self.fixed_y, 0)
        terminal_w = jnp.where(terminal_mask, self.bench.widths, 0)
        terminal_h = jnp.where(terminal_mask, self.bench.heights, 0)
        return float(jnp.max(terminal_x + terminal_w)), float(jnp.max(terminal_y + terminal_h))

    def _make_offsets(self, search_points):
        sp = search_points
        offsets_1d = jnp.arange(-sp, sp + 1, dtype=jnp.float32)
        gx, gy = jnp.meshgrid(offsets_1d, offsets_1d)
        gx, gy = gx.ravel(), gy.ravel()
        nonzero = (gx != 0) | (gy != 0)
        return jnp.where(nonzero, gx, 0.0), jnp.where(nonzero, gy, 0.0)

    def build_orderings(self, num_random_orderings: int = 4, seed: int = 0):
        """构造用于 optimize_batch 的模块遍历顺序集合（全随机）。

        动画重放 Stage 2 时可以直接根据 ordering 索引复用同一份 ordering。
        """
        orderings = []
        rng = jax.random.PRNGKey(seed)
        for _ in range(num_random_orderings):
            rng, subkey = jax.random.split(rng)
            orderings.append(jax.random.permutation(subkey, self.movable_indices))
        return orderings

    def ordering_labels(self, num_random_orderings: int = 4):
        return [f"随机{i+1}" for i in range(num_random_orderings)]

    def optimize_batch(self, all_x, all_y, all_w, all_h, all_pdx, all_pdy,
                       boundary_width=None, boundary_height=None,
                       max_iterations=5,
                       search_points=20, chunk_size=16,
                       num_random_orderings=4, seed=0):
        """批量后处理（合并版：整体平移 + 方向·位置联合 sweep）。

        K 个候选方案 × 多种 ordering，每个 phase 同时做整体平移、方向翻转和
        本地位置搜索，逐候选取最优。

        Returns:
          best_x/y/w/h/pdx/pdy: 最优几何
          best_hpwl: 对应 HPWL
          best_ord_idx: (K,) 每个候选达到最优时使用的 ordering 索引
          orderings: 本次使用的 ordering 列表（顺序与 index 一致）
        """
        if boundary_width is None or boundary_height is None:
            boundary_width, boundary_height = self._get_boundary_from_terminals()

        bw, bh = jnp.float32(boundary_width), jnp.float32(boundary_height)
        initial_step = jnp.float32(max(boundary_width, boundary_height) // search_points)
        base_offsets_x, base_offsets_y = self._make_offsets(search_points)

        base_w = jnp.array(self.bench.widths)
        base_h = jnp.array(self.bench.heights)
        base_pdx = jnp.array(self.bench.pins_dx)
        base_pdy = jnp.array(self.bench.pins_dy)

        orderings = self.build_orderings(num_random_orderings, seed)
        labels = self.ordering_labels(num_random_orderings)

        K = all_x.shape[0]
        pad = (-K) % chunk_size
        if pad > 0:
            def p(a):
                return jnp.concatenate([a, jnp.zeros((pad,) + a.shape[1:], dtype=a.dtype)])
            all_x, all_y = p(all_x), p(all_y)
            all_w, all_h = p(all_w), p(all_h)
            all_pdx, all_pdy = p(all_pdx), p(all_pdy)

        total = all_x.shape[0]
        best_x = jnp.zeros_like(all_x)
        best_y = jnp.zeros_like(all_y)
        best_w = jnp.zeros_like(all_w)
        best_h_geom = jnp.zeros_like(all_h)
        best_pdx = jnp.zeros_like(all_pdx)
        best_pdy = jnp.zeros_like(all_pdy)
        best_hpwl = jnp.full(total, jnp.inf)
        best_ord = jnp.full(total, -1, dtype=jnp.int32)

        for oi, mi in enumerate(orderings):
            shared = (mi, bw, bh,
                      self.nets_ptr, self.pins_nodes,
                      jnp.int32(max_iterations),
                      initial_step, jnp.float32(1),
                      base_offsets_x, base_offsets_y,
                      base_w, base_h, base_pdx, base_pdy)

            res_x, res_y, res_w, res_h, res_pdx, res_pdy, res_hpwl = [], [], [], [], [], [], []
            for start in range(0, total, chunk_size):
                end = start + chunk_size
                ox, oy, ow, oh, opdx, opdy, hpwl = self._vmap_annealing_merged(
                    all_x[start:end], all_y[start:end],
                    all_w[start:end], all_h[start:end],
                    all_pdx[start:end], all_pdy[start:end], *shared)
                res_x.append(ox)
                res_y.append(oy)
                res_w.append(ow)
                res_h.append(oh)
                res_pdx.append(opdx)
                res_pdy.append(opdy)
                res_hpwl.append(hpwl)

            cur_x = jnp.concatenate(res_x)
            cur_y = jnp.concatenate(res_y)
            cur_w = jnp.concatenate(res_w)
            cur_h = jnp.concatenate(res_h)
            cur_pdx = jnp.concatenate(res_pdx)
            cur_pdy = jnp.concatenate(res_pdy)
            cur_hpwl = jnp.concatenate(res_hpwl)

            improved = cur_hpwl < best_hpwl
            best_x = jnp.where(improved[:, None], cur_x, best_x)
            best_y = jnp.where(improved[:, None], cur_y, best_y)
            best_w = jnp.where(improved[:, None], cur_w, best_w)
            best_h_geom = jnp.where(improved[:, None], cur_h, best_h_geom)
            best_pdx = jnp.where(improved[:, None], cur_pdx, best_pdx)
            best_pdy = jnp.where(improved[:, None], cur_pdy, best_pdy)
            best_ord = jnp.where(improved, jnp.int32(oi), best_ord)
            best_hpwl = jnp.minimum(best_hpwl, cur_hpwl)
            cur_best = float(jnp.min(best_hpwl[:K]))
            print(f"    策略 {oi+1}/{len(orderings)} [{labels[oi]}] 完成, 当前全局最优={cur_best:.0f}")

        return (best_x[:K], best_y[:K],
                best_w[:K], best_h_geom[:K],
                best_pdx[:K], best_pdy[:K],
                best_hpwl[:K], best_ord[:K], orderings)
