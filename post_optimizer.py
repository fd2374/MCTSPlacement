"""
后处理优化模块 - 对MCTS布局结果进行局部优化（GPU加速版）

合并后的单阶段退火：
- 每个 phase 先做一次整体平移（把整个 movable 集群作为刚体搬到更优位置）
- 然后对每个 movable 做 4 方向 × 本地 offset 网格搜索
- phase 之间 step_size 从 max(bw,bh)/search_points 指数衰减到 1
所有计算均在 JAX/XLA 上批量完成，避免 CPU-GPU 频繁同步。
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, AxisType
from typing import Tuple

# 与 main.py 保持一致的 device mesh：把候选维度沿 'B' 轴 sharding 到所有可见 GPU。
# Auto axis type + 不 set_mesh，外部数组都按 replicated 处理；shard_map 调用时
# 显式传 mesh=_MESH。
_N_DEV = jax.local_device_count()
_MESH = jax.make_mesh((_N_DEV,), ('B',), axis_types=(AxisType.Auto,))

# ===== 退火 reheat 配置：现在通过 optimize_batch 的 n_runs / reheat_factor 参数传入 =====
# 语义：annealing_phases (total) 被均匀拆成 n_runs 段，每段 floor(total / n_runs) phase。
#   - n_runs=1: 纯 baseline 单段，无 reheat（initial_step → final_step 几何衰减一次）
#   - n_runs>=2: 跑 N 段。第 1 段从 initial_step 出发；后续段从 best snapshot 重启，
#     起始温度 cur_hot = reheat_factor × initial_step。每段都走相同的几何衰减到 final_step。
#     best snapshot 只在改善时前进，保证严格 ≥ baseline。
# improve_eps 仍硬编码：判断"新解严格好于 best"的容差（1.0 = HPWL 单位）。
_ANNEAL_IMPROVE_EPS = 1.0


class PostOptimizer:
    """后处理优化器（GPU加速）"""

    def __init__(self, bench, movable_indices: jnp.ndarray):
        self.bench = bench
        self.movable_indices = jnp.array(movable_indices)
        self.num_movable = len(movable_indices)

        # 缓存只在 shard_map closure 用到的 jit 输入引用
        self.nets_ptr = bench.nets_ptr
        self.pins_nodes = bench.pins_nodes

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
                                total_phases, initial_step, final_step,
                                base_offsets_x, base_offsets_y,
                                base_widths, base_heights,
                                base_pins_dx, base_pins_dy,
                                n_runs, reheat_factor):
        """多段 reheat 退火（总 phase 数固定）。

        语义：
          - total_phases 是所有 reheat 段加起来的总 phase 数
          - 拆分为 n_runs 段，每段 per_run = total_phases // n_runs（floor）
          - 每段从 best snapshot 出发，跑 per_run 个 phase 的 hot→cold 几何衰减：
              第 1 段起始温度 = initial_step
              后续段起始温度 = reheat_factor × initial_step
              终止温度 = final_step
          - best snapshot 只在 HPWL 改善时前进，保证严格 ≥ baseline 单段
          - n_runs=1 时退化为单段 baseline，等价旧 ANNEAL_DISABLE_REHEAT=1 行为
          - 总 GPU 算力代价 ≈ baseline × 1 倍（per_run 总和等于 total_phases）
        """
        per_run = jnp.maximum(jnp.int32(1), total_phases // jnp.int32(n_runs))

        # baseline 节奏：cur_step 从 cur_init 衰减到 final_step，分 per_run 步
        def baseline_body(phase, carry):
            ox, oy, w, h, pdx, pdy, cur_init = carry
            t = phase.astype(jnp.float32) / jnp.maximum(
                jnp.float32(1.0), (per_run - 1).astype(jnp.float32))
            ratio = final_step / jnp.maximum(cur_init, final_step)
            cur_step = cur_init * ratio ** t
            offsets_x = base_offsets_x * cur_step
            offsets_y = base_offsets_y * cur_step
            _, _, ox, oy, w, h, pdx, pdy = PostOptimizer._phase_step_merged(
                ox, oy, w, h, pdx, pdy,
                movable_indices, offsets_x, offsets_y, bw, bh,
                nets_ptr, pins_nodes,
                base_widths, base_heights, base_pins_dx, base_pins_dy)
            return ox, oy, w, h, pdx, pdy, cur_init

        # 初始 best = 输入态
        init_hpwl = PostOptimizer._compute_hpwl_direct(
            opt_x, opt_y, widths, heights,
            nets_ptr, pins_nodes, pins_dx, pins_dy)
        reheat_init = jnp.float32(reheat_factor) * initial_step
        improve_eps = jnp.float32(_ANNEAL_IMPROVE_EPS)

        def run_body(run_idx, run_carry):
            (bx, by, bwg, bhg, bpdx, bpdy, best_h) = run_carry
            # 第 1 段用完整 initial_step，后续段用 reheat_factor × initial_step
            cur_init = jnp.where(run_idx == 0, initial_step, reheat_init)
            ex, ey, ew, eh, epdx, epdy, _ = jax.lax.fori_loop(
                0, per_run, baseline_body,
                (bx, by, bwg, bhg, bpdx, bpdy, cur_init))
            end_h = PostOptimizer._compute_hpwl_direct(
                ex, ey, ew, eh, nets_ptr, pins_nodes, epdx, epdy)
            improved = end_h < (best_h - improve_eps)
            # best snapshot 只前进不后退
            n_bx   = jnp.where(improved, ex,   bx)
            n_by   = jnp.where(improved, ey,   by)
            n_bw   = jnp.where(improved, ew,   bwg)
            n_bh   = jnp.where(improved, eh,   bhg)
            n_bpdx = jnp.where(improved, epdx, bpdx)
            n_bpdy = jnp.where(improved, epdy, bpdy)
            n_best_h = jnp.where(improved, end_h, best_h)
            return (n_bx, n_by, n_bw, n_bh, n_bpdx, n_bpdy, n_best_h)

        init_carry = (opt_x, opt_y, widths, heights, pins_dx, pins_dy,
                      init_hpwl)
        final_carry = jax.lax.fori_loop(
            0, jnp.int32(n_runs), run_body, init_carry)
        return final_carry[:6]

    @staticmethod
    @jax.jit
    def _vmap_annealing_merged(batch_x, batch_y, batch_w, batch_h, batch_pdx, batch_pdy,
                                movable_indices, bw, bh, nets_ptr, pins_nodes,
                                total_phases, initial_step, final_step,
                                base_offsets_x, base_offsets_y,
                                base_widths, base_heights,
                                base_pins_dx, base_pins_dy,
                                n_runs, reheat_factor):
        """vmap 并行合并版退火：一次 GPU 调用处理整个 chunk。"""
        def single(args):
            x, y, w, h, pdx, pdy = args
            ox, oy, nw, nh, npdx, npdy = PostOptimizer._full_annealing_merged(
                x, y, w, h, pdx, pdy, movable_indices, bw, bh,
                nets_ptr, pins_nodes,
                total_phases, initial_step, final_step,
                base_offsets_x, base_offsets_y,
                base_widths, base_heights, base_pins_dx, base_pins_dy,
                n_runs, reheat_factor)
            hpwl = PostOptimizer._compute_hpwl_direct(
                ox, oy, nw, nh, nets_ptr, pins_nodes, npdx, npdy)
            return ox, oy, nw, nh, npdx, npdy, hpwl
        return jax.vmap(single)((batch_x, batch_y, batch_w, batch_h,
                                 batch_pdx, batch_pdy))

    def _make_offsets(self, search_points):
        sp = search_points
        offsets_1d = jnp.arange(-sp, sp + 1, dtype=jnp.float32)
        gx, gy = jnp.meshgrid(offsets_1d, offsets_1d)
        gx, gy = gx.ravel(), gy.ravel()
        nonzero = (gx != 0) | (gy != 0)
        return jnp.where(nonzero, gx, 0.0), jnp.where(nonzero, gy, 0.0)

    # ======================== 公开方法 ========================

    def build_orderings(self, num_random_orderings: int, seed: int = 0):
        """构造用于 optimize_batch 的模块遍历顺序集合（全随机）。

        动画重放 Stage 2 时可以直接根据 ordering 索引复用同一份 ordering。
        """
        orderings = []
        rng = jax.random.PRNGKey(seed)
        for _ in range(num_random_orderings):
            rng, subkey = jax.random.split(rng)
            orderings.append(jax.random.permutation(subkey, self.movable_indices))
        return orderings

    def ordering_labels(self, num_random_orderings: int):
        return [f"随机{i+1}" for i in range(num_random_orderings)]

    def optimize_batch(self, all_x, all_y, all_w, all_h, all_pdx, all_pdy,
                       boundary_width=None, boundary_height=None,
                       max_iterations=5,
                       search_points=20, chunk_size=16,
                       num_random_orderings=4, seed=0,
                       n_runs=1, reheat_factor=0.9):
        """批量后处理（合并版：整体平移 + 方向·位置联合 sweep）。

        K 个候选方案 × 多种 ordering，每个 phase 同时做整体平移、方向翻转和
        本地位置搜索，逐候选取最优。

        Args:
            max_iterations: 退火总 phase 数（拆为 n_runs 段，每段 floor(total/n_runs)）
            n_runs: reheat 次数（1 = 单段 baseline，>=2 = N 段 reheat）
            reheat_factor: 第 2 段及以后的起始温度系数（factor × initial_step）

        Returns:
          best_x/y/w/h/pdx/pdy: 最优几何
          best_hpwl: 对应 HPWL
          best_ord_idx: (K,) 每个候选达到最优时使用的 ordering 索引
          orderings: 本次使用的 ordering 列表（顺序与 index 一致）
        """
        if boundary_width is None or boundary_height is None:
            boundary_width, boundary_height = self.bench.boundary_from_terminals()

        bw, bh = jnp.float32(boundary_width), jnp.float32(boundary_height)
        initial_step = jnp.float32(max(boundary_width, boundary_height) // search_points)
        base_offsets_x, base_offsets_y = self._make_offsets(search_points)

        base_w = jnp.array(self.bench.widths)
        base_h = jnp.array(self.bench.heights)
        base_pdx = jnp.array(self.bench.pins_dx)
        base_pdy = jnp.array(self.bench.pins_dy)

        # 把 num_random_orderings 向上对齐到 n_dev 的整数倍：每卡承担 1 个 ordering，分 n_rounds 轮。
        # 用户给 4 + 8 张卡，自动升到 8（多生成 4 个随机 ordering，多样性更高，不会变差）；
        # n_dev=1 时此条件总成立，num_random_orderings 保持原值。
        n_dev = _N_DEV
        if num_random_orderings < n_dev:
            num_random_orderings = n_dev
        elif num_random_orderings % n_dev != 0:
            num_random_orderings = ((num_random_orderings + n_dev - 1) // n_dev) * n_dev

        orderings = self.build_orderings(num_random_orderings, seed)
        labels = self.ordering_labels(num_random_orderings)

        K = all_x.shape[0]
        pad = (-K) % chunk_size  # K 向上对齐到 chunk_size 的整数倍，避免单 vmap 一次性吞 K 个候选 OOM
        if pad > 0:
            def p(a):
                return jnp.concatenate([a, jnp.zeros((pad,) + a.shape[1:], dtype=a.dtype)])
            all_x, all_y = p(all_x), p(all_y)
            all_w, all_h = p(all_w), p(all_h)
            all_pdx, all_pdy = p(all_pdx), p(all_pdy)

        total = all_x.shape[0]
        n_chunks = total // chunk_size
        best_x = jnp.zeros_like(all_x)
        best_y = jnp.zeros_like(all_y)
        best_w = jnp.zeros_like(all_w)
        best_h_geom = jnp.zeros_like(all_h)
        best_pdx = jnp.zeros_like(all_pdx)
        best_pdy = jnp.zeros_like(all_pdy)
        best_hpwl = jnp.full(total, jnp.inf)
        best_ord = jnp.full(total, -1, dtype=jnp.int32)

        # 在 ordering 维度上跨设备并行：每张卡承担 1 个 ordering 在全部 K 候选上的退火，
        # 各卡间无通信。N_DEV=1 时退化为单卡顺序跑 num_random_orderings 轮，每轮 1 个 ordering，
        # 行为和原版语义一致；多余的开销只是一次性的 jit 编译。
        #
        # 为避免 shard_map 内部 Python for 循环展开 ~K/chunk_size 个 chunk 导致编译图爆炸，
        # 把候选维度 reshape 成 (n_chunks, chunk_size, ...) 后用 jax.lax.scan 滚动调用
        # _vmap_annealing_merged。scan 体只 trace 一次，编译规模与原版 _vmap_annealing_merged 相当。
        all_x_r = all_x.reshape(n_chunks, chunk_size, *all_x.shape[1:])
        all_y_r = all_y.reshape(n_chunks, chunk_size, *all_y.shape[1:])
        all_w_r = all_w.reshape(n_chunks, chunk_size, *all_w.shape[1:])
        all_h_r = all_h.reshape(n_chunks, chunk_size, *all_h.shape[1:])
        all_pdx_r = all_pdx.reshape(n_chunks, chunk_size, *all_pdx.shape[1:])
        all_pdy_r = all_pdy.reshape(n_chunks, chunk_size, *all_pdy.shape[1:])

        @functools.partial(jax.shard_map,
                           mesh=_MESH,
                           in_specs=(P('B', None),) + (P(),) * 6,
                           out_specs=(P('B', None, None),) * 6 + (P('B', None),),
                           check_vma=False)
        def _shard_orderings(mi_b, lx_r, ly_r, lw_r, lh_r, lpdx_r, lpdy_r):
            # mi_b: (1, num_movable) per device；lx_r 等：(n_chunks, chunk_size, ...) 复制到每卡
            mi = mi_b[0]
            shared = (mi, bw, bh,
                      self.nets_ptr, self.pins_nodes,
                      jnp.int32(max_iterations),
                      initial_step, jnp.float32(1),
                      base_offsets_x, base_offsets_y,
                      base_w, base_h, base_pdx, base_pdy,
                      jnp.int32(n_runs), jnp.float32(reheat_factor))

            def scan_body(_, chunk_in):
                cx, cy, cw, ch, cpdx, cpdy = chunk_in
                return None, self._vmap_annealing_merged(
                    cx, cy, cw, ch, cpdx, cpdy, *shared)

            _, results = jax.lax.scan(
                scan_body, None,
                (lx_r, ly_r, lw_r, lh_r, lpdx_r, lpdy_r))
            # results 是 7 元组，前 6 项 (n_chunks, chunk_size, ...)，最后一项 (n_chunks, chunk_size)
            cx = results[0].reshape(total, *lx_r.shape[2:])
            cy = results[1].reshape(total, *ly_r.shape[2:])
            cw = results[2].reshape(total, *lw_r.shape[2:])
            ch = results[3].reshape(total, *lh_r.shape[2:])
            cpdx = results[4].reshape(total, *lpdx_r.shape[2:])
            cpdy = results[5].reshape(total, *lpdy_r.shape[2:])
            chpwl = results[6].reshape(total)
            # 加 leading 1 维 -> shard_map out_specs P('B', ...) 拼回 (n_dev, total, ...)
            return (cx[None], cy[None], cw[None], ch[None],
                    cpdx[None], cpdy[None], chpwl[None])

        n_rounds = num_random_orderings // n_dev
        for r in range(n_rounds):
            mi_round = jnp.stack(orderings[r * n_dev:(r + 1) * n_dev])  # (n_dev, num_movable)
            cur_x_b, cur_y_b, cur_w_b, cur_h_b, cur_pdx_b, cur_pdy_b, cur_hpwl_b = \
                _shard_orderings(mi_round, all_x_r, all_y_r, all_w_r, all_h_r, all_pdx_r, all_pdy_r)
            # 各 cur_*_b shape: (n_dev, total, ...)；cur_hpwl_b: (n_dev, total)

            # 从本轮 n_dev 个 ordering 里每个 candidate 选 HPWL 最低者，一次 vectorize
            argmin_local = jnp.argmin(cur_hpwl_b, axis=0)            # (total,)
            round_hpwl = jnp.take_along_axis(cur_hpwl_b, argmin_local[None], axis=0)[0]
            round_ord = (r * n_dev + argmin_local).astype(jnp.int32)
            def _gather(arr_b):  # (n_dev, total, M) -> (total, M)
                return jnp.take_along_axis(
                    arr_b, argmin_local[None, :, None], axis=0)[0]
            round_x   = _gather(cur_x_b)
            round_y   = _gather(cur_y_b)
            round_w   = _gather(cur_w_b)
            round_h   = _gather(cur_h_b)
            round_pdx = _gather(cur_pdx_b)
            round_pdy = _gather(cur_pdy_b)

            # 跨 round 更新 best
            improved = round_hpwl < best_hpwl
            best_x = jnp.where(improved[:, None], round_x, best_x)
            best_y = jnp.where(improved[:, None], round_y, best_y)
            best_w = jnp.where(improved[:, None], round_w, best_w)
            best_h_geom = jnp.where(improved[:, None], round_h, best_h_geom)
            best_pdx = jnp.where(improved[:, None], round_pdx, best_pdx)
            best_pdy = jnp.where(improved[:, None], round_pdy, best_pdy)
            best_ord = jnp.where(improved, round_ord, best_ord)
            best_hpwl = jnp.minimum(best_hpwl, round_hpwl)

            cur_best = float(jnp.min(best_hpwl[:K]))
            if n_dev == 1:
                # 单卡：每轮 = 1 个 ordering
                print(f"    策略 {r+1}/{num_random_orderings} [{labels[r]}] 完成, "
                      f"当前全局最优={cur_best:.0f}")
            else:
                print(f"    第 {r+1}/{n_rounds} 轮 (策略 {r*n_dev+1}~{(r+1)*n_dev}, "
                      f"{n_dev} 路 ordering 并行) 完成, 当前全局最优={cur_best:.0f}")

        return (best_x[:K], best_y[:K],
                best_w[:K], best_h_geom[:K],
                best_pdx[:K], best_pdy[:K],
                best_hpwl[:K], best_ord[:K], orderings)
