"""动画模块 - 生成 MCTS 搜索 + 后处理退火两个阶段的演示 GIF。

两份 GIF 都严格绑定到"最终被选中的最优解"：
  * create_mcts_gif: 展示最优解所在 batch 的搜索树，并在结束时高亮获胜终端节点
  * create_sa_gif:   使用 optimize_batch 选中的那个 ordering 重放真正的
                     merged annealing 过程（整体平移 + 方向·位置联合 sweep）
"""
from __future__ import annotations

import io
import time
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
from PIL import Image

from post_optimizer import PostOptimizer


# ======================== Shared Helpers ========================

def _draw_placement(ax, bench, x, y, w, h, movable_indices, bw, bh,
                    highlight=None, ghosts=None):
    """highlight: iterable of indices to show in gold.
    ghosts: [(idx, old_x, old_y), ...] 以虚线框显示上一帧的位置。"""
    x_np, y_np = np.asarray(x), np.asarray(y)
    w_np, h_np = np.asarray(w), np.asarray(h)
    mi_set = set(int(i) for i in np.asarray(movable_indices))
    hl = set(int(i) for i in (highlight or []))

    if ghosts:
        for idx, gx, gy in ghosts:
            ax.add_patch(patches.Rectangle(
                (float(gx), float(gy)), float(w_np[idx]), float(h_np[idx]),
                linewidth=1.5, edgecolor='red', facecolor='none',
                linestyle='--', alpha=0.5, zorder=3))

    for i in range(len(x_np)):
        xi, yi, wi, hi = float(x_np[i]), float(y_np[i]), float(w_np[i]), float(h_np[i])
        if np.isnan(xi) or np.isnan(yi) or (wi == 0 and hi == 0):
            continue
        if i in hl:
            fc, ec, lw = '#FFD700', '#FF4500', 2.5
        elif i in mi_set:
            fc, ec, lw = '#ADD8E6', '#4682B4', 1.2
        else:
            fc, ec, lw = '#E8E8E8', '#888888', 0.8
        ax.add_patch(patches.Rectangle(
            (xi, yi), wi, hi, linewidth=lw,
            edgecolor=ec, facecolor=fc, alpha=0.75, zorder=2))

    pad = max(bw, bh) * 0.03
    ax.set_xlim(-pad, bw + pad)
    ax.set_ylim(-pad, bh + pad)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15)


def _fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    return img


def _save_gif(frames, path, fps):
    if not frames:
        return
    max_w = max(f.width for f in frames)
    max_h = max(f.height for f in frames)
    uniform = []
    for f in frames:
        if f.size != (max_w, max_h):
            new = Image.new('RGB', (max_w, max_h), (255, 255, 255))
            new.paste(f, (0, 0))
            uniform.append(new)
        else:
            uniform.append(f.convert('RGB'))
    uniform[0].save(path, save_all=True, append_images=uniform[1:],
                    duration=1000 // max(1, fps), loop=0, optimize=False)
    print(f"  GIF: {path} ({len(uniform)} frames, {fps}fps)")


# ======================== Tree Layout ========================

def _tree_layout(children_index_np, num_nodes):
    children = {i: [] for i in range(num_nodes)}
    for p in range(min(children_index_np.shape[0], num_nodes)):
        for a in range(children_index_np.shape[1]):
            c = int(children_index_np[p, a])
            if 0 < c < num_nodes:
                children[p].append(c)

    positions = {}
    counter = [0]

    def dfs(node, depth):
        kids = sorted(k for k in children.get(node, []) if k not in positions)
        if not kids:
            positions[node] = (counter[0], -depth)
            counter[0] += 1
        else:
            for k in kids:
                dfs(k, depth + 1)
            xs = [positions[k][0] for k in kids if k in positions]
            positions[node] = (sum(xs) / len(xs) if xs else counter[0], -depth)

    dfs(0, 0)
    for i in range(num_nodes):
        if i not in positions:
            positions[i] = (counter[0], 0)
            counter[0] += 1
    return positions, children


def _path_to_root(parents_np, node):
    """返回 node -> root 的 node 索引集合（含 node 自身）。parents[0] 应为 -1/0。"""
    seen = set()
    cur = int(node)
    guard = 0
    while 0 <= cur < parents_np.shape[0] and cur not in seen:
        seen.add(cur)
        nxt = int(parents_np[cur])
        if nxt == cur or nxt < 0:
            break
        cur = nxt
        guard += 1
        if guard > parents_np.shape[0]:
            break
    return seen


# ======================== Stage 1: MCTS GIF ========================

def create_mcts_gif(tree, placer, num_movable, batch_idx, target_node,
                    output_path, target_frames=30, fps=3,
                    boundary_wh=None):
    """绘制最优解所在 batch 的 MCTS 搜索树及其 best-so-far 布局演变。

    Args:
        tree: mctx 输出的搜索树（含 batch 维）
        placer: MCTSPlacer 实例（用于反算坐标）
        num_movable: 可移动模块数
        batch_idx: 最优解所在的 batch 下标
        target_node: 最优解在该 batch 的搜索树中的 node 下标
        boundary_wh: (bw, bh)，interposer 边界
    """
    t0 = time.time()
    print(f"  [MCTS GIF] 开始 (batch={batch_idx}, target_node={target_node})...", flush=True)
    num_sims = tree.num_simulations
    num_nodes = num_sims + 1
    target_step = 3 * num_movable

    ci = np.array(tree.children_index[batch_idx])
    parents = np.array(tree.parents[batch_idx]) if hasattr(tree, 'parents') else None
    vals = np.array(tree.node_values[batch_idx])
    vis = np.array(tree.node_visits[batch_idx])
    steps = np.array(tree.embeddings.step[batch_idx])

    print("  [MCTS GIF] 计算树布局...", flush=True)
    pos, _ = _tree_layout(ci, num_nodes)

    bench = placer.bench
    mi = np.array(placer.movable_indices)
    bw, bh = boundary_wh if boundary_wh is not None else (
        float(np.max(np.where(np.array(bench.is_terminal) == 1,
                              np.array(bench.x_fixed) + np.array(bench.widths), 0))),
        float(np.max(np.where(np.array(bench.is_terminal) == 1,
                              np.array(bench.y_fixed) + np.array(bench.heights), 0))),
    )

    # 批量计算所有终端节点的布局 + HPWL
    terminal_data = {}
    term_indices = np.where(steps == target_step)[0]
    print(f"  [MCTS GIF] 找到 {len(term_indices)} 个终端节点, 批量计算布局...", flush=True)
    if len(term_indices) > 0:
        t_s1 = tree.embeddings.s1[batch_idx][term_indices]
        t_s2 = tree.embeddings.s2[batch_idx][term_indices]
        t_ori = tree.embeddings.orientations[batch_idx][term_indices]
        t_px, t_py, t_pw, t_ph, t_pdx, t_pdy = jax.vmap(
            placer.placement_solver.compute_final_positions)(t_s1, t_s2, t_ori)
        t_hpwl = jax.vmap(lambda px, py, pw, ph, pdx_, pdy_:
            PostOptimizer._compute_hpwl_direct(
                px, py, pw, ph, bench.nets_ptr, bench.pins_nodes, pdx_, pdy_)
        )(t_px, t_py, t_pw, t_ph, t_pdx, t_pdy)
        for i, ni in enumerate(term_indices):
            terminal_data[int(ni)] = (t_px[i], t_py[i], t_pw[i], t_ph[i], float(t_hpwl[i]))

    target_node = int(target_node)
    target_path = (_path_to_root(parents, target_node)
                   if parents is not None and target_node in pos else {target_node})

    vv = vals[vis > 0]
    vmin, vmax = (float(vv.min()), float(vv.max())) if len(vv) else (-1.0, 0.0)
    if vmin >= vmax:
        vmax = vmin + 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.RdYlGn

    edge_times, edge_segs = [], []
    for p in range(ci.shape[0]):
        px_t, py_t = pos.get(p, (0, 0))
        for a in range(ci.shape[1]):
            c = int(ci[p, a])
            if 0 < c < num_nodes and c in pos:
                edge_times.append(max(p, c))
                edge_segs.append([(px_t, py_t), pos[c]])
    order = np.argsort(edge_times)
    edge_times = np.array(edge_times)[order]
    edge_segs = [edge_segs[i] for i in order]

    node_ids, node_xy, node_sz, node_val = [], [], [], []
    for ni in range(num_nodes):
        if vis[ni] <= 0 and ni > 0:
            continue
        node_ids.append(ni)
        node_xy.append(pos[ni])
        node_sz.append(max(15, min(180, int(vis[ni]) * 2 + 10)))
        node_val.append(float(vals[ni]))
    node_ids = np.array(node_ids)
    node_xy = np.array(node_xy)
    node_sz = np.array(node_sz)
    node_colors = np.array([cmap(norm(v)) for v in node_val])

    frame_step = max(1, num_sims // target_frames)
    idxs = list(range(0, num_sims + 1, frame_step))
    if idxs[-1] != num_sims:
        idxs.append(num_sims)

    print(f"  [MCTS GIF] 渲染 {len(idxs)} 帧 (step={frame_step})...", flush=True)
    frames = []
    winning_revealed_at = None  # 最优终端节点首次出现的帧
    for fi, t in enumerate(idxs):
        fig, (ax_t, ax_p) = plt.subplots(1, 2, figsize=(16, 8))

        n_vis_edges = int(np.searchsorted(edge_times, t, side='right'))
        if n_vis_edges > 0:
            ax_t.add_collection(LineCollection(
                edge_segs[:n_vis_edges], colors='#CCCCCC', linewidths=0.7, zorder=1))

        mask = node_ids <= t
        if mask.any():
            on_winning_path = np.isin(node_ids[mask], list(target_path))
            just_added = (node_ids[mask] == t) & (t > 0)
            ec = np.where(on_winning_path & (node_ids[mask] == target_node) & (t >= target_node),
                          '#FF1493',
                          np.where(on_winning_path & (t >= target_node), '#FF8C00',
                                   np.where(just_added, '#FF3333', '#555555')))
            ew = np.where(on_winning_path & (t >= target_node), 2.2,
                          np.where(just_added, 2.0, 0.4))
            ax_t.scatter(node_xy[mask, 0], node_xy[mask, 1],
                        s=node_sz[mask], c=node_colors[mask],
                        edgecolors=ec, linewidths=ew, zorder=2)

        ax_t.set_title(f'Search Tree  (Sim {t}/{num_sims})', fontsize=13)
        ax_t.set_xticks([])
        ax_t.set_yticks([])
        for sp in ax_t.spines.values():
            sp.set_visible(False)
        ax_t.autoscale_view()

        # 右侧面板：已揭示的最优终端对应布局
        best_n, best_v = None, -np.inf
        for ni in terminal_data:
            if ni <= t and vals[ni] > best_v:
                best_v, best_n = vals[ni], ni
        if best_n is not None:
            px, py, pw, ph, hpwl = terminal_data[best_n]
            tag = ' [WINNER]' if best_n == target_node else ''
            _draw_placement(ax_p, bench, px, py, pw, ph, mi, bw, bh)
            ax_p.set_title(
                f'Best Placement{tag}  HPWL={hpwl:.0f}  (node {best_n})',
                fontsize=13)
            if winning_revealed_at is None and best_n == target_node:
                winning_revealed_at = t
        else:
            ax_p.text(0.5, 0.5, 'Searching...', transform=ax_p.transAxes,
                     ha='center', va='center', fontsize=22, color='#999999')
            ax_p.set_xlim(0, bw)
            ax_p.set_ylim(0, bh)
            ax_p.set_aspect('equal')
            ax_p.set_title('Best Placement', fontsize=13)

        suffix = f'  winner @ sim {winning_revealed_at}' if winning_revealed_at is not None else ''
        fig.suptitle(f'Stage 1: MCTS Search  (batch {batch_idx}, Sim {t}/{num_sims}){suffix}',
                     fontsize=15, y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        frames.append(_fig_to_pil(fig))
        if (fi + 1) % 5 == 0 or fi == len(idxs) - 1:
            print(f"  [MCTS GIF] 帧 {fi+1}/{len(idxs)} ({time.time()-t0:.1f}s)", flush=True)

    for _ in range(4):
        frames.append(frames[-1].copy())
    print(f"  [MCTS GIF] 保存GIF...", flush=True)
    _save_gif(frames, output_path, fps)
    print(f"  [MCTS GIF] 完成 ({time.time()-t0:.1f}s)", flush=True)


# ======================== Stage 2: SA GIF ========================

def create_sa_gif(optimizer, bench,
                  init_x, init_y, init_w, init_h, init_pdx, init_pdy,
                  mi_order, ord_label,
                  bw, bh, search_points, num_phases,
                  output_path, fps=2):
    """重放 optimize_batch 为选中候选所选的那一条 merged annealing 轨迹。

    每个 phase 内部分两步：
      (1) 整体平移：把 movable 集群作为刚体搬到更优位置（单帧）
      (2) 方向+位置联合 sweep：对每个模块做 4 方向 + 本地 offset 网格搜索（单帧）
    因此帧数 ≈ 1 (初始) + 2 * num_phases + 1 (final)。
    """
    t0 = time.time()
    print(f"  [SA GIF] 开始 (ordering='{ord_label}', {num_phases} phases)...", flush=True)

    ox = jnp.array(init_x, dtype=jnp.float32)
    oy = jnp.array(init_y, dtype=jnp.float32)
    w = jnp.array(init_w, dtype=jnp.float32)
    h = jnp.array(init_h, dtype=jnp.float32)
    pdx = jnp.array(init_pdx, dtype=jnp.float32)
    pdy = jnp.array(init_pdy, dtype=jnp.float32)

    bw_j, bh_j = jnp.float32(bw), jnp.float32(bh)
    initial_step = float(max(bw, bh) // search_points)
    final_step = 1.0
    base_ox, base_oy = optimizer._make_offsets(search_points)

    mi_all = jnp.array(optimizer.movable_indices)
    mi_order = jnp.array(mi_order)
    base_w = jnp.array(bench.widths)
    base_h = jnp.array(bench.heights)
    base_pdx = jnp.array(bench.pins_dx)
    base_pdy = jnp.array(bench.pins_dy)

    def compute_hpwl(x_, y_, w_, h_, pdx_, pdy_):
        return float(PostOptimizer._compute_hpwl_direct(
            x_, y_, w_, h_, bench.nets_ptr, bench.pins_nodes, pdx_, pdy_))

    def snap(title, x_, y_, w_, h_, pdx_, pdy_, hl=None, ghost_list=None):
        fig, ax = plt.subplots(figsize=(10, 10))
        _draw_placement(ax, bench, x_, y_, w_, h_, mi_all, bw, bh,
                        highlight=hl, ghosts=ghost_list)
        ax.set_title(title, fontsize=12)
        fig.suptitle(f'Stage 2: Merged Annealing  [ordering={ord_label}]',
                     fontsize=15, y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        return _fig_to_pil(fig)

    print(f"  [SA GIF] 渲染初始帧...", flush=True)
    frames = [snap(f'Initial  HPWL={compute_hpwl(ox, oy, w, h, pdx, pdy):.0f}',
                   ox, oy, w, h, pdx, pdy)]

    for phase in range(num_phases):
        t_frac = phase / max(1, num_phases - 1)
        ratio = final_step / max(initial_step, final_step)
        cur_step = initial_step * ratio ** t_frac
        offsets_x = base_ox * cur_step
        offsets_y = base_oy * cur_step

        prev_x, prev_y = np.array(ox), np.array(oy)
        prev_w, prev_h = np.array(w), np.array(h)

        # _phase_step_merged 同时返回平移后和 sweep 后的状态
        tx, ty, nx, ny, nw, nh, npdx, npdy = PostOptimizer._phase_step_merged(
            ox, oy, w, h, pdx, pdy,
            mi_order, offsets_x, offsets_y, bw_j, bh_j,
            bench.nets_ptr, bench.pins_nodes,
            base_w, base_h, base_pdx, base_pdy)

        tx_np, ty_np = np.array(tx), np.array(ty)
        shift_dx = float(np.max(np.abs(tx_np[np.asarray(mi_order)] - prev_x[np.asarray(mi_order)])))
        shift_dy = float(np.max(np.abs(ty_np[np.asarray(mi_order)] - prev_y[np.asarray(mi_order)])))
        hpwl_after_shift = compute_hpwl(tx, ty, w, h, pdx, pdy)
        print(f"  [SA GIF] phase {phase+1}/{num_phases} (step={cur_step:.0f}) "
              f"shift=({shift_dx:.0f},{shift_dy:.0f}) "
              f"HPWL_after_shift={hpwl_after_shift:.0f} "
              f"({time.time()-t0:.1f}s)", flush=True)

        if shift_dx + shift_dy > 0.1:
            frames.append(snap(
                f'Phase {phase+1}/{num_phases}  Step={cur_step:.0f}  '
                f'Global Shift (Δx={shift_dx:.0f}, Δy={shift_dy:.0f})  '
                f'HPWL={hpwl_after_shift:.0f}',
                tx, ty, w, h, pdx, pdy,
                hl=set(int(i) for i in np.asarray(mi_order))))

        # sweep 后
        hpwl_after_sweep = compute_hpwl(nx, ny, nw, nh, npdx, npdy)
        cur_x, cur_y = np.array(nx), np.array(ny)
        cur_w, cur_h = np.array(nw), np.array(nh)
        moved = []
        for i in range(len(mi_order)):
            idx = int(mi_order[i])
            pos_diff = abs(cur_x[idx] - tx_np[idx]) + abs(cur_y[idx] - ty_np[idx])
            geom_diff = abs(cur_w[idx] - prev_w[idx]) + abs(cur_h[idx] - prev_h[idx])
            if pos_diff > 0.1 or geom_diff > 0.1:
                moved.append((idx, pos_diff + geom_diff, tx_np[idx], ty_np[idx]))

        print(f"           sweep: {len(moved)} changed, "
              f"HPWL={hpwl_after_sweep:.0f}", flush=True)

        if moved:
            moved.sort(key=lambda x: -x[1])
            hl_set = {m[0] for m in moved}
            ghost_top = [(m[0], m[2], m[3]) for m in moved[:8]]
            frames.append(snap(
                f'Phase {phase+1}/{num_phases}  Step={cur_step:.0f}  '
                f'Orient+Move Sweep ({len(moved)} changed)  '
                f'HPWL={hpwl_after_sweep:.0f}',
                nx, ny, nw, nh, npdx, npdy,
                hl=hl_set, ghost_list=ghost_top))

        ox, oy, w, h, pdx, pdy = nx, ny, nw, nh, npdx, npdy

    final_hpwl = compute_hpwl(ox, oy, w, h, pdx, pdy)
    frames.append(snap(f'Final  HPWL={final_hpwl:.0f}', ox, oy, w, h, pdx, pdy))
    for _ in range(4):
        frames.append(frames[-1].copy())

    print(f"  [SA GIF] 保存GIF ({len(frames)} frames)...", flush=True)
    _save_gif(frames, output_path, fps)
    print(f"  [SA GIF] 完成, Final HPWL={final_hpwl:.0f} ({time.time()-t0:.1f}s)", flush=True)
