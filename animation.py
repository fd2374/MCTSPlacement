"""动画模块 - 生成三个阶段的算法演示GIF"""
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
    """highlight: set of indices to show in gold. ghosts: [(idx, old_x, old_y), ...]."""
    x_np, y_np = np.asarray(x), np.asarray(y)
    w_np, h_np = np.asarray(w), np.asarray(h)
    mi_set = set(int(i) for i in np.asarray(movable_indices))
    hl = highlight or set()

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


def _get_boundary(bench):
    tm = np.array(bench.is_terminal) == 1
    bw = float(np.max(np.where(tm, np.array(bench.x_fixed) + np.array(bench.widths), 0)))
    bh = float(np.max(np.where(tm, np.array(bench.y_fixed) + np.array(bench.heights), 0)))
    return bw, bh


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


# ======================== Stage 1: MCTS GIF ========================

def create_mcts_gif(tree, placer, num_movable, batch_idx, output_path,
                    target_frames=30, fps=3, final_placement=None,
                    boundary_wh=None):
    t0 = time.time()
    print("  [MCTS GIF] 开始...", flush=True)
    num_sims = tree.num_simulations
    num_nodes = num_sims + 1
    target = 3 * num_movable

    print(f"  [MCTS GIF] 提取树数据 ({num_nodes} nodes)...", flush=True)
    ci = np.array(tree.children_index[batch_idx])
    vals = np.array(tree.node_values[batch_idx])
    vis = np.array(tree.node_visits[batch_idx])
    steps = np.array(tree.embeddings.step[batch_idx])
    print(f"  [MCTS GIF] 提取完成 ({time.time()-t0:.1f}s)", flush=True)

    print("  [MCTS GIF] 计算树布局...", flush=True)
    pos, _ = _tree_layout(ci, num_nodes)
    print(f"  [MCTS GIF] 树布局完成 ({time.time()-t0:.1f}s)", flush=True)

    bench = placer.bench
    mi = np.array(placer.movable_indices)
    if boundary_wh is not None:
        bw, bh = boundary_wh
    else:
        bw, bh = _get_boundary(bench)

    terminal_data = {}
    term_indices = np.where(steps == target)[0]
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
    print(f"  [MCTS GIF] 终端布局计算完成 ({time.time()-t0:.1f}s)", flush=True)

    vv = vals[vis > 0]
    vmin, vmax = (float(vv.min()), float(vv.max())) if len(vv) else (-1.0, 0.0)
    if vmin >= vmax:
        vmax = vmin + 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.RdYlGn

    # Precompute edges: (reveal_time, segment) sorted by reveal_time
    print("  [MCTS GIF] 预计算边...", flush=True)
    edge_times = []
    edge_segs = []
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

    # Precompute node data sorted by node index
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
    for fi, t in enumerate(idxs):
        fig, (ax_t, ax_p) = plt.subplots(1, 2, figsize=(16, 8))

        # Edges: binary search for visible subset
        n_vis_edges = int(np.searchsorted(edge_times, t, side='right'))
        if n_vis_edges > 0:
            ax_t.add_collection(LineCollection(
                edge_segs[:n_vis_edges], colors='#CCCCCC', linewidths=0.7, zorder=1))

        # Nodes: mask by index <= t, single scatter call
        mask = node_ids <= t
        if mask.any():
            ec = np.where((node_ids[mask] == t) & (t > 0), '#FF3333', '#555555')
            ew = np.where((node_ids[mask] == t) & (t > 0), 2.0, 0.4)
            ax_t.scatter(node_xy[mask, 0], node_xy[mask, 1],
                        s=node_sz[mask], c=node_colors[mask],
                        edgecolors=ec, linewidths=ew, zorder=2)

        ax_t.set_title(f'Search Tree  (Sim {t}/{num_sims})', fontsize=13)
        ax_t.set_xticks([])
        ax_t.set_yticks([])
        for sp in ax_t.spines.values():
            sp.set_visible(False)
        ax_t.autoscale_view()

        is_last = (fi == len(idxs) - 1)
        if is_last and final_placement is not None:
            fx, fy, fw, fh, fpdx, fpdy = final_placement
            fhpwl = float(PostOptimizer._compute_hpwl_direct(
                fx, fy, fw, fh, bench.nets_ptr, bench.pins_nodes, fpdx, fpdy))
            _draw_placement(ax_p, bench, fx, fy, fw, fh, mi, bw, bh)
            ax_p.set_title(f'Best Placement  HPWL={fhpwl:.0f}', fontsize=13)
        else:
            best_n, best_v = None, -np.inf
            for ni in terminal_data:
                if ni <= t and vals[ni] > best_v:
                    best_v, best_n = vals[ni], ni
            if best_n is not None:
                px, py, pw, ph, hpwl = terminal_data[best_n]
                _draw_placement(ax_p, bench, px, py, pw, ph, mi, bw, bh)
                ax_p.set_title(f'Best Placement  HPWL={hpwl:.0f}', fontsize=13)
            else:
                ax_p.text(0.5, 0.5, 'Searching...', transform=ax_p.transAxes,
                         ha='center', va='center', fontsize=22, color='#999999')
                ax_p.set_xlim(0, bw)
                ax_p.set_ylim(0, bh)
                ax_p.set_aspect('equal')
                ax_p.set_title('Best Placement', fontsize=13)

        fig.suptitle(f'Stage 1: MCTS Search  (Sim {t}/{num_sims})', fontsize=15, y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        frames.append(_fig_to_pil(fig))
        if (fi + 1) % 5 == 0 or fi == len(idxs) - 1:
            print(f"  [MCTS GIF] 帧 {fi+1}/{len(idxs)} ({time.time()-t0:.1f}s)", flush=True)

    for _ in range(3):
        frames.append(frames[-1].copy())
    print(f"  [MCTS GIF] 保存GIF...", flush=True)
    _save_gif(frames, output_path, fps)
    print(f"  [MCTS GIF] 完成 ({time.time()-t0:.1f}s)", flush=True)


# ======================== Stage 2: SA GIF ========================

def create_sa_gif(optimizer, init_x, init_y, w, h, pdx, pdy,
                  bench, bw, bh, search_points, num_phases,
                  output_path, fps=2,
                  target_x=None, target_y=None,
                  target_w=None, target_h=None,
                  target_pdx=None, target_pdy=None):
    t0 = time.time()
    print("  [SA GIF] 开始...", flush=True)
    opt_x = jnp.array(init_x, dtype=jnp.float32)
    opt_y = jnp.array(init_y, dtype=jnp.float32)
    widths, heights = jnp.array(w), jnp.array(h)
    p_dx, p_dy = jnp.array(pdx), jnp.array(pdy)

    mi_all = optimizer.movable_indices
    bw_j, bh_j = jnp.float32(bw), jnp.float32(bh)
    init_step = float(max(bw, bh) // search_points)
    final_step = 1.0
    base_ox, base_oy = optimizer._make_offsets(search_points)

    print(f"  [SA GIF] 尝试5种排序, 选取最优...", flush=True)
    orderings = [optimizer.movable_indices_asc, optimizer.movable_indices_desc]
    rng = jax.random.PRNGKey(0)
    for _ in range(3):
        rng, subkey = jax.random.split(rng)
        orderings.append(jax.random.permutation(subkey, optimizer.movable_indices_asc))
    labels = ["面积↑", "面积↓", "随机1", "随机2", "随机3"]
    best_mi = orderings[0]
    best_h = float('inf')
    for oi, mi_order in enumerate(orderings):
        rx, ry = PostOptimizer._full_annealing(
            opt_x, opt_y, widths, heights,
            mi_order, bw_j, bh_j,
            bench.nets_ptr, bench.pins_nodes, p_dx, p_dy,
            jnp.int32(num_phases), jnp.float32(init_step), jnp.float32(final_step),
            base_ox, base_oy)
        h_val = float(PostOptimizer._compute_hpwl_direct(
            rx, ry, widths, heights, bench.nets_ptr, bench.pins_nodes, p_dx, p_dy))
        tag = " ← best" if h_val < best_h else ""
        print(f"    排序 {labels[oi]}: HPWL={h_val:.0f}{tag}", flush=True)
        if h_val < best_h:
            best_h = h_val
            best_mi = mi_order
    mi = best_mi

    gif_phases = max(num_phases * 4, 15)
    print(f"  [SA GIF] {gif_phases} phases, {len(mi)} modules", flush=True)

    def compute_hpwl():
        return float(PostOptimizer._compute_hpwl_direct(
            opt_x, opt_y, widths, heights,
            bench.nets_ptr, bench.pins_nodes, p_dx, p_dy))

    def snap(title, hl=None, ghost_list=None):
        fig, ax = plt.subplots(figsize=(10, 10))
        _draw_placement(ax, bench, opt_x, opt_y, widths, heights, mi_all, bw, bh,
                       highlight=hl, ghosts=ghost_list)
        ax.set_title(title, fontsize=12 if hl else 13)
        fig.suptitle('Stage 2: SA Post-Optimization', fontsize=15, y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        return _fig_to_pil(fig)

    print(f"  [SA GIF] 渲染初始帧...", flush=True)
    frames = [snap(f'Initial  HPWL={compute_hpwl():.0f}')]

    for phase in range(gif_phases):
        t = phase / max(1, gif_phases - 1)
        ratio = final_step / max(init_step, final_step)
        cur_step = init_step * ratio ** t
        offsets_x = base_ox * cur_step
        offsets_y = base_oy * cur_step

        print(f"  [SA GIF] sweep {phase+1}/{gif_phases} (step={cur_step:.0f})...", end="", flush=True)
        prev_x, prev_y = np.array(opt_x), np.array(opt_y)
        opt_x, opt_y = PostOptimizer._sweep_modules(
            opt_x, opt_y, widths, heights,
            mi, offsets_x, offsets_y, bw_j, bh_j,
            bench.nets_ptr, bench.pins_nodes, p_dx, p_dy)

        cur_x, cur_y = np.array(opt_x), np.array(opt_y)
        moved = []
        for i in range(len(mi)):
            idx = int(mi[i])
            dist = abs(cur_x[idx] - prev_x[idx]) + abs(cur_y[idx] - prev_y[idx])
            if dist > 0.1:
                moved.append((idx, dist, prev_x[idx], prev_y[idx]))

        hpwl = compute_hpwl()
        print(f" {len(moved)} moved, HPWL={hpwl:.0f} ({time.time()-t0:.1f}s)", flush=True)
        if moved:
            moved.sort(key=lambda x: -x[1])
            hl_set = {m[0] for m in moved}
            ghost_top = [(m[0], m[2], m[3]) for m in moved[:5]]
            title = (f'Phase {phase+1}/{gif_phases}  Step={cur_step:.0f}  '
                     f'{len(moved)} moved  HPWL={hpwl:.0f}')
            frames.append(snap(title, hl=hl_set, ghost_list=ghost_top))

    if target_x is not None:
        opt_x = jnp.array(target_x)
        opt_y = jnp.array(target_y)
        if target_w is not None:
            widths = jnp.array(target_w)
            heights = jnp.array(target_h)
            p_dx = jnp.array(target_pdx)
            p_dy = jnp.array(target_pdy)
    print(f"  [SA GIF] 保存GIF ({len(frames)} frames)...", flush=True)
    final_hpwl = compute_hpwl()
    print(f"  [SA GIF] Final HPWL={final_hpwl:.0f}", flush=True)
    final = snap(f'Final  HPWL={final_hpwl:.0f}')
    for _ in range(4):
        frames.append(final.copy())
    _save_gif(frames, output_path, fps)
    print(f"  [SA GIF] 完成 ({time.time()-t0:.1f}s)", flush=True)


# ======================== Stage 3: Orientation GIF ========================

def create_orientation_gif(optimizer, init_x, init_y, init_w, init_h,
                           init_pdx, init_pdy, bench, bw, bh,
                           search_points, annealing_phases,
                           output_path, fps=2, max_rounds=20):
    t0 = time.time()
    print(f"  [ORI GIF] 开始 (max {max_rounds} rounds)...", flush=True)
    bw_j, bh_j = jnp.float32(bw), jnp.float32(bh)
    initial_step = jnp.float32(max(bw, bh) // search_points)
    base_ox, base_oy = optimizer._make_offsets(search_points)

    x_cands = jnp.arange(0, float(bw) + 1, 1.0)
    y_cands = jnp.arange(0, float(bh) + 1, 1.0)
    base_w, base_h = jnp.array(bench.widths), jnp.array(bench.heights)
    base_pdx, base_pdy = jnp.array(bench.pins_dx), jnp.array(bench.pins_dy)

    mi_desc = optimizer.movable_indices_desc
    mi_all = jnp.array(optimizer.movable_indices)

    ox, oy = jnp.array(init_x), jnp.array(init_y)
    cw, ch = jnp.array(init_w), jnp.array(init_h)
    cpdx, cpdy = jnp.array(init_pdx), jnp.array(init_pdy)

    def compute_hpwl():
        return float(PostOptimizer._compute_hpwl_direct(
            ox, oy, cw, ch, bench.nets_ptr, bench.pins_nodes, cpdx, cpdy))

    def snap(title, hl=None):
        fig, ax = plt.subplots(figsize=(10, 10))
        _draw_placement(ax, bench, ox, oy, cw, ch, mi_all, bw, bh, highlight=hl)
        ax.set_title(title, fontsize=12)
        fig.suptitle('Stage 3: Orientation Optimization', fontsize=15, y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        return _fig_to_pil(fig)

    print(f"  [ORI GIF] 渲染初始帧...", flush=True)
    frames = [snap(f'Initial  HPWL={compute_hpwl():.0f}')]

    for rd in range(max_rounds):
        hpwl_before = compute_hpwl()
        prev_w, prev_h = np.array(cw), np.array(ch)
        prev_x, prev_y = np.array(ox), np.array(oy)

        print(f"  [ORI GIF] Round {rd+1} orientation_sweep...", end="", flush=True)
        ox, oy, cw, ch, cpdx, cpdy = PostOptimizer._orientation_sweep(
            ox, oy, cw, ch, cpdx, cpdy, mi_desc,
            x_cands, y_cands, bw_j, bh_j,
            bench.nets_ptr, bench.pins_nodes,
            base_w, base_h, base_pdx, base_pdy)

        cur_w, cur_h = np.array(cw), np.array(ch)
        cur_x, cur_y = np.array(ox), np.array(oy)
        changed = []
        for i in range(len(mi_desc)):
            idx = int(mi_desc[i])
            if (abs(cur_w[idx] - prev_w[idx]) > 0.1 or
                abs(cur_h[idx] - prev_h[idx]) > 0.1 or
                abs(cur_x[idx] - prev_x[idx]) > 0.1 or
                abs(cur_y[idx] - prev_y[idx]) > 0.1):
                changed.append(idx)

        hpwl_sweep = compute_hpwl()
        print(f" {len(changed)} changed, HPWL={hpwl_sweep:.0f} ({time.time()-t0:.1f}s)", flush=True)
        if changed:
            hl_set = set(changed)
            names = ', '.join(bench.names[i] if i < len(bench.names) else f'M{i}'
                              for i in changed[:5])
            suffix = f' +{len(changed)-5}' if len(changed) > 5 else ''
            frames.append(snap(
                f'Round {rd+1} Sweep  {names}{suffix}  HPWL={hpwl_sweep:.0f}',
                hl=hl_set))

        print(f"  [ORI GIF] Round {rd+1} full_annealing...", end="", flush=True)
        ox, oy = PostOptimizer._full_annealing(
            ox, oy, cw, ch, optimizer.movable_indices, bw_j, bh_j,
            bench.nets_ptr, bench.pins_nodes, cpdx, cpdy,
            jnp.int32(annealing_phases), initial_step, jnp.float32(1),
            base_ox, base_oy)

        hpwl_after = compute_hpwl()
        print(f" HPWL: {hpwl_before:.0f}->{hpwl_after:.0f} ({time.time()-t0:.1f}s)", flush=True)
        frames.append(snap(
            f'Round {rd+1} Refine  HPWL: {hpwl_before:.0f} \u2192 {hpwl_after:.0f}'))

        if hpwl_after >= hpwl_before:
            print(f"  [ORI GIF] 收敛, 停止", flush=True)
            break

    print(f"  [ORI GIF] 保存GIF ({len(frames)} frames)...", flush=True)
    final = snap(f'Final  HPWL={compute_hpwl():.0f}')
    for _ in range(4):
        frames.append(final.copy())
    _save_gif(frames, output_path, fps)
    print(f"  [ORI GIF] 完成 ({time.time()-t0:.1f}s)", flush=True)
