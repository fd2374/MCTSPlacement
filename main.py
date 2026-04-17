"""
主程序模块 - MCTS布局算法主程序
"""
from __future__ import annotations

import argparse
import time
import jax
import jax.numpy as jnp
import mctx
import functools

from data_loader import BookshelfLoader
from placement_state import PlacementState, StateManager
from mcts_placer import MCTSPlacer
from visualizer import PlacementVisualizer
from config import PlacementConfig
from post_optimizer import PostOptimizer


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
        self.placer = None  # MCTS布局器
        
    def load_benchmark(self) -> None:
        """加载基准测试数据并计算interposer边界"""
        self.bench = BookshelfLoader.load_bookshelf_from_base_path(self.config.base_path)
        
        # 识别可移动模块
        movable_mask = self.bench.is_terminal == 0
        self.movable_indices = jnp.where(movable_mask)[0]
        self.num_movable = len(self.movable_indices)
        
        # 计算interposer边界：优先使用config中手动指定的值，否则从terminal自动计算
        if self.config.boundary_width is not None and self.config.boundary_height is not None:
            self.boundary_width = self.config.boundary_width
            self.boundary_height = self.config.boundary_height
        else:
            self.boundary_width, self.boundary_height = self._calc_boundary_from_terminals()
        
        print(f"总节点数: {len(self.bench.names)}")
        print(f"可移动模块: {self.num_movable}")
        print(f"终端/固定节点: {jnp.sum(self.bench.is_terminal)}")
        print(f"网络数: {len(self.bench.nets_ptr) - 1}")
        print(f"Interposer边界: {self.boundary_width:.2f} x {self.boundary_height:.2f}")
    
    def run_mcts(self) -> tuple:
        """运行MCTS算法"""
        print(f"\n运行MCTS，{self.config.num_simulations}次模拟...")
        
        # 按面积排序模块（降序）
        ordered_modules = jax.random.permutation(jax.random.PRNGKey(self.config.seed), self.movable_indices)
        
        # 创建MCTS布局器（传入 interposer 边界 + OOB 软惩罚系数）
        self.placer = MCTSPlacer(
            self.bench, jnp.array(self.movable_indices), ordered_modules,
            boundary_width=self.boundary_width,
            boundary_height=self.boundary_height,
            oob_penalty_alpha=self.config.oob_penalty_alpha,
        )
        
        # 运行MCTS
        rng_key = jax.random.PRNGKey(self.config.seed)
        rng_key, subkey = jax.random.split(rng_key)
        
        initial_state = StateManager.create_initial_state(self.num_movable)
        recurrent_fn = jax.vmap(self.placer.create_recurrent_fn(), (None, None, 0, 0))
        root = jax.vmap(self.placer.root_fn, (None, None, 0))(
            initial_state, self.placer.max_actions, jax.random.split(subkey, self.config.batch_size)
        )
        
        policy_output = mctx.gumbel_muzero_policy(
            params=None,
            rng_key=rng_key,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=self.config.num_simulations,
            max_depth=3 * self.num_movable,
            gumbel_scale=self.config.gumbel_scale,
            qtransform=functools.partial(mctx.qtransform_completed_by_mix_value)
        )
        
        return policy_output

    def _extract_per_batch_best(self, tree, chunk_size=1024):
        """对每个 batch 取其"最优合法终端候选"（HPWL 最低且不越界）。

        候选来自两个来源：
          A) tree 中已落地的终端节点（step == 3N）；
          B) 每个 MCTS 节点在 recurrent_fn 里跑 rollout 时保留下来的最优终态
             （存在 embedding.roll_* 中，roll_value>-inf 时有效）。
        二者合并后再做合法性过滤。分块 bounds check 避免 pins_dx/pdy 中间数组
        常驻显存导致 OOM。

        返回：
          states:           PlacementState，shape (B,) —— 选中的状态
          hpwls:            (B,) 最优合法 HPWL（+inf 表示无合法解）
          node_indices:     (B,) 选中候选对应的 tree 节点下标（若来源于 rollout
                             leaf，则是发起 rollout 的父节点，供 GIF 高亮用）
          from_rollout:     (B,) 布尔，表示选中候选是否来自 rollout leaf
        """
        target_step = 3 * self.num_movable
        emb = tree.embeddings
        terminal_mask = emb.step == target_step  # (B, N)
        B, N = terminal_mask.shape
        M = emb.s1.shape[-1]

        mi = jnp.array(self.movable_indices)
        bw = jnp.float32(self.boundary_width)
        bh = jnp.float32(self.boundary_height)
        compute_final = self.placer.placement_solver.compute_final_positions

        @jax.jit
        def check_chunk(s1c, s2c, oric):
            # 只保留 x/y/w/h 用于 bounds 检查，pins_dx/pdy 立即丢弃避免留住显存
            x, y, w, h, _, _ = jax.vmap(compute_final)(s1c, s2c, oric)
            return (
                jnp.all(x[:, mi] >= 0, axis=1) &
                jnp.all(y[:, mi] >= 0, axis=1) &
                jnp.all(x[:, mi] + w[:, mi] <= bw, axis=1) &
                jnp.all(y[:, mi] + h[:, mi] <= bh, axis=1)
            )

        def _bounds_check(s1, s2, ori):
            """(B, N, M) -> (B, N) 合法掩码，走分块 vmap 控显存。"""
            total = B * N
            flat_s1 = s1.reshape(total, M)
            flat_s2 = s2.reshape(total, M)
            flat_ori = ori.reshape(total, M)
            pad = (-total) % chunk_size
            if pad > 0:
                flat_s1 = jnp.concatenate([flat_s1, jnp.zeros((pad, M), dtype=flat_s1.dtype)])
                flat_s2 = jnp.concatenate([flat_s2, jnp.zeros((pad, M), dtype=flat_s2.dtype)])
                flat_ori = jnp.concatenate([flat_ori, jnp.zeros((pad, M), dtype=flat_ori.dtype)])
            padded_total = total + pad
            parts = []
            for start in range(0, padded_total, chunk_size):
                end = start + chunk_size
                parts.append(check_chunk(
                    flat_s1[start:end], flat_s2[start:end], flat_ori[start:end]))
            return jnp.concatenate(parts)[:total].reshape(B, N)

        # 源 A：节点自身（仅 terminal 才算）
        in_bounds_A = _bounds_check(emb.s1, emb.s2, emb.orientations)
        val_A = jnp.where(terminal_mask & in_bounds_A, tree.node_values, -jnp.inf)

        # 源 B：每节点 rollout best leaf。
        # 注意 mctx 把 tree embeddings 全体用 jnp.zeros 初始化，所以"未被访问节点"的
        # roll_value=0；root 节点 roll_value=-inf（来自 initial_state）。真实 rollout
        # 写入的 reward 是 -hpwl 严格 <0 且有限，所以用 isfinite & <0 精确过滤。
        roll_valid = jnp.isfinite(emb.roll_value) & (emb.roll_value < 0)
        in_bounds_B = _bounds_check(emb.roll_s1, emb.roll_s2, emb.roll_ori)
        val_B = jnp.where(roll_valid & in_bounds_B, emb.roll_value, -jnp.inf)

        # 每个 (batch, node) 取两源较优者
        use_roll = val_B > val_A
        per_node_val = jnp.maximum(val_A, val_B)

        best_node_per_batch = jnp.argmax(per_node_val, axis=1)
        best_value_per_batch = jnp.max(per_node_val, axis=1)

        ba = jnp.arange(B)
        best_use_roll = use_roll[ba, best_node_per_batch]  # (B,)

        # 按来源挑 s1/s2/ori
        sel_s1 = jnp.where(best_use_roll[:, None],
                           emb.roll_s1[ba, best_node_per_batch],
                           emb.s1[ba, best_node_per_batch])
        sel_s2 = jnp.where(best_use_roll[:, None],
                           emb.roll_s2[ba, best_node_per_batch],
                           emb.s2[ba, best_node_per_batch])
        sel_ori = jnp.where(best_use_roll[:, None],
                            emb.roll_ori[ba, best_node_per_batch],
                            emb.orientations[ba, best_node_per_batch])

        states = PlacementState(
            s1=sel_s1, s2=sel_s2, orientations=sel_ori,
            step=jnp.full((B,), target_step, dtype=jnp.int32),
            # 后续不再使用 roll_* 字段，这里填占位值即可
            roll_s1=sel_s1, roll_s2=sel_s2, roll_ori=sel_ori,
            roll_value=best_value_per_batch,
        )
        hpwls = jnp.where(best_value_per_batch > -jnp.inf,
                          -best_value_per_batch, jnp.inf)

        # 统计
        num_tree_terms = int(jnp.sum(terminal_mask))
        num_tree_valid = int(jnp.sum(terminal_mask & in_bounds_A))
        num_roll_valid = int(jnp.sum(roll_valid & in_bounds_B))
        num_no_valid = int(jnp.sum(best_value_per_batch == -jnp.inf))
        num_from_roll = int(jnp.sum(best_use_roll & (best_value_per_batch > -jnp.inf)))
        best_hpwl = float(jnp.min(hpwls))
        print(f"  候选池: tree 终端 {num_tree_valid}/{num_tree_terms} 合法 | "
              f"rollout leaves {num_roll_valid} 合法")
        print(f"  {B - num_no_valid}/{B} 个 batch 找到合法解；"
              f"其中 {num_from_roll} 来自 rollout leaf")
        print(f"  最低合法 HPWL: {best_hpwl:.2f}")
        return states, hpwls, best_node_per_batch, best_use_roll
    
    def _calc_boundary_from_terminals(self):
        """从terminal节点计算interposer边界"""
        terminal_mask = self.bench.is_terminal == 1
        tx = jnp.where(terminal_mask, self.bench.x_fixed, 0)
        ty = jnp.where(terminal_mask, self.bench.y_fixed, 0)
        tw = jnp.where(terminal_mask, self.bench.widths, 0)
        th = jnp.where(terminal_mask, self.bench.heights, 0)
        return float(jnp.max(tx + tw)), float(jnp.max(ty + th))
    
    def plot(self, x, y, w, h, pins_dx, pins_dy, filename, title=None):
        """绘制布局图（受 --no-viz 控制）"""
        hpwl = float(PostOptimizer._compute_hpwl_direct(
            x, y, w, h, self.bench.nets_ptr, self.bench.pins_nodes, pins_dx, pins_dy
        ))
        if self.config.save_visualization:
            path = f"{self.config.output_dir}/{filename}"
            PlacementVisualizer.plot_placement(
                self.bench, jnp.array(x), jnp.array(y), jnp.array(w), jnp.array(h),
                jnp.array(pins_dx), jnp.array(pins_dy), self.movable_indices, path, draw_connections=True
            )
            print(f"  {title or filename}: HPWL={hpwl:.2f} -> {path}")
        else:
            print(f"  {title or filename}: HPWL={hpwl:.2f}")
        return hpwl
    
    def save_tree(self, policy_output):
        """保存搜索树"""
        if self.config.save_tree:
            graph = PlacementVisualizer.convert_tree_to_graph(policy_output.search_tree)
            path = f"{self.config.output_dir}/search_tree.png"
            graph.draw(path, prog="dot")
            print(f"搜索树已保存到: {path}")


def create_config_from_args() -> PlacementConfig:
    """从YAML配置文件 + 命令行参数创建配置
    
    优先级：命令行参数 > YAML文件 > 默认值
    """
    parser = argparse.ArgumentParser(description='MCTS序列对布局器')
    
    # 配置文件
    parser.add_argument('-c', '--config', type=str, default=None,
                        help='YAML配置文件路径（命令行参数可覆盖）')
    
    # 所有参数默认None，只有显式指定时才覆盖YAML
    parser.add_argument('--base-path', default=None, help='数据路径')
    parser.add_argument('--sims', type=int, default=None, help='MCTS模拟次数')
    parser.add_argument('--seed', type=int, default=None, help='随机种子')
    parser.add_argument('--batch', type=int, default=None, help='批处理大小')
    parser.add_argument('--output', default=None, help='输出目录')
    parser.add_argument('--gumbel-scale', type=float, default=None, help='Gumbel缩放因子')
    parser.add_argument('--width', type=float, default=None, help='Interposer宽度')
    parser.add_argument('--height', type=float, default=None, help='Interposer高度')
    parser.add_argument('--search-points', type=int, default=None, help='搜索点数')
    parser.add_argument('--annealing-phases', type=int, default=None, help='退火阶段数')
    parser.add_argument('--oob-penalty-alpha', type=float, default=0.0,
                        help='OOB软惩罚系数 (reward=-HPWL*(1+alpha*oob_ratio)); 默认1.0')
    parser.add_argument('--no-tree', action='store_true', help='不保存搜索树图')
    parser.add_argument('--no-viz', action='store_true', help='不保存可视化')
    parser.add_argument('--gif', action='store_true', help='生成各阶段动画GIF')
    
    args = parser.parse_args()
    
    # 1. 从YAML加载 或 使用默认值
    if args.config:
        config = PlacementConfig.from_yaml(args.config)
        print(f"已加载配置文件: {args.config}")
    else:
        config = PlacementConfig()
    
    # 2. 命令行参数覆盖（只覆盖显式指定的）
    cli = {k.replace('-', '_'): v for k, v in vars(args).items() if k != 'config'}
    config.merge_cli(cli)
    
    config.validate()
    return config


def main():
    """主函数"""
    # 创建配置
    config = create_config_from_args()
    
    # 创建布局运行器
    runner = PlacementRunner(config)
    
    # 加载基准测试（边界在此计算并存储到runner中）
    runner.load_benchmark()
    
    # 运行MCTS
    start = time.time()
    policy_output = runner.run_mcts()
    print(f"MCTS运行时间: {time.time() - start:.2f}秒")
    
    runner.save_tree(policy_output)
    
    # 提取每个 batch 的最优合法候选（tree 终端 ∪ rollout leaves，越界的自动跳过）
    per_batch_states, per_batch_hpwls, per_batch_node, per_batch_from_roll = \
        runner._extract_per_batch_best(policy_output.search_tree)
    B = int(per_batch_hpwls.shape[0])

    all_x, all_y, all_w, all_h, all_pdx, all_pdy = jax.vmap(
        lambda s1, s2, ori: runner.placer.placement_solver.compute_final_positions(s1, s2, ori)
    )(per_batch_states.s1, per_batch_states.s2, per_batch_states.orientations)

    # _extract_per_batch_best 已经过滤过越界解，这里只需按 HPWL 升序取 top-K
    num_valid = int(jnp.sum(per_batch_hpwls < jnp.inf))
    top_k = min(config.batch_size, max(num_valid, 1))
    # batch_of_cand[i] = 候选 i 来自哪个 batch（全局 batch index），
    # 配合 per_batch_node 可直接定位到 tree 中对应终端节点
    batch_of_cand = jnp.argsort(per_batch_hpwls)[:top_k]

    all_x = all_x[batch_of_cand]
    all_y = all_y[batch_of_cand]
    all_w = all_w[batch_of_cand]
    all_h = all_h[batch_of_cand]
    all_pdx = all_pdx[batch_of_cand]
    all_pdy = all_pdy[batch_of_cand]
    top_mcts_hpwls = per_batch_hpwls[batch_of_cand]
    k = top_k

    # 保留 MCTS 原始坐标用于最终对比可视化
    mcts_raw_x, mcts_raw_y = all_x.copy(), all_y.copy()
    mcts_raw_w, mcts_raw_h = all_w.copy(), all_h.copy()
    mcts_raw_pdx, mcts_raw_pdy = all_pdx.copy(), all_pdy.copy()

    # Phase 1: 批量后处理（合并版：整体平移 + 位置退火 + 方向翻转）
    print(f"\n开始后处理优化（{k} 个候选方案，合并 位置+方向，GPU 并行）...")
    start = time.time()

    optimizer = PostOptimizer(runner.bench, runner.movable_indices)
    (all_opt_x, all_opt_y,
     all_opt_w, all_opt_h,
     all_opt_pdx, all_opt_pdy,
     all_hpwls, all_best_ord, orderings) = optimizer.optimize_batch(
        all_x, all_y, all_w, all_h, all_pdx, all_pdy,
        boundary_width=runner.boundary_width,
        boundary_height=runner.boundary_height,
        max_iterations=config.annealing_phases,
        search_points=config.search_points)

    print(f"后处理优化时间: {time.time() - start:.2f}秒")

    print(f"\n候选结果（按 MCTS HPWL 升序）:")
    running_best_hpwl = float('inf')
    running_best_idx = -1
    for i in range(k):
        mcts_h = float(top_mcts_hpwls[i])
        opt_h = float(all_hpwls[i])
        if opt_h < running_best_hpwl:
            running_best_hpwl = opt_h
            running_best_idx = i
            print(f"  候选 {i+1}/{k}: MCTS={mcts_h:.0f} → PostOpt={opt_h:.0f} ← 最优")
        else:
            print(f"  候选 {i+1}/{k}: MCTS={mcts_h:.0f} → PostOpt={opt_h:.0f}")

    trace_idx = running_best_idx

    runner.plot(mcts_raw_x[trace_idx], mcts_raw_y[trace_idx],
                mcts_raw_w[trace_idx], mcts_raw_h[trace_idx],
                mcts_raw_pdx[trace_idx], mcts_raw_pdy[trace_idx],
                "stage1_mcts.png", "Stage 1: MCTS")
    runner.plot(all_opt_x[trace_idx], all_opt_y[trace_idx],
                all_opt_w[trace_idx], all_opt_h[trace_idx],
                all_opt_pdx[trace_idx], all_opt_pdy[trace_idx],
                "stage2_postopt.png", "Stage 2: PostOpt (Final)")

    if config.save_gif:
        from animation import create_mcts_gif, create_sa_gif
        gif_t0 = time.time()
        print("\n生成动画GIF...")

        gif_bw, gif_bh = float(runner.boundary_width), float(runner.boundary_height)

        # 最优解对应的 MCTS 源 (batch 索引 + 树内 node 索引)
        origin_batch = int(batch_of_cand[trace_idx])
        origin_node = int(per_batch_node[origin_batch])
        origin_from_roll = bool(per_batch_from_roll[origin_batch])
        winning_ord = int(all_best_ord[trace_idx])
        labels = optimizer.ordering_labels(len(orderings))
        src_label = "rollout leaf" if origin_from_roll else "tree terminal"
        print(f"  最优解来自 batch {origin_batch} 的 node {origin_node} ({src_label})，"
              f"使用 ordering {winning_ord} [{labels[winning_ord]}]")

        create_mcts_gif(
            policy_output.search_tree, runner.placer,
            runner.num_movable, origin_batch, origin_node,
            f'{config.output_dir}/stage1_mcts.gif',
            boundary_wh=(gif_bw, gif_bh))

        create_sa_gif(
            optimizer, runner.bench,
            mcts_raw_x[trace_idx], mcts_raw_y[trace_idx],
            mcts_raw_w[trace_idx], mcts_raw_h[trace_idx],
            mcts_raw_pdx[trace_idx], mcts_raw_pdy[trace_idx],
            orderings[winning_ord], labels[winning_ord],
            gif_bw, gif_bh,
            config.search_points, config.annealing_phases,
            f'{config.output_dir}/stage2_sa.gif')

        print(f"GIF生成总耗时: {time.time()-gif_t0:.1f}s")

    print("\n" + "="*60)
    print("完成！")
    print("="*60)


if __name__ == '__main__':
    main()