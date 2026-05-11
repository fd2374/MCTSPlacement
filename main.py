"""
主程序模块 - MCTS布局算法主程序
"""
from __future__ import annotations

# 必须在 import jax 之前设置：避免单卡预占 90% 显存导致其它卡放不下数据
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import time
import jax
import jax.numpy as jnp
import mctx
import functools
import numpy as np
from jax.sharding import PartitionSpec as P, AxisType

from data_loader import BookshelfLoader
from placement_state import PlacementState, StateManager
from mcts_placer import MCTSPlacer
from visualizer import PlacementVisualizer
from config import PlacementConfig
from post_optimizer import PostOptimizer

# 全局 device mesh：把 batch / 候选维度沿 'B' 轴 sharding 到所有可见 GPU。
# 这里**不**调 jax.set_mesh，而是把 mesh 显式传给 shard_map。原因：
#   - set_mesh 会让所有外部 jnp 数组（包括 self.movable_indices、bench 数据等）
#     绑定该 mesh 的 axis type，而 shard_map 内部 axis 会变成 Manual，
#     外部 Auto/Explicit 与内部 Manual 在 broadcast 时类型不匹配会报错。
#   - 不 set_mesh 时，外部数组无 mesh 绑定，进入 shard_map 后被自动当成 replicated，
#     与 in_specs=P() 等价，最稳。
N_DEV = jax.local_device_count()
MESH = jax.make_mesh((N_DEV,), ('B',), axis_types=(AxisType.Auto,))


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
            self.boundary_width, self.boundary_height = self.bench.boundary_from_terminals()
        
        print(f"总节点数: {len(self.bench.names)}")
        print(f"可移动模块: {self.num_movable}")
        print(f"终端/固定节点: {jnp.sum(self.bench.is_terminal)}")
        print(f"网络数: {len(self.bench.nets_ptr) - 1}")
        print(f"Interposer边界: {self.boundary_width:.2f} x {self.boundary_height:.2f}")
    
    def run_mcts(self) -> "mctx.PolicyOutput":
        """运行MCTS算法（多卡 SPMD：batch 维沿 mesh 轴 'B' sharding）"""
        print(f"\n运行MCTS，{self.config.num_simulations}次模拟...")

        # 随机排列模块顺序作为 MCTS 决策序列
        ordered_modules = jax.random.permutation(jax.random.PRNGKey(self.config.seed), self.movable_indices)
        
        # 创建MCTS布局器（传入 interposer 边界 + rollout leaf 并行数）
        self.placer = MCTSPlacer(
            self.bench, jnp.array(self.movable_indices), ordered_modules,
            boundary_width=self.boundary_width,
            boundary_height=self.boundary_height,
            rollout_leaves=self.config.rollout_leaves,
        )
        
        # 运行MCTS
        rng_key = jax.random.PRNGKey(self.config.seed)
        rng_key, subkey = jax.random.split(rng_key)
        
        initial_state = StateManager.create_initial_state(self.num_movable)
        recurrent_fn = jax.vmap(self.placer.create_recurrent_fn(), (None, None, 0, 0))
        
        # 把 batch 维度向上 pad 到 N_DEV 的整数倍，pad 出来的 batch 用全 0 rng 跑出来一般是越界解
        # （HPWL=+inf），后续 _extract_per_batch_best 自动过滤；输出 PolicyOutput 的 leading 维仍是 B_pad。
        B_orig = self.config.batch_size
        B_pad = ((B_orig + N_DEV - 1) // N_DEV) * N_DEV
        if B_pad != B_orig:
            print(f"  batch={B_orig} 不是 {N_DEV} 的整数倍，向上 pad 到 {B_pad}")
        
        # shard_map: 把 (B_pad, 2) 的 subkeys 沿 'B' 切到 N_DEV 张卡，每卡处理 B_pad/N_DEV 个 batch。
        # rng 是标量 key，replicate 到所有卡，但用 axis_index('B') fold 进去让各卡随机性独立。
        # N_DEV=1 时退化为单卡执行（mesh 大小为 1，shard_map 等价于 identity wrap），
        # 多一次性的 jit 编译开销（~5s），但代码无需分支。
        @functools.partial(jax.shard_map,
                           mesh=MESH,
                           in_specs=(P(), P('B', None)),
                           out_specs=P('B'),
                           check_vma=False)
        def shard_run(rng, subkeys_local):
            rng = jax.random.fold_in(rng, jax.lax.axis_index('B'))
            root = jax.vmap(self.placer.root_fn, (None, None, 0))(
                initial_state, self.placer.max_actions, subkeys_local)
            return mctx.gumbel_muzero_policy(
                params=None,
                rng_key=rng,
                root=root,
                recurrent_fn=recurrent_fn,
                num_simulations=self.config.num_simulations,
                max_depth=3 * self.num_movable,
                gumbel_scale=self.config.gumbel_scale,
                qtransform=functools.partial(mctx.qtransform_completed_by_mix_value),
            )
        
        subkeys = jax.random.split(subkey, B_pad)
        policy_output = shard_run(rng_key, subkeys)
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
        B = terminal_mask.shape[0]

        mi = jnp.array(self.movable_indices)
        bw = jnp.float32(self.boundary_width)
        bh = jnp.float32(self.boundary_height)
        compute_final = self.placer.placement_solver.compute_final_positions

        # 走 shard_map: 把 emb.s1 等 (B, N, M) 的 leading 'B' 轴沿设备切，
        # 每卡只跑 (B/N_DEV, N, M) -> (B/N_DEV, N)，无跨卡 gather。
        # 内部用 lax.scan 取代 Python for 循环，避免 jit trace 时把 chunks 全部展开
        # 造成编译图爆炸。性能瓶颈历史：
        #   - 8 卡裸跑（旧实现）：sharded array 切 1024 行 chunk 跨设备触发 all-gather
        #     ~600 次，整段 ~50s 反而比单卡（5s）慢 10x+。
        #   - shard_map + scan：单卡 ~5s，8 卡理论 ~0.7s。
        @functools.partial(jax.shard_map,
                           mesh=MESH,
                           in_specs=(P('B', None, None),) * 3,
                           out_specs=P('B', None),
                           check_vma=False)
        def _shard_bounds_check(s1l, s2l, oril):
            Bl, Nl, Ml = s1l.shape
            total_l = Bl * Nl
            flat_s1 = s1l.reshape(total_l, Ml)
            flat_s2 = s2l.reshape(total_l, Ml)
            flat_ori = oril.reshape(total_l, Ml)
            pad_l = (-total_l) % chunk_size
            if pad_l > 0:
                flat_s1 = jnp.concatenate([flat_s1, jnp.zeros((pad_l, Ml), dtype=flat_s1.dtype)])
                flat_s2 = jnp.concatenate([flat_s2, jnp.zeros((pad_l, Ml), dtype=flat_s2.dtype)])
                flat_ori = jnp.concatenate([flat_ori, jnp.zeros((pad_l, Ml), dtype=flat_ori.dtype)])
            n_chunks_l = (total_l + pad_l) // chunk_size
            c_s1 = flat_s1.reshape(n_chunks_l, chunk_size, Ml)
            c_s2 = flat_s2.reshape(n_chunks_l, chunk_size, Ml)
            c_ori = flat_ori.reshape(n_chunks_l, chunk_size, Ml)

            def body(_, inp):
                cs1, cs2, cori = inp
                x, y, w, h, _, _ = jax.vmap(compute_final)(cs1, cs2, cori)
                ok = (jnp.all(x[:, mi] >= 0, axis=1) &
                      jnp.all(y[:, mi] >= 0, axis=1) &
                      jnp.all(x[:, mi] + w[:, mi] <= bw, axis=1) &
                      jnp.all(y[:, mi] + h[:, mi] <= bh, axis=1))
                return None, ok

            _, parts = jax.lax.scan(body, None, (c_s1, c_s2, c_ori))
            flat_out = parts.reshape(-1)[:total_l]
            return flat_out.reshape(Bl, Nl)

        def _bounds_check(s1, s2, ori):
            """(B, N, M) -> (B, N) 合法掩码"""
            return _shard_bounds_check(s1, s2, ori)

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
    parser.add_argument('--annealing-phases', type=int, default=None,
                        help='退火总 phase 数（拆给 n_runs 段，每段 floor(total/n_runs)）')
    parser.add_argument('--n-runs', type=int, default=None,
                        help='退火 reheat 次数（1 = baseline 单段，>=2 = N 段，每段从 best snapshot 重启）')
    parser.add_argument('--reheat-factor', type=float, default=0.9,
                        help='reheat 段起始温度系数（cur_hot = factor × initial_step，仅 n_runs>=2 时生效）')
    parser.add_argument('--rollout-leaves', type=int, default=128,
                        help='MCTS 每节点 rollout leaf 并行数（vmap 维度）')
    parser.add_argument('--no-tree', action='store_true', help='不保存搜索树图')
    parser.add_argument('--no-viz', action='store_true', help='不保存可视化')
    parser.add_argument('--gif', action='store_true', help='生成各阶段动画GIF')
    parser.add_argument('--mcts-cache', type=str, default=None,
                        help='MCTS top-k 候选缓存 .npz 路径；存在则跳过 MCTS 直接 load，否则跑 MCTS 并 dump（用于后处理 A/B 实验）')
    
    args = parser.parse_args()
    
    # 1. 从YAML加载 或 使用默认值
    if args.config:
        config = PlacementConfig.from_yaml(args.config)
        print(f"已加载配置文件: {args.config}")
    else:
        config = PlacementConfig()
    
    # 2. 命令行参数覆盖（argparse 已自动 - → _，只覆盖显式指定的）
    cli = {k: v for k, v in vars(args).items() if k not in ('config', 'mcts_cache')}
    config.merge_cli(cli)

    config.validate()
    config.mcts_cache_path = args.mcts_cache
    return config


def main():
    """主函数"""
    config = create_config_from_args()
    runner = PlacementRunner(config)
    runner.load_benchmark()

    cache_path = config.mcts_cache_path
    cache_hit = bool(cache_path) and os.path.exists(cache_path)
    # cache 模式禁 GIF / save_tree（它们直接依赖 policy_output.search_tree）
    if cache_hit and (config.save_gif or config.save_tree):
        raise RuntimeError(
            "--mcts-cache load 模式必须配合 --no-tree 且不带 --gif（无 search_tree）")

    if cache_hit:
        load_start = time.time()
        npz = np.load(cache_path)
        all_x = jnp.asarray(npz['all_x'])
        all_y = jnp.asarray(npz['all_y'])
        all_w = jnp.asarray(npz['all_w'])
        all_h = jnp.asarray(npz['all_h'])
        all_pdx = jnp.asarray(npz['all_pdx'])
        all_pdy = jnp.asarray(npz['all_pdy'])
        top_mcts_hpwls = jnp.asarray(npz['top_mcts_hpwls'])
        k = int(all_x.shape[0])
        # 确保 expected num_movable 一致，避免后处理形状错配
        if int(npz['num_movable']) != runner.num_movable:
            raise RuntimeError(
                f"cache num_movable={int(npz['num_movable'])} 与当前 bench "
                f"num_movable={runner.num_movable} 不匹配")
        print(f"[cache] 从 {cache_path} 加载 top-{k} 候选, 用时 "
              f"{time.time() - load_start:.2f}秒（跳过 MCTS/extract/cfp/topk）")
    else:
        # ---- MCTS ----
        start = time.time()
        policy_output = runner.run_mcts()
        print(f"MCTS运行时间: {time.time() - start:.2f}秒")

        runner.save_tree(policy_output)

        # ---- 候选提取 ----
        extract_start = time.time()
        per_batch_states, per_batch_hpwls, per_batch_node, per_batch_from_roll = \
            runner._extract_per_batch_best(policy_output.search_tree)
        print(f"候选提取时间: {time.time() - extract_start:.2f}秒")
        B = int(per_batch_hpwls.shape[0])

        # ---- compute_final_positions for 全 batch ----
        cfp_start = time.time()
        all_x, all_y, all_w, all_h, all_pdx, all_pdy = jax.vmap(
            lambda s1, s2, ori: runner.placer.placement_solver.compute_final_positions(s1, s2, ori)
        )(per_batch_states.s1, per_batch_states.s2, per_batch_states.orientations)
        print(f"compute_final_positions 时间: {time.time() - cfp_start:.2f}秒")

        # ---- top-K 排序 ----
        topk_start = time.time()
        num_valid = int(jnp.sum(per_batch_hpwls < jnp.inf))
        top_k = min(config.batch_size // 5, max(num_valid, 1))
        batch_of_cand = jnp.argsort(per_batch_hpwls)[:top_k]
        all_x = all_x[batch_of_cand]
        all_y = all_y[batch_of_cand]
        all_w = all_w[batch_of_cand]
        all_h = all_h[batch_of_cand]
        all_pdx = all_pdx[batch_of_cand]
        all_pdy = all_pdy[batch_of_cand]
        top_mcts_hpwls = per_batch_hpwls[batch_of_cand]
        k = top_k
        print(f"top-K 排序时间: {time.time() - topk_start:.2f}秒")

        if cache_path:
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)) or '.', exist_ok=True)
            np.savez(cache_path,
                     all_x=np.asarray(all_x), all_y=np.asarray(all_y),
                     all_w=np.asarray(all_w), all_h=np.asarray(all_h),
                     all_pdx=np.asarray(all_pdx), all_pdy=np.asarray(all_pdy),
                     top_mcts_hpwls=np.asarray(top_mcts_hpwls),
                     num_movable=np.int32(runner.num_movable))
            print(f"[cache] 已写入 {cache_path}（top-{k} 候选）")
    
    mcts_raw_x, mcts_raw_y = all_x.copy(), all_y.copy()
    mcts_raw_w, mcts_raw_h = all_w.copy(), all_h.copy()
    mcts_raw_pdx, mcts_raw_pdy = all_pdx.copy(), all_pdy.copy()

    # ---- 后处理 ----
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
        search_points=config.search_points,
        n_runs=config.n_runs,
        reheat_factor=config.reheat_factor)
    print(f"后处理优化时间: {time.time() - start:.2f}秒")

    # 一次性 device→host，避免循环里 K 次单元素同步
    top_mcts_np = jax.device_get(top_mcts_hpwls)
    all_hpwls_np = jax.device_get(all_hpwls)
    print(f"\n候选结果（按 MCTS HPWL 升序）:")
    running_best_hpwl = float('inf')
    running_best_idx = -1
    for i in range(k):
        mcts_h = float(top_mcts_np[i])
        opt_h = float(all_hpwls_np[i])
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