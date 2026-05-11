"""
MCTS布局算法模块 - 实现基于MCTS的布局算法
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import mctx
from typing import Callable

from placement_state import PlacementState, StateManager
from placement_solver import PlacementSolver


class MCTSPlacer:
    """MCTS布局器"""
    
    def __init__(self, bench, movable_indices: jnp.ndarray, sorted_modules: jnp.ndarray,
                 boundary_width: float, boundary_height: float,
                 rollout_leaves: int = 128):
        """初始化MCTS布局器

        Args:
            bench: BookshelfData对象
            movable_indices: 可移动模块的索引
            sorted_modules: 排序后的模块
            boundary_width / boundary_height: interposer 边界（用于合法性判定）
            rollout_leaves: 每次 MCTS expansion 下并行 rollout 到终态的 leaves 数（vmap 维度）
        """
        # 存储必要的数据
        self.movable_indices = movable_indices
        self.sorted_modules = sorted_modules
        self.num_movable = len(movable_indices)
        self.max_actions = self.num_movable
        self.bench = bench  # 存储bench对象
        self.boundary_width = jnp.float32(boundary_width)
        self.boundary_height = jnp.float32(boundary_height)
        self.rollout_leaves = int(rollout_leaves)
        
        # 创建状态管理器
        self.state_manager = StateManager()
        
        # 创建布局求解器
        self.placement_solver = PlacementSolver(bench, movable_indices)
    
    def root_fn(self, state: PlacementState, max_actions: int, rng_key) -> mctx.RootFnOutput:
        """MCTS根函数"""
        return mctx.RootFnOutput(
            prior_logits=self.policy_function(state),  # 使用有效动作掩码
            value=jnp.array(0.0, dtype=jnp.float32),
            embedding=state
        )
    
    def policy_function(self, state: PlacementState) -> jnp.ndarray:
        """简单策略函数（均匀分布有效动作）"""
        valid_mask = self.state_manager.get_valid_actions(state, self.num_movable)
        logits = jnp.where(valid_mask, 0.0, -1e9)
        return logits

    def _eval_terminal(self, state: PlacementState):
        """在 terminal state 上计算 (hpwl, reward, is_legal)。

        reward = -hpwl；is_legal 通过 movable bbox 是否落在 interposer 边界内判定。
        非法解不做软惩罚，留给下游 _extract_per_batch_best 按 is_legal 过滤即可。
        """
        x, y, w, h, pdx, pdy = self.placement_solver.compute_final_positions(
            state.s1, state.s2, state.orientations)
        hpwl = PlacementSolver._calculate_hpwl_core(
            x, y, w, h, self.bench.nets_ptr, self.bench.pins_nodes, pdx, pdy)

        mi = self.movable_indices
        is_legal = ((jnp.min(x[mi]) >= 0.0) &
                    (jnp.min(y[mi]) >= 0.0) &
                    (jnp.max(x[mi] + w[mi]) <= self.boundary_width) &
                    (jnp.max(y[mi] + h[mi]) <= self.boundary_height))
        return hpwl, -hpwl, is_legal

    def _single_rollout(self, state: PlacementState, rng_key):
        """单次 rollout 到终态，返回 (leaf_state, reward, is_legal)。reward = -hpwl。"""
        def cond(a):
            state, key = a
            return state.step < 3 * self.num_movable

        def step(a):
            state, key = a
            key, subkey = jax.random.split(key)
            action = jax.random.categorical(subkey, self.policy_function(state))
            state = self.state_manager.apply_action(state, action, self.num_movable, self.sorted_modules)
            return state, key

        leaf, _ = jax.lax.while_loop(cond, step, (state, rng_key))
        _, reward, is_legal = self._eval_terminal(leaf)
        return leaf, reward, is_legal

    def rollout(self, state: PlacementState, rng_key, n_rollouts: int = 128):
        """K 次并行 rollout 到终态。

        返回：
          mcts_value: K 次 rollout reward (=-hpwl) 的最大值，作为 MCTS 节点 value
                      （不做合法优先，保持 UCT 的"上界估计"语义）。
          roll_value: **合法优先** 挑出的那次 rollout 的 reward，
                      与 best_leaf 来自同一次 rollout，供 extraction 排序使用。
          best_leaf: 同上 idx 对应的终态；合法存在时取合法里 -HPWL 最大者，
                     否则退回 reward argmax。

        解耦 mcts_value 和 roll_value 的原因：
        - MCTS 搜索喜欢紧上界（max pvals） → 分支选择更有效
        - Extraction 排序要 roll_value 与 roll_s1/s2/ori 指向同一次 rollout，
          否则 val_B 会虚高，top-K 选错
        """
        keys = jax.random.split(rng_key, n_rollouts)
        leaves, pvals, legal = jax.vmap(lambda k: self._single_rollout(state, k))(keys)

        # MCTS value：不做合法优先，取全体 reward 的 max
        mcts_value = jnp.max(pvals)

        # best_leaf：合法优先挑 idx；roll_value 用同一个 idx 保证一致
        legal_vals = jnp.where(legal, pvals, -jnp.inf)
        best_legal = jnp.argmax(legal_vals)
        best_any   = jnp.argmax(pvals)
        has_legal  = jnp.any(legal)
        best_idx   = jnp.where(has_legal, best_legal, best_any)

        roll_value = pvals[best_idx]
        best_leaf  = jax.tree_util.tree_map(lambda x: x[best_idx], leaves)
        return mcts_value, roll_value, best_leaf

    def compute_reward(self, state: PlacementState) -> jnp.ndarray:
        """计算奖励（仅在终端状态）：reward = -hpwl。"""
        is_terminal = self.state_manager.is_terminal(state, self.num_movable)
        
        def terminal_reward():
            _, r, _ = self._eval_terminal(state)
            return r
            
        reward = jax.lax.cond(
            is_terminal,
            terminal_reward,
            lambda: jnp.array(0.0, dtype=jnp.float32)
        )
        
        return reward
    
    def create_recurrent_fn(self) -> Callable:
        """创建MCTS的递归函数"""
        def recurrent_fn(params, rng_key, action, embedding):
            """MCTS的递归函数"""
            state = embedding
            
            # 应用动作
            new_state = self.state_manager.apply_action(
                state, action, self.num_movable, self.sorted_modules
            )
            
            # 检查是否为终端
            is_terminal = self.state_manager.is_terminal(new_state, self.num_movable)
            
            # 计算奖励
            reward = self.compute_reward(new_state)
            
            # 从 new_state 出发做 rollout：
            # - mcts_value：全体 rollout reward max，喂给 MCTS 做上界估计
            # - roll_value + best_leaf：合法优先配对，存到 embedding 供后处理
            mcts_value, roll_value, best_leaf = self.rollout(
                new_state, rng_key, n_rollouts=self.rollout_leaves)
            
            new_state = new_state._replace(
                roll_s1=best_leaf.s1,
                roll_s2=best_leaf.s2,
                roll_ori=best_leaf.orientations,
                roll_value=roll_value,
            )
            
            return mctx.RecurrentFnOutput(
                prior_logits=self.policy_function(new_state),
                value=mcts_value,
                reward=reward,
                discount=jnp.where(is_terminal, 0.0, 1.0),
            ), new_state
        
        return recurrent_fn
