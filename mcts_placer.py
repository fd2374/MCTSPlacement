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
    
    def __init__(self, bench, movable_indices: jnp.ndarray, sorted_modules: jnp.ndarray):
        """初始化MCTS布局器
        
        Args:
            bench: BookshelfData对象
            movable_indices: 可移动模块的索引
            sorted_modules: 排序后的模块
        """
        # 存储必要的数据
        self.movable_indices = movable_indices
        self.sorted_modules = sorted_modules
        self.num_movable = len(movable_indices)
        self.max_actions = self.num_movable
        self.bench = bench  # 存储bench对象
        
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

    def _single_rollout(self, state: PlacementState, rng_key):
        """单次 rollout 到终态，返回 (leaf_state, reward)。"""
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
        return leaf, self.compute_reward(leaf)

    def rollout(self, state: PlacementState, rng_key, n_rollouts: int = 256):
        """K 次并行 rollout 取最大值（= 最低 HPWL），同时返回最优 leaf。

        - value：max over samples，用作 MCTS 节点的上界估计。
        - best_leaf：对应最优 reward 的那次 rollout 的终态 (s1, s2, orientations)，
          之后写入到子节点 embedding 的 roll_* 字段，供后处理候选池复用。
        """
        keys = jax.random.split(rng_key, n_rollouts)
        leaves, values = jax.vmap(lambda k: self._single_rollout(state, k))(keys)
        best_idx = jnp.argmax(values)
        best_value = values[best_idx]
        best_leaf = jax.tree_util.tree_map(lambda x: x[best_idx], leaves)
        return best_value, best_leaf
    
    def compute_reward(self, state: PlacementState) -> jnp.ndarray:
        """计算奖励（仅在终端状态）"""
        is_terminal = self.state_manager.is_terminal(state, self.num_movable)
        
        def terminal_reward():
            # 使用布局求解器计算HPWL
            hpwl = self.placement_solver.compute_hpwl(
                state.s1, state.s2, state.orientations
            )
            return -hpwl  # 负值因为我们想要最小化pygraphviz
            
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
            
            # 从 new_state 出发做 rollout，拿到最优 value 以及对应的终态 leaf。
            best_value, best_leaf = self.rollout(new_state, rng_key)
            
            # 把最优 rollout leaf 挂到当前节点的 embedding，作为后处理候选之一。
            new_state = new_state._replace(
                roll_s1=best_leaf.s1,
                roll_s2=best_leaf.s2,
                roll_ori=best_leaf.orientations,
                roll_value=best_value,
            )
            
            return mctx.RecurrentFnOutput(
                prior_logits=self.policy_function(new_state),
                value=best_value,
                reward=reward,
                discount=jnp.where(is_terminal, 0.0, 1.0),
            ), new_state
        
        return recurrent_fn
