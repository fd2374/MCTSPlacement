"""
MCTS布局算法模块 - 实现基于MCTS的布局算法
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import mctx
from typing import Callable, Tuple

from placement_state import PlacementState, StateManager
from placement_solver import PlacementSolver
from sequence_pair import SequencePairSolver


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
    
    
    def _compute_final_positions(self, state: PlacementState) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """计算最终位置"""
        return self.placement_solver.compute_final_positions(
            state.s1, state.s2, state.orientations
        )
    
    def rollout(self, state: PlacementState, rng_key) -> jnp.ndarray:
        """执行rollout直到结束并返回奖励"""
        def cond(a):
            state, key = a
            return state.step < 3 * self.num_movable
            
        def step(a):
            state, key = a
            key, subkey = jax.random.split(key)
            action = jax.random.categorical(subkey, self.policy_function(state))
            state = self.state_manager.apply_action(state, action, self.num_movable, self.sorted_modules)
            return state, key
            
        leaf, key = jax.lax.while_loop(cond, step, (state, rng_key))
        return self.compute_reward(leaf)
    
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
            
            return mctx.RecurrentFnOutput(
                prior_logits=self.policy_function(new_state),
                value=self.rollout(new_state, rng_key),
                reward=reward,
                discount=jnp.where(is_terminal, 0.0, 1.0),
            ), new_state
        
        return recurrent_fn
