"""
MCTS布局算法模块 - 实现基于MCTS的布局算法
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import mctx
from typing import Callable, Tuple

from placement_state import PlacementState, StateManager
from hpwl_calculator import HPWLCalculator
from sequence_pair import SequencePairSolver


class MCTSPlacer:
    """MCTS布局器"""
    
    def __init__(self, widths: jnp.ndarray, heights: jnp.ndarray,
                 nets_ptr: jnp.ndarray, pins_nodes: jnp.ndarray,
                 pins_dx: jnp.ndarray, pins_dy: jnp.ndarray,
                 num_movable: int, movable_indices: jnp.ndarray,
                 sorted_modules: jnp.ndarray):
        """初始化MCTS布局器"""
        self.widths = widths
        self.heights = heights
        self.nets_ptr = nets_ptr
        self.pins_nodes = pins_nodes
        self.pins_dx = pins_dx
        self.pins_dy = pins_dy
        self.num_movable = int(num_movable)
        self.movable_indices = movable_indices
        self.sorted_modules = jnp.asarray(sorted_modules, dtype=jnp.int32)
        
        # 创建状态管理器
        self.state_manager = StateManager()
        
        # 预计算常用值以提高性能
        self._precompute_constants()
    
    def _precompute_constants(self):
        """预计算常用常量以提高性能"""
        # 预计算按放置顺序的可移动模块宽度和高度
        self.ordered_widths = self.widths[self.sorted_modules]
        self.ordered_heights = self.heights[self.sorted_modules]
        
        # 预计算最大动作数
        self.max_actions = max(self.num_movable + 1, 4)
    
    def root_fn(self, state: PlacementState, max_actions: int, rng_key) -> mctx.RootFnOutput:
        """MCTS根函数"""
        return mctx.RootFnOutput(
            prior_logits=jnp.zeros(max_actions, dtype=jnp.float32),
            value=jnp.array(0.0, dtype=jnp.float32),
            embedding=state
        )
    
    def policy_function(self, state: PlacementState) -> jnp.ndarray:
        """简单策略函数（均匀分布有效动作）"""
        valid_mask = self.state_manager.get_valid_actions(state, self.num_movable)
        logits = jnp.where(valid_mask, 0.0, -1e9)
        return logits
    
    def _apply_orientations(self, state: PlacementState) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """应用方向到宽度和高度"""
        w = self.ordered_widths
        h = self.ordered_heights
        
        # 为E/W方向（1, 3）交换宽度/高度
        should_swap = (state.orientations == 1) | (state.orientations == 3)
        w_final = jnp.where(should_swap, h, w)
        h_final = jnp.where(should_swap, w, h)
        
        return w_final, h_final
    
    def _compute_final_positions(self, state: PlacementState) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """计算最终位置"""
        # 应用方向
        w_final, h_final = self._apply_orientations(state)
        
        # 从序列对获取位置
        x_mov, y_mov = SequencePairSolver.seqpair_to_positions(
            state.s1, state.s2, w_final, h_final
        )
        
        # 与固定终端合并
        x = jnp.zeros_like(self.widths)
        y = jnp.zeros_like(self.heights)
        x = x.at[self.sorted_modules].set(x_mov)
        y = y.at[self.sorted_modules].set(y_mov)
        
        widths_all = self.widths
        heights_all = self.heights
        widths_all = widths_all.at[self.sorted_modules].set(w_final)
        heights_all = heights_all.at[self.sorted_modules].set(h_final)
        
        return x, y, widths_all, heights_all
    
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
            # 使用优化的位置计算
            x, y, widths_all, heights_all = self._compute_final_positions(state)
            
            # 计算HPWL
            hpwl = HPWLCalculator.calculate_hpwl(
                x, y, widths_all, heights_all, 
                self.nets_ptr, self.pins_nodes, self.pins_dx, self.pins_dy
            )
            return -hpwl  # 负值因为我们想要最小化
        
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
