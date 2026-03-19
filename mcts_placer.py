"""
MCTS布局算法模块 - 实现基于MCTS的布局算法

支持两种模式:
  - use_nn=False: 传统模式，rollout + uniform policy
  - use_nn=True:  神经网络模式，MLP value/policy 替换 rollout
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import mctx
from typing import Callable, Tuple

from placement_state import PlacementState, StateManager
from placement_solver import PlacementSolver
from sequence_pair import SequencePairSolver
from nn_model import forward as nn_forward


class MCTSPlacer:
    """MCTS布局器"""
    
    POLICY_TEMPERATURE = 1.5
    
    def __init__(self, bench, movable_indices: jnp.ndarray, sorted_modules: jnp.ndarray,
                 use_nn: bool = False,
                 boundary_width: float = None, boundary_height: float = None):
        self.movable_indices = movable_indices
        self.sorted_modules = sorted_modules
        self.num_movable = len(movable_indices)
        self.max_actions = self.num_movable
        self.bench = bench
        self.use_nn = use_nn
        
        if boundary_width is not None and boundary_height is not None:
            self.boundary_w = jnp.float32(boundary_width)
            self.boundary_h = jnp.float32(boundary_height)
        else:
            terminal_mask = bench.is_terminal == 1
            self.boundary_w = jnp.float32(jnp.max(
                jnp.where(terminal_mask, bench.x_fixed + bench.widths, 0)))
            self.boundary_h = jnp.float32(jnp.max(
                jnp.where(terminal_mask, bench.y_fixed + bench.heights, 0)))
        
        self.state_manager = StateManager()
        self.placement_solver = PlacementSolver(bench, movable_indices)
    
    def root_fn(self, params, state: PlacementState, max_actions: int, rng_key) -> mctx.RootFnOutput:
        """MCTS根函数"""
        if self.use_nn:
            value, logits = nn_forward(params, state, self.num_movable)
            valid_mask = self.state_manager.get_valid_actions(state, self.num_movable)
            logits = jnp.where(valid_mask, logits / self.POLICY_TEMPERATURE, -1e9)
        else:
            value = jnp.array(0.0, dtype=jnp.float32)
            logits = self._uniform_policy(state)
        return mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)
    
    def _uniform_policy(self, state: PlacementState) -> jnp.ndarray:
        valid_mask = self.state_manager.get_valid_actions(state, self.num_movable)
        return jnp.where(valid_mask, 0.0, -1e9)
    
    def _compute_final_positions(self, state: PlacementState) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self.placement_solver.compute_final_positions(
            state.s1, state.s2, state.orientations
        )
    
    def rollout(self, state: PlacementState, rng_key) -> jnp.ndarray:
        """贪心启发式 rollout（仅在 use_nn=False 时使用）"""
        def cond(a):
            state, key = a
            return state.step < 3 * self.num_movable
            
        def step(a):
            state, key = a
            step_type = state.step % 3
            valid_mask = self.state_manager.get_valid_actions(state, self.num_movable)
            first_valid = jnp.argmax(valid_mask.astype(jnp.int32))
            action = jnp.where(step_type == 2, jnp.int32(0), first_valid)
            state = self.state_manager.apply_action(state, action, self.num_movable, self.sorted_modules)
            return state, key
            
        leaf, key = jax.lax.while_loop(cond, step, (state, rng_key))
        return self.compute_reward(leaf)
    
    def compute_reward(self, state: PlacementState) -> jnp.ndarray:
        """计算奖励（仅在终端状态），含边界违规惩罚"""
        is_terminal = self.state_manager.is_terminal(state, self.num_movable)
        boundary_w = self.boundary_w
        boundary_h = self.boundary_h
        
        def terminal_reward():
            x, y, w, h, _, _ = self.placement_solver.compute_final_positions(
                state.s1, state.s2, state.orientations)
            hpwl = self.placement_solver._calculate_hpwl_core(
                x, y, w, h, self.bench.nets_ptr, self.bench.pins_nodes,
                self.bench.pins_dx, self.bench.pins_dy)
            
            mx = x[self.movable_indices]
            my = y[self.movable_indices]
            mw = w[self.movable_indices]
            mh = h[self.movable_indices]
            overflow_x = jnp.sum(jnp.maximum(0, mx + mw - boundary_w))
            overflow_y = jnp.sum(jnp.maximum(0, my + mh - boundary_h))
            overflow_neg_x = jnp.sum(jnp.maximum(0, -mx))
            overflow_neg_y = jnp.sum(jnp.maximum(0, -my))
            penalty = (overflow_x + overflow_y + overflow_neg_x + overflow_neg_y) * 10.0
            
            return -(hpwl + penalty)
            
        return jax.lax.cond(
            is_terminal,
            terminal_reward,
            lambda: jnp.array(0.0, dtype=jnp.float32)
        )
    
    def create_recurrent_fn(self) -> Callable:
        """创建MCTS的递归函数（根据 use_nn 自动选择 value/policy 来源）"""
        use_nn = self.use_nn
        num_movable = self.num_movable
        
        def recurrent_fn(params, rng_key, action, embedding):
            state = embedding
            new_state = self.state_manager.apply_action(
                state, action, self.num_movable, self.sorted_modules
            )
            is_terminal = self.state_manager.is_terminal(new_state, self.num_movable)
            reward = self.compute_reward(new_state)
            
            if use_nn:
                value, logits = nn_forward(params, new_state, num_movable)
                valid_mask = self.state_manager.get_valid_actions(new_state, num_movable)
                logits = jnp.where(valid_mask, logits / self.POLICY_TEMPERATURE, -1e9)
            else:
                value = self.rollout(new_state, rng_key)
                logits = self._uniform_policy(new_state)
            
            return mctx.RecurrentFnOutput(
                prior_logits=logits,
                value=value,
                reward=reward,
                discount=jnp.where(is_terminal, 0.0, 1.0),
            ), new_state
        
        return recurrent_fn
