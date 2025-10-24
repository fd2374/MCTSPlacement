"""
布局状态管理模块 - 处理序列对状态和动作
"""
from __future__ import annotations

from typing import NamedTuple
import jax.numpy as jnp
import jax


class PlacementState(NamedTuple):
    """布局状态表示"""
    s1: jnp.ndarray           # 当前序列1（部分构建）
    s2: jnp.ndarray           # 当前序列2（部分构建）
    orientations: jnp.ndarray # 当前方向（0=N, 1=E, 2=S, 3=W）
    step: jnp.ndarray         # 当前步骤（0..3N-1）


class StateManager:
    """状态管理器"""
    
    @staticmethod
    def create_initial_state(num_movable: int) -> PlacementState:
        """创建初始空布局状态"""
        return PlacementState(
            s1=jnp.full(num_movable, -1, dtype=jnp.int32),
            s2=jnp.full(num_movable, -1, dtype=jnp.int32),
            orientations=jnp.full(num_movable, -1, dtype=jnp.int32),
            step=jnp.array(0, dtype=jnp.int32),
        )
    
    @staticmethod
    def apply_action(state: PlacementState, action: jnp.ndarray, 
                    num_movable: int, sorted_modules: jnp.ndarray) -> PlacementState:
        """应用动作到状态"""
        step_type = state.step % 3  # 0: s1填充, 1: s2填充, 2: 方向
        module_idx = sorted_modules[state.step // 3]
        
        def update_s1():
            """将当前模块放入s1的空位"""
            new_s1 = state.s1.at[action].set(module_idx)
            return state._replace(s1=new_s1, step=state.step + 1)

        def update_s2():
            """将当前模块放入s2的空位"""
            new_s2 = state.s2.at[action].set(module_idx)
            return state._replace(s2=new_s2, step=state.step + 1)
        
        def update_orientation():
            """设置当前模块的方向"""
            new_orient = state.orientations.at[module_idx].set(action)
            return state._replace(orientations=new_orient, step=state.step + 1)
        
        state = jax.lax.cond(
            step_type == 0,
            update_s1,
            lambda: jax.lax.cond(
                step_type == 1,
                update_s2,
                update_orientation
            )
        )
        
        return state
    
    @staticmethod
    def get_valid_actions(state: PlacementState, num_movable: int) -> jnp.ndarray:
        """获取有效动作掩码"""
        step_type = state.step % 3
        max_actions = num_movable
        
        def placement_mask(seq: jnp.ndarray) -> jnp.ndarray:
            mask = jnp.zeros(max_actions, dtype=jnp.bool_)
            available = seq == -1
            mask = mask.at[:max_actions].set(available)
            return mask
        
        def orientation_mask() -> jnp.ndarray:
            mask = jnp.zeros(max_actions, dtype=jnp.bool_)
            #will be error if max_actions < 4
            mask = mask.at[:4].set(True)
            return mask
        
        valid_mask = jax.lax.cond(
            state.step >= 3 * num_movable,
            lambda: jnp.zeros(max_actions, dtype=jnp.bool_),
            lambda: jax.lax.cond(
                step_type == 0,
                lambda: placement_mask(state.s1),
                lambda: jax.lax.cond(
                    step_type == 1,
                    lambda: placement_mask(state.s2),
                    orientation_mask
                )
            )
        )
        return valid_mask
    
    @staticmethod
    def is_terminal(state: PlacementState, num_movable: int) -> bool:
        """检查是否为终端状态"""
        return state.step >= 3 * num_movable
