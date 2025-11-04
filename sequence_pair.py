"""
序列对算法模块 - 处理序列对到位置的转换
"""
from __future__ import annotations

from typing import Tuple
import jax.numpy as jnp
import jax


class SequencePairSolver:
    """序列对求解器"""
    
    @staticmethod
    @jax.jit
    def seqpair_to_positions(s1: jnp.ndarray, s2: jnp.ndarray,
                            widths: jnp.ndarray, heights: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        将序列对(s1, s2)转换为(x, y)位置，使用约束图
        
        Args:
            s1, s2: 模块索引数组，形状 (N,)
            widths, heights: 形状 (N,)
            
        Returns:
            x, y位置，形状 (N,)
        
        序列对解释：
        - 模块a在模块b左边当且仅当：a在s1中出现在b之前 AND a在s2中出现在b之前
        - 模块a在模块b下面当且仅当：a在s1中出现在b之前 AND a在s2中出现在b之后
        """
        N = s1.shape[0]
        
        # 创建逆映射：对于每个模块ID，它在s1/s2中的位置是什么？
        pos1 = jnp.zeros(N, dtype=jnp.int32).at[s1].set(jnp.arange(N))
        pos2 = jnp.zeros(N, dtype=jnp.int32).at[s2].set(jnp.arange(N))
        
        # 使用JAX的fori_loop优化X坐标计算（替代Python循环）
        def compute_x_step(i, x):
            mod_i = s1[i]
            # 模块j在mod_i左边当且仅当：
            # j在s1和s2中都出现在mod_i之前
            mask = (pos1 < pos1[mod_i]) & (pos2 < pos2[mod_i])
            # x[mod_i]必须至少是x[j] + width[j]对于所有在左边的j
            x_candidates = jnp.where(mask, x + widths, 0.0)
            return x.at[mod_i].set(jnp.max(x_candidates))
        
        # 使用JAX的fori_loop优化Y坐标计算（替代Python循环）
        def compute_y_step(i, y):
            mod_i = s2[i]
            # 模块j在mod_i下面当且仅当：
            # j在s1中出现在mod_i之前但在s2中出现在mod_i之后
            mask = (pos2 < pos2[mod_i]) & (pos1 > pos1[mod_i])
            y_candidates = jnp.where(mask, y + heights, 0.0)
            return y.at[mod_i].set(jnp.max(y_candidates))
        
        x = jax.lax.fori_loop(0, N, compute_x_step, jnp.zeros(N, dtype=jnp.float32))
        y = jax.lax.fori_loop(0, N, compute_y_step, jnp.zeros(N, dtype=jnp.float32))
        
        return x, y
