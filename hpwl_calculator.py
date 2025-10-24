"""
HPWL计算模块 - 计算半周长线长
"""
from __future__ import annotations

import jax.numpy as jnp
import jax


class HPWLCalculator:
    """HPWL计算器"""
    
    @staticmethod
    @jax.jit
    def calculate_hpwl(x: jnp.ndarray,
                       y: jnp.ndarray,
                       widths: jnp.ndarray,
                       heights: jnp.ndarray,
                       nets_ptr: jnp.ndarray,
                       pins_nodes: jnp.ndarray,
                       pins_dx: jnp.ndarray,
                       pins_dy: jnp.ndarray) -> jnp.ndarray:
        """
        计算HPWL（半周长线长）
        
        Args:
            x, y: 模块位置
            widths, heights: 模块尺寸
            nets_ptr: 网络指针
            pins_nodes: 引脚节点索引
            pins_dx, pins_dy: 引脚偏移（百分比）
            
        Returns:
            HPWL值
        """
        centers_x = x + 0.5 * widths
        centers_y = y + 0.5 * heights

        pw = widths[pins_nodes]
        ph = heights[pins_nodes]

        node_x = centers_x[pins_nodes]
        node_y = centers_y[pins_nodes]

        pin_x = node_x + (pins_dx / 100.0) * pw
        pin_y = node_y + (pins_dy / 100.0) * ph

        num_nets = nets_ptr.shape[0] - 1
        counts = nets_ptr[1:] - nets_ptr[:-1]
        seg_ids = jnp.repeat(jnp.arange(num_nets, dtype=jnp.int32), counts, 
                           total_repeat_length=pins_nodes.shape[0])

        maxx = jax.ops.segment_max(pin_x, seg_ids, num_segments=num_nets)
        minx = jax.ops.segment_min(pin_x, seg_ids, num_segments=num_nets)
        maxy = jax.ops.segment_max(pin_y, seg_ids, num_segments=num_nets)
        miny = jax.ops.segment_min(pin_y, seg_ids, num_segments=num_nets)
        
        hpwl = jnp.sum((maxx - minx) + (maxy - miny))
        return hpwl
    
