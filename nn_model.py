"""
轻量 MLP Value/Policy Network（纯 JAX，无外部依赖）

状态编码: [s1_0..s1_{N-1}, s2_0..s2_{N-1}, ori_0..ori_{N-1}, step/(3N)]
           长度 = 3N + 1

网络结构:
  Input(3N+1) -> Linear(128) + ReLU -> Linear(64) + ReLU
  ├─ Value  Head: Linear(1)   -> scalar（估计 -HPWL）
  └─ Policy Head: Linear(N)   -> logits（动作先验）
"""
from __future__ import annotations

import json
import jax
import jax.numpy as jnp


def init_params(num_movable: int, rng_key) -> dict:
    """初始化网络参数（Xavier uniform）"""
    input_dim = 3 * num_movable + 1
    h1, h2 = 128, 64

    def xavier(key, fan_in, fan_out):
        limit = jnp.sqrt(6.0 / (fan_in + fan_out))
        return jax.random.uniform(key, (fan_in, fan_out), minval=-limit, maxval=limit)

    keys = jax.random.split(rng_key, 5)
    return {
        'w1': xavier(keys[0], input_dim, h1),
        'b1': jnp.zeros(h1),
        'w2': xavier(keys[1], h1, h2),
        'b2': jnp.zeros(h2),
        'wv': xavier(keys[2], h2, 1),
        'bv': jnp.zeros(1),
        'wp': xavier(keys[3], h2, num_movable),
        'bp': jnp.zeros(num_movable),
    }


def encode_state(state, num_movable: int) -> jnp.ndarray:
    """将 PlacementState 编码为定长向量"""
    s1_norm = state.s1.astype(jnp.float32) / jnp.maximum(num_movable, 1)
    s2_norm = state.s2.astype(jnp.float32) / jnp.maximum(num_movable, 1)
    ori_norm = state.orientations.astype(jnp.float32) / 4.0
    step_norm = state.step.astype(jnp.float32) / jnp.maximum(3 * num_movable, 1)
    return jnp.concatenate([s1_norm, s2_norm, ori_norm, step_norm[None]])


def forward(params, state, num_movable: int):
    """前向传播，返回 (value, policy_logits)

    Args:
        params: init_params 返回的权重字典
        state:  PlacementState
        num_movable: 可移动模块数

    Returns:
        value:  标量，估计的状态价值（约 -HPWL 量级）
        logits: shape (num_movable,)，动作先验 logits
    """
    x = encode_state(state, num_movable)
    x = jax.nn.relu(x @ params['w1'] + params['b1'])
    x = jax.nn.relu(x @ params['w2'] + params['b2'])
    value_normalized = (x @ params['wv'] + params['bv'])[0]
    logits = x @ params['wp'] + params['bp']
    v_mean = params.get('_value_mean', jnp.float32(0.0))
    v_std = params.get('_value_std', jnp.float32(1.0))
    value = value_normalized * v_std + v_mean
    return value, logits


def load_params(path: str) -> dict:
    """从 JSON 文件加载训练好的权重 + 归一化统计量"""
    with open(path, 'r') as f:
        raw = json.load(f)
    
    if 'params' in raw:
        params = {k: jnp.array(v, dtype=jnp.float32) for k, v in raw['params'].items()}
        params['_value_mean'] = jnp.float32(raw.get('value_mean', 0.0))
        params['_value_std'] = jnp.float32(raw.get('value_std', 1.0))
    else:
        params = {k: jnp.array(v, dtype=jnp.float32) for k, v in raw.items()}
        params['_value_mean'] = jnp.float32(0.0)
        params['_value_std'] = jnp.float32(1.0)
    return params
