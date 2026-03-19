"""
训练脚本 - 用 MCTS 自我对弈数据训练 value/policy 网络

流程:
  1. 用无网络 MCTS 跑多轮，收集搜索树中的 (state, value, best_action) 数据
  2. 对 value target 做在线归一化 (z-score)
  3. 用收集的数据监督学习训练 MLP
  4. 保存训练好的权重 + 归一化统计量

用法:
    python train_nn.py --base-path ./data/apte --episodes 30 --sims 1000 --batch 100
"""
from __future__ import annotations

import argparse
import json
import time
import jax
import jax.numpy as jnp
import mctx
import functools
import optax
from pathlib import Path

from data_loader import BookshelfLoader
from placement_state import StateManager, PlacementState
from mcts_placer import MCTSPlacer
from nn_model import init_params, forward, encode_state


def collect_data_from_tree(tree, num_movable, num_actions):
    """从 MCTS 搜索树中提取训练数据（向量化，零 Python 循环）"""
    target_step = 3 * num_movable
    
    valid_mask = (tree.node_visits >= 2) & (tree.embeddings.step < target_step)
    
    s1_flat = tree.embeddings.s1[valid_mask]
    s2_flat = tree.embeddings.s2[valid_mask]
    ori_flat = tree.embeddings.orientations[valid_mask]
    step_flat = tree.embeddings.step[valid_mask]
    
    if s1_flat.shape[0] == 0:
        return None
    
    N = num_movable
    s1_norm = s1_flat.astype(jnp.float32) / jnp.maximum(N, 1)
    s2_norm = s2_flat.astype(jnp.float32) / jnp.maximum(N, 1)
    ori_norm = ori_flat.astype(jnp.float32) / 4.0
    step_norm = step_flat.astype(jnp.float32) / jnp.maximum(3 * N, 1)
    states = jnp.concatenate([s1_norm, s2_norm, ori_norm, step_norm[:, None]], axis=-1)
    
    values = tree.node_values[valid_mask]
    
    child_visits = tree.children_visits[valid_mask][:, :num_actions]
    total_child = jnp.sum(child_visits, axis=-1, keepdims=True)
    uniform = jnp.ones((child_visits.shape[0], num_actions)) / num_actions
    policies = jnp.where(total_child > 0,
                         child_visits / jnp.maximum(total_child, 1),
                         uniform)
    
    return {
        'states': states,
        'values': values,
        'policies': policies,
    }


def make_loss_fn(optimizer_ref):
    """创建损失函数和训练步骤（闭包引用 optimizer）"""
    
    def loss_fn(params, batch, value_mean, value_std):
        states = batch['states']
        value_targets_normalized = (batch['values'] - value_mean) / jnp.maximum(value_std, 1e-6)
        policy_targets = batch['policies']
        
        def single_forward(x_encoded):
            x = jax.nn.relu(x_encoded @ params['w1'] + params['b1'])
            x = jax.nn.relu(x @ params['w2'] + params['b2'])
            value = (x @ params['wv'] + params['bv'])[0]
            logits = x @ params['wp'] + params['bp']
            return value, logits
        
        values, logits = jax.vmap(single_forward)(states)
        
        value_loss = jnp.mean((values - value_targets_normalized) ** 2)
        
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        policy_loss = -jnp.mean(jnp.sum(policy_targets * log_probs, axis=-1))
        
        return value_loss + policy_loss, (value_loss, policy_loss)
    
    @jax.jit
    def train_step(params, opt_state, batch, value_mean, value_std):
        (loss, (v_loss, p_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, batch, value_mean, value_std)
        updates, opt_state = optimizer_ref[0].update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, v_loss, p_loss
    
    return loss_fn, train_step


def run_mcts_episode(bench, movable_indices, ordered_modules, num_movable,
                     sims, batch_size, seed):
    """跑一次 MCTS 并返回搜索树"""
    placer = MCTSPlacer(bench, jnp.array(movable_indices), ordered_modules, use_nn=False)
    
    rng_key = jax.random.PRNGKey(seed)
    rng_key, subkey = jax.random.split(rng_key)
    
    initial_state = StateManager.create_initial_state(num_movable)
    recurrent_fn = jax.vmap(placer.create_recurrent_fn(), (None, None, 0, 0))
    root = jax.vmap(placer.root_fn, (None, None, None, 0))(
        None, initial_state, placer.max_actions,
        jax.random.split(subkey, batch_size)
    )
    
    policy_output = mctx.gumbel_muzero_policy(
        params=None,
        rng_key=rng_key,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=sims,
        max_depth=3 * num_movable,
        gumbel_scale=1.0,
        qtransform=functools.partial(mctx.qtransform_completed_by_mix_value)
    )
    
    return policy_output


def main():
    parser = argparse.ArgumentParser(description='训练 value/policy 网络')
    parser.add_argument('--base-path', default='./data/apte', help='benchmark 路径')
    parser.add_argument('--episodes', type=int, default=30, help='MCTS 对弈轮数')
    parser.add_argument('--sims', type=int, default=1000, help='每轮模拟次数')
    parser.add_argument('--batch', type=int, default=100, help='MCTS batch size')
    parser.add_argument('--train-epochs', type=int, default=200, help='每轮收集后训练的 epoch 数')
    parser.add_argument('--lr', type=float, default=3e-4, help='学习率')
    parser.add_argument('--output', default='./nn_weights.json', help='权重输出路径')
    args = parser.parse_args()
    
    bench = BookshelfLoader.load_bookshelf_from_base_path(args.base_path)
    movable_mask = bench.is_terminal == 0
    movable_indices = jnp.where(movable_mask)[0]
    num_movable = len(movable_indices)
    
    areas = bench.widths[movable_indices] * bench.heights[movable_indices]
    ordered_modules = movable_indices[jnp.argsort(-areas)]
    
    print(f"Benchmark: {args.base_path}")
    print(f"Modules: {num_movable}, Episodes: {args.episodes}, "
          f"Sims: {args.sims}, Batch: {args.batch}, LR: {args.lr}")
    print(f"{'='*60}")
    
    rng_key = jax.random.PRNGKey(42)
    rng_key, init_key = jax.random.split(rng_key)
    params = init_params(num_movable, init_key)
    
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(params)
    optimizer_ref = [optimizer]
    _, train_step = make_loss_fn(optimizer_ref)
    
    all_data = None
    value_mean = jnp.float32(0.0)
    value_std = jnp.float32(1.0)
    best_loss = float('inf')
    
    for episode in range(args.episodes):
        seed = episode * 137 + 7
        
        t0 = time.time()
        policy_output = run_mcts_episode(
            bench, movable_indices, ordered_modules, num_movable,
            args.sims, args.batch, seed)
        mcts_time = time.time() - t0
        
        ep_data = collect_data_from_tree(
            policy_output.search_tree, num_movable, num_movable)
        
        if ep_data is None:
            print(f"Episode {episode+1}: no data collected")
            continue
        
        if all_data is None:
            all_data = ep_data
        else:
            all_data = {
                'states': jnp.concatenate([all_data['states'], ep_data['states']]),
                'values': jnp.concatenate([all_data['values'], ep_data['values']]),
                'policies': jnp.concatenate([all_data['policies'], ep_data['policies']]),
            }
        
        n_samples = all_data['states'].shape[0]
        
        MAX_SAMPLES = 100000
        if n_samples > MAX_SAMPLES:
            all_data = {k: v[-MAX_SAMPLES:] for k, v in all_data.items()}
            n_samples = MAX_SAMPLES
        
        value_mean = jnp.mean(all_data['values'])
        value_std = jnp.std(all_data['values'])
        
        for epoch in range(args.train_epochs):
            params, opt_state, loss, v_loss, p_loss = train_step(
                params, opt_state, all_data, value_mean, value_std)
        
        loss_val = float(loss)
        v_loss_val = float(v_loss)
        p_loss_val = float(p_loss)
        
        tag = ""
        if loss_val < best_loss:
            best_loss = loss_val
            tag = " *"
        
        target_min = float(jnp.min(all_data['values']))
        target_max = float(jnp.max(all_data['values']))
        
        print(f"Ep {episode+1:3d}/{args.episodes} | "
              f"MCTS {mcts_time:.1f}s | "
              f"samples={n_samples:5d} | "
              f"loss={loss_val:.4f} (v={v_loss_val:.4f} p={p_loss_val:.4f}) | "
              f"raw range=[{target_min:.0f}, {target_max:.0f}]{tag}")
    
    print(f"\n{'='*60}")
    print(f"Training complete. Saving weights...")
    
    save_data = {
        'params': {k: v.tolist() for k, v in params.items()},
        'value_mean': float(value_mean),
        'value_std': float(value_std),
        'num_movable': num_movable,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(save_data, f)
    print(f"Weights saved to: {output_path}")
    print(f"Total samples: {all_data['states'].shape[0]}")
    print(f"Value normalization: mean={float(value_mean):.2f}, std={float(value_std):.2f}")
    print(f"Best loss: {best_loss:.4f}")


if __name__ == '__main__':
    main()
