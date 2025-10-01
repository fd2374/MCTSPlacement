"""
Google MCTX-based constructive placement (sequence-pair) with terminal reward = -HPWL.

Per request:
- Input files: Bookshelf **.blocks / .nets / .pl** (no .aux).
- **Action is constructive**: build the sequence pair step-by-step;
  We alternate decisions: first of three steps insert the current module into S1; second of three steps insert the current module for S2; the third step decide the direction of the current module (N/E/S/W).
  Repeat until all modules are placed (3*N steps for N movable modules).
  Sort the modules in a decreasing order of area before placement, so that larger modules are placed earlier.
  All these actions are the edges in the MCTS search tree, and all the possible (S1, S2, orientations) combinations are the nodes.
- **Reward only at the end** (after 3*N decisions): reward = **-HPWL** (unscaled).

This is a minimal, research-friendly baseline that runs on JAX + mctx.

Dependencies:
  pip install jax jaxlib mctx numpy

Tested with mctx>=0.0.6 and jax>=0.4.
"""
from __future__ import annotations

import re
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, NamedTuple, Sequence

import numpy as onp
import jax
import jax.numpy as jnp
import mctx
import chex
import pygraphviz

# -----------------------------
# Bookshelf (.blocks / .nets / .pl) reader
# -----------------------------

def _strip_comments(line: str) -> str:
    return line.split('#', 1)[0].strip()

@dataclass
class Bookshelf:
    names: List[str]                 # size G (all nodes)
    widths: onp.ndarray              # (G,)
    heights: onp.ndarray             # (G,)
    is_terminal: onp.ndarray         # (G,), 1 if terminal (fixed IO etc.)
    x_fixed: onp.ndarray             # (G,), NaN if not fixed
    y_fixed: onp.ndarray             # (G,), NaN if not fixed
    nets_ptr: onp.ndarray            # (M+1,)
    pins_nodes: onp.ndarray          # (P,), indices in [0..G)
    pins_dx: onp.ndarray             # (P,)
    pins_dy: onp.ndarray             # (P,)


def load_bookshelf(blocks_path: str, nets_path: str, pl_path: Optional[str]) -> Bookshelf:
    names, widths, heights, is_term = _parse_blocks(blocks_path)
    x_fixed, y_fixed = _parse_pl(pl_path, names) if pl_path else (onp.full(len(names), onp.nan), onp.full(len(names), onp.nan))
    nets_ptr, pins_nodes, pins_dx, pins_dy = _parse_nets(nets_path, names)
    return Bookshelf(
        names=names,
        widths=onp.asarray(widths, dtype=onp.float32),
        heights=onp.asarray(heights, dtype=onp.float32),
        is_terminal=onp.asarray(is_term, dtype=onp.int32),
        x_fixed=onp.asarray(x_fixed, dtype=onp.float32),
        y_fixed=onp.asarray(y_fixed, dtype=onp.float32),
        nets_ptr=onp.asarray(nets_ptr, dtype=onp.int32),
        pins_nodes=onp.asarray(pins_nodes, dtype=onp.int32),
        pins_dx=onp.asarray(pins_dx, dtype=onp.float32),
        pins_dy=onp.asarray(pins_dy, dtype=onp.float32),
    )


def _parse_blocks(path: str) -> Tuple[List[str], List[float], List[float], List[int]]:
    names: List[str] = []
    widths: List[float] = []
    heights: List[float] = []
    is_term: List[int] = []

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [l.rstrip('\n') for l in f]

    i = 0
    header_skip = {'UCSC', 'UCLA', 'NumBlocks', 'NumTerminals', 'NumSoftRectangularBlocks', 'NumHardRectilinearBlocks'}
    while i < len(lines):
        line = _strip_comments(lines[i])
        i += 1
        if not line or any(tok in line for tok in header_skip):
            continue
        parts = line.split()
        if len(parts) == 1 and parts[0].lower() == 'terminal':
            continue
        name = parts[0]
        w = h = None
        term = 0
        rest = ' '.join(parts[1:])
        if 'hardrectilinear' in rest.lower():
            pts = re.findall(r'\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)', rest)
            xs = [float(x) for x, _ in pts]
            ys = [float(y) for _, y in pts]
            if xs and ys:
                w = max(xs) - min(xs)
                h = max(ys) - min(ys)
        elif len(parts) >= 3:
            try:
                w = float(parts[1]); h = float(parts[2])
            except ValueError:
                pass
        if len(parts) >= 2 and parts[1].lower().startswith('terminal'):
            term = 1
        if i < len(lines):
            nxt = _strip_comments(lines[i])
            if nxt.lower() == 'terminal':
                term = 1
                i += 1
        if w is None or h is None:
            w = 0.0; h = 0.0
        names.append(name)
        widths.append(w)
        heights.append(h)
        is_term.append(term)
    return names, widths, heights, is_term


def _parse_pl(path: str, names: List[str]) -> Tuple[onp.ndarray, onp.ndarray]:
    name2idx = {n: i for i, n in enumerate(names)}
    x = onp.full((len(names),), onp.nan, dtype=onp.float32)
    y = onp.full((len(names),), onp.nan, dtype=onp.float32)
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for raw in f:
            s = _strip_comments(raw)
            if not s or 'UCLA' in s: continue
            parts = s.split()
            if len(parts) < 3: continue
            nm = parts[0]
            if nm not in name2idx: continue
            try:
                xi = float(parts[1]); yi = float(parts[2])
            except ValueError:
                continue
            idx = name2idx[nm]
            x[idx] = xi; y[idx] = yi
    return x, y


def _parse_nets(path: str, names: List[str]) -> Tuple[onp.ndarray, onp.ndarray, onp.ndarray, onp.ndarray]:
    name2idx = {n: i for i, n in enumerate(names)}
    nets_ptr: List[int] = [0]
    pins_nodes: List[int] = []
    pins_dx_pct: List[float] = []
    pins_dy_pct: List[float] = []

    cur_deg = None
    cur_cnt = 0
    pct_re = re.compile(r'%\s*(-?\d+(?:\.\d+)?)')

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for raw in f:
            s = _strip_comments(raw)
            if not s:
                continue
            if 'UCLA' in s or 'NumNets' in s or 'NumPins' in s:
                continue

            m = re.match(r'NetDegree\s*:\s*(\d+)', s, flags=re.IGNORECASE)
            if m:
                if cur_deg is not None and cur_cnt != cur_deg:
                    print(f'[warn] net degree mismatch, expected {cur_deg}, saw {cur_cnt}')
                cur_deg = int(m.group(1))
                cur_cnt = 0
                continue

            parts = s.split()
            if not parts:
                continue
            node = parts[0]
            if node not in name2idx:
                continue

            ps = pct_re.findall(s)
            if len(ps) >= 2:
                dx_pct = float(ps[0])
                dy_pct = float(ps[1])
            else:
                dx_pct = 0.0
                dy_pct = 0.0

            pins_nodes.append(name2idx[node])
            pins_dx_pct.append(dx_pct)
            pins_dy_pct.append(dy_pct)
            cur_cnt += 1

            if cur_deg is not None and cur_cnt == cur_deg:
                nets_ptr.append(nets_ptr[-1] + cur_deg)
                cur_deg = None
                cur_cnt = 0

    if cur_deg is not None and cur_cnt > 0:
        nets_ptr.append(nets_ptr[-1] + cur_cnt)

    return (
        onp.asarray(nets_ptr, dtype=onp.int32),
        onp.asarray(pins_nodes, dtype=onp.int32),
        onp.asarray(pins_dx_pct, dtype=onp.float32),
        onp.asarray(pins_dy_pct, dtype=onp.float32),
    )

# -----------------------------
# Sequence pair utilities (JAX)
# -----------------------------

@jax.jit
def seqpair_to_positions(s1: jnp.ndarray, s2: jnp.ndarray,
                         widths: jnp.ndarray, heights: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Convert sequence pair (s1, s2) to (x, y) positions using constraint graph.
    s1, s2: arrays of module indices, shape (N,)
    widths, heights: shape (N,)
    Returns: x, y positions, shape (N,)
    
    Sequence pair interpretation:
    - Module a is left of module b if: a appears before b in s1 AND a appears before b in s2
    - Module a is below module b if: a appears before b in s1 AND a appears AFTER b in s2
    """
    N = s1.shape[0]
    
    # Create inverse mappings: for each module ID, what position is it at in s1/s2?
    pos1 = jnp.zeros(N, dtype=jnp.int32).at[s1].set(jnp.arange(N))
    pos2 = jnp.zeros(N, dtype=jnp.int32).at[s2].set(jnp.arange(N))
    
    # X coordinates: process modules in s1 order
    x = jnp.zeros(N, dtype=jnp.float32)
    for i in range(N):
        mod_i = s1[i]
        # Module j is to the left of mod_i if:
        # j appears before mod_i in both s1 and s2
        # This means: pos1[j] < pos1[mod_i] AND pos2[j] < pos2[mod_i]
        mask = (pos1 < pos1[mod_i]) & (pos2 < pos2[mod_i])
        # x[mod_i] must be at least x[j] + width[j] for all j to the left
        x_candidates = jnp.where(mask, x + widths, 0.0)
        x = x.at[mod_i].set(jnp.max(x_candidates))
    
    # Y coordinates: process modules in s2 order
    y = jnp.zeros(N, dtype=jnp.float32)
    for i in range(N):
        mod_i = s2[i]
        # Module j is below mod_i if:
        # j appears before mod_i in s1 but AFTER mod_i in s2
        # This means: pos1[j] < pos1[mod_i] AND pos2[j] > pos2[mod_i]
        # But we're processing in s2 order, so we want modules that come before in s2
        # and follow the constraint properly
        mask = (pos2 < pos2[mod_i]) & (pos1 > pos1[mod_i])
        y_candidates = jnp.where(mask, y + heights, 0.0)
        y = y.at[mod_i].set(jnp.max(y_candidates))
    
    return x, y


# -----------------------------
# HPWL utilities (JAX)
# -----------------------------

@jax.jit
def hpwl_from_positions(x: jnp.ndarray,
                        y: jnp.ndarray,
                        widths: jnp.ndarray,
                        heights: jnp.ndarray,
                        nets_ptr: jnp.ndarray,
                        pins_nodes: jnp.ndarray,
                        pins_dx: jnp.ndarray,
                        pins_dy: jnp.ndarray) -> jnp.ndarray:
    """HPWL calculation with percentage-based pin offsets"""
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
    seg_ids = jnp.repeat(jnp.arange(num_nets, dtype=jnp.int32), counts, total_repeat_length=pins_nodes.shape[0])

    maxx = jax.ops.segment_max(pin_x, seg_ids, num_segments=num_nets)
    minx = jax.ops.segment_min(pin_x, seg_ids, num_segments=num_nets)
    maxy = jax.ops.segment_max(pin_y, seg_ids, num_segments=num_nets)
    miny = jax.ops.segment_min(pin_y, seg_ids, num_segments=num_nets)
    hpwl = jnp.sum((maxx - minx) + (maxy - miny))
    return hpwl

# -----------------------------
# Constructive action space
# -----------------------------

ordered_modules: jnp.ndarray  # The order in which modules are placed

class PlacementState(NamedTuple):
    """State for constructive placement"""
    s1: jnp.ndarray           # Current sequence 1 (partially built)
    s2: jnp.ndarray           # Current sequence 2 (partially built)
    orientations: jnp.ndarray # Current orientations (0=N, 1=E, 2=S, 3=W)
    step: jnp.ndarray         # Which step we're at (0..3N-1)    

def create_initial_state(num_movable: int) -> PlacementState:
    """Create initial empty placement state"""
    return PlacementState(
        s1=jnp.full(num_movable, -1, dtype=jnp.int32),
        s2=jnp.full(num_movable, -1, dtype=jnp.int32),
        orientations=jnp.full(num_movable, -1, dtype=jnp.int32),
        step=jnp.array(0, dtype=jnp.int32),
    )

# -----------------------------
# MCTX interface
# -----------------------------

"""root function for MCTX"""
def root_fn(state: PlacementState, max_actions: int, rng_key) -> mctx.RootFnOutput:                
    return mctx.RootFnOutput(
        prior_logits=jnp.zeros(max_actions, dtype=jnp.float32),
        value=jnp.array(0.0, dtype=jnp.float32),
        embedding=state
    )



def make_recurrent_fn(widths: jnp.ndarray, heights: jnp.ndarray,
                      nets_ptr: jnp.ndarray, pins_nodes: jnp.ndarray,
                      pins_dx: jnp.ndarray, pins_dy: jnp.ndarray,
                      num_movable: int, movable_indices: jnp.ndarray,
                      sorted_module: jnp.ndarray):
    """Create recurrent function for MCTX"""
    
    def apply_action(state: PlacementState, action: jnp.ndarray) -> PlacementState:
        """Apply an action to the state"""
        step_type = state.step % 3  # 0: s1 insert, 1: s2 insert, 2: orientation
        actual_module = sorted_module[state.step // 3]  # which module we're placing
        
        # Update based on step type
        def update_s1():
            pos = action                       # 标量 int
            N = num_movable
            seq = state.s1                     # (N,)
            idx = jnp.arange(N, dtype=jnp.int32)

            # 整体右移一格： [x0, x1, ..., x_{N-1}] -> [x0, x0, x1, ..., x_{N-2}]
            # （首元素占位，反正 i>pos 才会用到）
            seq_shifted = jnp.concatenate([seq[:1], seq[:-1]])

            ins_vec = jnp.full((N,), actual_module, dtype=jnp.int32)

            # 先在 i>pos 的位置用右移版本，其余用原值
            tmp = jnp.where(idx > pos, seq_shifted, seq)
            # 再在 i==pos 的位置放插入的模块 id
            new_s1 = jnp.where(idx == pos, ins_vec, tmp)

            return state._replace(s1=new_s1, step=state.step + 1)


        def update_s2():
            pos = action
            N = num_movable
            seq = state.s2
            idx = jnp.arange(N, dtype=jnp.int32)

            seq_shifted = jnp.concatenate([seq[:1], seq[:-1]])
            ins_vec = jnp.full((N,), actual_module, dtype=jnp.int32)

            tmp = jnp.where(idx > pos, seq_shifted, seq)
            new_s2 = jnp.where(idx == pos, ins_vec, tmp)

            return state._replace(s2=new_s2, step=state.step + 1)
        
        def update_orientation():
            # Set orientation for current module
            new_orient = state.orientations.at[actual_module].set(action)
            return state._replace(orientations=new_orient, step=state.step + 1)
        
        # Apply the appropriate update
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
    
    def compute_reward(state: PlacementState) -> jnp.ndarray:
        """Compute reward (only at terminal state)"""
        is_terminal = state.step >= 3 * num_movable
        
        # Only compute HPWL if terminal
        def terminal_reward():
            # Apply orientations to widths/heights
            w = widths[movable_indices]
            h = heights[movable_indices]
            
            # Swap width/height for E/W orientations (1, 3)
            should_swap = (state.orientations == 1) | (state.orientations == 3)
            w_final = jnp.where(should_swap, h, w)
            h_final = jnp.where(should_swap, w, h)
            
            # Get positions from sequence pair
            x_mov, y_mov = seqpair_to_positions(state.s1, state.s2, w_final, h_final)
            
            # Merge with fixed terminals
            x = jnp.zeros_like(widths)
            y = jnp.zeros_like(heights)
            x = x.at[movable_indices].set(x_mov)
            y = y.at[movable_indices].set(y_mov)
            
            # For terminals, use fixed positions
            is_fixed = ~jnp.isnan(jnp.zeros_like(widths))  # Placeholder, should use actual fixed info
            
            hpwl = hpwl_from_positions(x, y, widths, heights, 
                                      nets_ptr, pins_nodes, pins_dx, pins_dy)
            return -hpwl  # Negative because we want to minimize
        
        reward = jax.lax.cond(
            is_terminal,
            terminal_reward,
            lambda: jnp.array(0.0, dtype=jnp.float32)
        )
        
        return reward
    
    def recurrent_fn(params, rng_key, action, embedding):
        """Recurrent function for MCTS"""
        state = embedding
        
        # Apply action
        new_state = apply_action(state, action)
        
        # Check if terminal
        is_terminal = new_state.step >= 3 * num_movable
        
        # Compute reward
        reward = compute_reward(new_state)
        
        # Set up prior for next step
        step_type = new_state.step % 3
        module_idx = new_state.step // 3
        num_placed = module_idx
        
        max_actions = num_movable + 1
        
        # Mask valid actions (only if not terminal)
        valid_mask = jax.lax.cond(
            is_terminal,
            lambda: jnp.zeros(max_actions, dtype=bool),
            lambda: jax.lax.cond(
                step_type < 2,
                lambda: jnp.arange(max_actions) <= num_placed,
                lambda: jnp.arange(max_actions) < 4
            )
        )
        prior_logits = jnp.where(valid_mask, 0.0, -1e9)
        
        return mctx.RecurrentFnOutput(
            prior_logits=prior_logits,
            value=jnp.array(0.0, dtype=jnp.float32),
            reward=reward,
            discount=jnp.where(is_terminal, 0.0, 1.0),
        ), new_state
    
    return recurrent_fn


# -----------------------------
# Visualization
# -----------------------------

def plot_placement(bench: Bookshelf, x: onp.ndarray, y: onp.ndarray, 
                   movable_indices: onp.ndarray, output_path: str = "output_placement.png"):
    """Plot the final placement with modules and nets"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        print("Matplotlib not installed. Skipping visualization.")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # Plot modules
    for i, (name, w, h, is_term) in enumerate(zip(bench.names, bench.widths, 
                                                    bench.heights, bench.is_terminal)):
        xi, yi = x[i], y[i]
        if onp.isnan(xi) or onp.isnan(yi):
            continue
        
        # Different colors for movable vs terminal
        color = 'lightblue' if not is_term else 'lightgray'
        edgecolor = 'blue' if not is_term else 'gray'
        
        rect = patches.Rectangle((xi, yi), w, h, 
                                linewidth=1.5, 
                                edgecolor=edgecolor, 
                                facecolor=color,
                                alpha=0.7)
        ax.add_patch(rect)
    
    # Plot nets as lines connecting pin centers
    centers_x = x + 0.5 * bench.widths
    centers_y = y + 0.5 * bench.heights
    
    num_nets = len(bench.nets_ptr) - 1
    for net_idx in range(num_nets):
        start = bench.nets_ptr[net_idx]
        end = bench.nets_ptr[net_idx + 1]
        
        # Get all pin positions for this net
        pin_x = []
        pin_y = []
        for pin_idx in range(start, end):
            node_idx = bench.pins_nodes[pin_idx]
            dx_pct = bench.pins_dx[pin_idx]
            dy_pct = bench.pins_dy[pin_idx]
            
            # Calculate pin position with offset
            px = centers_x[node_idx] + (dx_pct / 100.0) * bench.widths[node_idx]
            py = centers_y[node_idx] + (dy_pct / 100.0) * bench.heights[node_idx]
            
            if not onp.isnan(px) and not onp.isnan(py):
                pin_x.append(px)
                pin_y.append(py)
        
        # Draw lines connecting all pins in this net (star topology from first pin)
        if len(pin_x) > 1:
            for i in range(1, len(pin_x)):
                ax.plot([pin_x[0], pin_x[i]], [pin_y[0], pin_y[i]], 
                       'orange', alpha=0.4, linewidth=0.8)
            
            # Draw pin dots
            ax.scatter(pin_x, pin_y, c='orange', s=10, zorder=5, alpha=0.6)
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title('Final Placement with Nets', fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlacement visualization saved to: {output_path}")
    plt.close()

def convert_tree_to_graph(
    tree: mctx.Tree,
    action_labels: Optional[Sequence[str]] = None,
    batch_index: int = 0
) -> pygraphviz.AGraph:
  """Converts a search tree into a Graphviz graph.

  Args:
    tree: A `Tree` containing a batch of search data.
    action_labels: Optional labels for edges, defaults to the action index.
    batch_index: Index of the batch element to plot.

  Returns:
    A Graphviz graph representation of `tree`.
  """
  chex.assert_rank(tree.node_values, 2)
  batch_size = tree.node_values.shape[0]
  if action_labels is None:
    action_labels = range(tree.num_actions)
  elif len(action_labels) != tree.num_actions:
    raise ValueError(
        f"action_labels {action_labels} has the wrong number of actions "
        f"({len(action_labels)}). "
        f"Expecting {tree.num_actions}.")

  def node_to_str(node_i, reward=0, discount=1):
    return (f"{node_i}\n"
            f"Reward: {reward:.2f}\n"
            f"Discount: {discount:.2f}\n"
            f"Value: {tree.node_values[batch_index, node_i]:.2f}\n"
            f"Visits: {tree.node_visits[batch_index, node_i]}\n")

  def edge_to_str(node_i, a_i):
    node_index = jnp.full([batch_size], node_i)
    probs = jax.nn.softmax(tree.children_prior_logits[batch_index, node_i])
    return (f"{action_labels[a_i]}\n"
            f"Q: {tree.qvalues(node_index)[batch_index, a_i]:.2f}\n"  # pytype: disable=unsupported-operands  # always-use-return-annotations
            f"p: {probs[a_i]:.2f}\n")

  graph = pygraphviz.AGraph(directed=True)

  # Add root
  graph.add_node(0, label=node_to_str(node_i=0), color="green")
  # Add all other nodes and connect them up.
  for node_i in range(tree.num_simulations):
    for a_i in range(tree.num_actions):
      # Index of children, or -1 if not expanded
      children_i = tree.children_index[batch_index, node_i, a_i]
      if children_i >= 0:
        graph.add_node(
            children_i,
            label=node_to_str(
                node_i=children_i,
                reward=tree.children_rewards[batch_index, node_i, a_i],
                discount=tree.children_discounts[batch_index, node_i, a_i]),
            color="red")
        graph.add_edge(node_i, children_i, label=edge_to_str(node_i, a_i))

  return graph


# -----------------------------
# CLI
# -----------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description='Constructive MCTS sequence-pair placer (terminal reward = -HPWL).')
    ap.add_argument('--blocks', default="apte.blocks", help='Path to .blocks file')
    ap.add_argument('--nets', default="apte.nets", help='Path to .nets file')
    ap.add_argument('--pl', default="apte.pl", help='Path to .pl file (for terminals)')
    ap.add_argument('--sims', type=int, default=200, help='MCTS simulations')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--batch', type=int, default=1, help='Batch size for parallel roots')

    args = ap.parse_args()

    B = args.batch

    # Load benchmark
    print(f"Loading benchmark: {args.blocks}")
    bench = load_bookshelf(args.blocks, args.nets, args.pl)
    
    # Identify movable modules
    movable_mask = bench.is_terminal == 0
    movable_indices = onp.where(movable_mask)[0]
    num_movable = len(movable_indices)
    
    print(f"Total nodes: {len(bench.names)}")
    print(f"Movable modules: {num_movable}")
    print(f"Terminal/fixed nodes: {onp.sum(bench.is_terminal)}")
    print(f"Nets: {len(bench.nets_ptr) - 1}")
    
    # Sort movable modules by area (decreasing)
    areas = bench.widths[movable_indices] * bench.heights[movable_indices]
    sorted_order = onp.argsort(-areas)  # Descending order
    ordered_modules = movable_indices[sorted_order]
    
    print(f"\nModule placement order (by decreasing area):")
    for i, idx in enumerate(ordered_modules[:5]):
        print(f"  {i+1}. {bench.names[idx]}: {bench.widths[idx]:.1f} x {bench.heights[idx]:.1f}")
    if num_movable > 5:
        print(f"  ... and {num_movable - 5} more")
    
    # Create initial state
    initial_state = create_initial_state(num_movable)
    
    # Create MCTS functions
    recurrent_fn = make_recurrent_fn(
        jnp.array(bench.widths),
        jnp.array(bench.heights),
        jnp.array(bench.nets_ptr),
        jnp.array(bench.pins_nodes),
        jnp.array(bench.pins_dx),
        jnp.array(bench.pins_dy),
        num_movable,
        jnp.array(movable_indices),
        jnp.array(ordered_modules)
    )
    
    print(f"\nRunning MCTS with {args.sims} simulations...")
    
    rng_key = jax.random.PRNGKey(args.seed)
    
    # Run MCTS
    rng_key, subkey = jax.random.split(rng_key)
    recurrent_fn=jax.vmap(recurrent_fn, (None, None, 0, 0))
    root=jax.vmap(root_fn, (None, None, 0)) (initial_state, num_movable+1, jax.random.split(subkey, B))
    
    policy_output = mctx.gumbel_muzero_policy(
        params=None,
        rng_key=rng_key,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=args.sims,
        max_depth=3 * num_movable,
        gumbel_scale=1.0
    )
    
    # Select action with highest visit count
    action = jnp.argmax(policy_output.action_weights)
    print(f"  Selected action: {int(action)}")

    graph = convert_tree_to_graph(policy_output.search_tree)
    print("Saving tree diagram to:", "search_tree.png")
    graph.draw("search_tree.png", prog="dot")
    
    # # Apply action
    # step_type_val = int(get_step_type(state))
    # module_idx_val = int(get_current_module_idx(state))
    # actual_module = int(state.module_order[module_idx_val])
    
    # if step_type_val == 0:
    #     # S1 insertion
    #     pos = int(action)
    #     new_s1 = state.s1.at[pos].set(actual_module)
    #     state = state._replace(s1=new_s1, step=state.step + 1)
    # elif step_type_val == 1:
    #     # S2 insertion
    #     pos = int(action)
    #     new_s2 = state.s2.at[pos].set(actual_module)
    #     state = state._replace(s2=new_s2, step=state.step + 1)
    # else:
    #     # Orientation
    #     new_orient = state.orientations.at[actual_module].set(action)
    #     state = state._replace(orientations=new_orient, step=state.step + 1)
    
    # print("\n" + "=" * 60)
    # print("MCTS placement complete!")
    # print("=" * 60)
    
    # # Compute final placement
    # print("\nComputing final placement...")
    
    # # Apply orientations
    # w = bench.widths[movable_indices]
    # h = bench.heights[movable_indices]
    # orientations = onp.array(state.orientations)
    
    # should_swap = (orientations == 1) | (orientations == 3)
    # w_final = onp.where(should_swap, h, w)
    # h_final = onp.where(should_swap, w, h)
    
    # # Convert sequence pair to positions
    # x_mov, y_mov = seqpair_to_positions(
    #     jnp.array(state.s1), 
    #     jnp.array(state.s2), 
    #     jnp.array(w_final), 
    #     jnp.array(h_final)
    # )
    # x_mov = onp.array(x_mov)
    # y_mov = onp.array(y_mov)
    
    # # Merge with terminals
    # x_all = onp.copy(bench.x_fixed)
    # y_all = onp.copy(bench.y_fixed)
    # x_all[movable_indices] = x_mov
    # y_all[movable_indices] = y_mov
    
    # # Compute final HPWL
    # final_hpwl = hpwl_from_positions(
    #     jnp.array(x_all), jnp.array(y_all),
    #     jnp.array(bench.widths), jnp.array(bench.heights),
    #     jnp.array(bench.nets_ptr), jnp.array(bench.pins_nodes),
    #     jnp.array(bench.pins_dx), jnp.array(bench.pins_dy)
    # )
    
    # print(f"\nFinal HPWL: {float(final_hpwl):.2f}")
    
    # # Compute bounding box
    # max_x = onp.max(x_all[~onp.isnan(x_all)] + bench.widths[~onp.isnan(x_all)])
    # max_y = onp.max(y_all[~onp.isnan(y_all)] + bench.heights[~onp.isnan(y_all)])
    # print(f"Bounding box: {max_x:.2f} x {max_y:.2f}")
    # print(f"Area: {max_x * max_y:.2f}")
    
    # # Visualize
    # plot_placement(bench, x_all, y_all, movable_indices)
    
    # # Save results
    # result = {
    #     'final_hpwl': float(final_hpwl),
    #     'bounding_box': {'width': float(max_x), 'height': float(max_y)},
    #     'area': float(max_x * max_y),
    #     'num_movable': int(num_movable),
    #     'mcts_sims': args.sims
    # }
    
    # with open('placement_result.json', 'w') as f:
    #     json.dump(result, f, indent=2)
    
    # print("\nResults saved to placement_result.json")


if __name__ == '__main__':
    main()