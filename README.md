# MCTS 序列对布局器

基于 **Gumbel MuZero MCTS + GPU 并行位置/方向退火** 的 2.5D interposer chiplet 布局引擎。整个流水线用 JAX / mctx 编写，前向搜索与后处理全部 vmap/jit 化。

---

## 流水线概览

```
Bookshelf 数据 → MCTS 搜索 → 合法候选提取 → 并行后处理 → 最终布局 + 可视化/GIF
```

1. **MCTS 搜索**（`mcts_placer.py`）：Gumbel MuZero + Sequential Halving，单步动作是"在 s1/s2 序列对的空位放入下一个模块，或选择方向"。每次 `recurrent_fn` 里同步跑 `n_rollouts` 次随机 rollout 估计节点 value，同时**保留 rollout best leaf** 作为额外候选。
2. **合法候选提取**（`main.py::_extract_per_batch_best`）：把两路候选合并成候选池——
   - **源 A**：tree 中已落地的终端节点
   - **源 B**：每个 MCTS 节点 embedding 里挂的 rollout best leaf
   - 分块跑几何 bounds-check，越界的置 -inf，每个 batch argmax 选最优合法候选。
3. **并行后处理**（`post_optimizer.py`）：对 top-K 个候选做**合并退火**——整体平移 + 位置扰动 + 方向翻转，vmap 跨候选 + lax.fori_loop 跨迭代，全 GPU 并行。
4. **动画生成**（`animation.py`，可选）：GIF 同步展示 MCTS 搜索树和对应的位置演变。

---

## 文件结构

| 模块 | 作用 |
|---|---|
| `data_loader.py` | Bookshelf 格式 (`.blocks` / `.nets` / `.pl`) 解析 |
| `placement_state.py` | `PlacementState` NamedTuple（包含 `roll_*` 字段存 rollout best leaf），`StateManager` 负责状态转移与合法动作 |
| `sequence_pair.py` | 序列对 → 坐标（纯 JAX，基于 LCS 的 O(M²) 实现） |
| `placement_solver.py` | 组合 seqpair + 方向应用 + pin 偏移旋转 → 最终坐标 + HPWL |
| `mcts_placer.py` | `MCTSPlacer`：`root_fn`、`recurrent_fn`、带 OOB 惩罚的 `compute_reward`、合法优先 `rollout` |
| `main.py` | `PlacementRunner`：YAML/CLI 配置、加载 benchmark、MCTS、候选提取、调用 post-opt、绘图 |
| `post_optimizer.py` | `PostOptimizer.optimize_batch`：合并版退火（translate + shuffle + orient），多 ordering 并行搜索 |
| `animation.py` | `create_mcts_gif` + `create_sa_gif`：搜索过程与退火过程可视化 |
| `visualizer.py` | 静态布局图、搜索树 graphviz 图 |
| `config.py` | `PlacementConfig` dataclass + YAML 加载 |
| `configs/*.yaml` | 各 benchmark 的参数档案（apte / xerox / hp / 多 die testcases） |

---

## 关键特性

### 1. OOB 软惩罚（`oob_penalty_alpha`）

MCTS 在 `recurrent_fn` 的 rollout 终态 reward 上加乘性惩罚：

```
reward = -HPWL * (1 + alpha * oob_ratio)
oob_ratio = (bbox 左右越界之和)/interposer_width + (bbox 上下越界之和)/interposer_height
oob_ratio ← clip(oob_ratio, 0, 10)
```

- `alpha=0`：等同无惩罚（回退到纯 -HPWL）
- `alpha=1~3`：对紧约束 benchmark（interposer 贴合模块尺寸）有效，PUCT 会自发避开越界分支
- 合法解 `oob_ratio=0` 时 penalty=1，不影响 ranking

### 2. Rollout best leaf 作为候选

每个 MCTS 节点在 `recurrent_fn` 里跑 n_rollouts 次，**合法优先**挑出一个 best_leaf 挂到 embedding 的 `roll_*` 字段。Extraction 时与 tree 自身终端合并，显著扩大候选池（尤其对紧约束场景效果明显）。

- MCTS 的 `value` 回传用 `max(pvals)`，保持 UCT 紧上界语义
- embedding 的 `roll_value` 与 `roll_s1/s2/ori` 来自同一次 rollout（合法优先），保证 extraction 排序一致

### 3. 合并版后处理

单一 `_phase_step_merged` 内同时做：
- **整体平移**：候选所有模块沿 x/y 平移若干像素
- **位置扰动**：在预先生成的 orderings 中挑一组、按顺序对每个模块做小幅坐标抖动
- **方向翻转**：对每个模块 4 个方向 vmap 并行评估 HPWL，取最佳

每个 ordering 独立贪心下降，4 组 ordering 并行，外层 batch 维再 vmap，最终 `(ordering, candidate)` argmin → 汇总为每个候选的最终 HPWL。

### 4. 输出

- `stage1_mcts.png` / `stage2_postopt.png`：MCTS 起点 vs 后处理终点对比图
- `search_tree.png`（可选）：graphviz 渲染的 MCTS 树
- `stage1_mcts.gif` / `stage2_sa.gif`（`--gif`）：搜索树扩展 + 布局演化动画

---

## 使用方法

### YAML 配置（推荐）

`configs/apte.yaml`：

```yaml
base_path: "./data/apte"
sims: 1000
batch: 1000
width: 10500
height: 10500
search_points: 10
annealing_phases: 10
oob_penalty_alpha: 1.0
no_tree: true
```

运行：

```bash
python main.py --config ./configs/apte.yaml
python main.py --config ./configs/apte.yaml --gif        # 带动画
python main.py --config ./configs/apte.yaml --oob-penalty-alpha 0.5   # 命令行覆盖
```

### 直接用命令行

```bash
python main.py --base-path ./data/apte --sims 1000 --batch 1000 \
               --width 10500 --height 10500 \
               --search-points 10 --annealing-phases 10 \
               --oob-penalty-alpha 1.0
```

优先级：**命令行 > YAML > 默认值**。命令行只覆盖用户显式指定的参数（`None` 不覆盖）。

---

## 参数参考

| 参数 | CLI | YAML 键 | 默认 | 说明 |
|---|---|---|---|---|
| 数据路径 | `--base-path` | `base_path` | `./data/apte` | 自动拼 `.blocks / .nets / .pl` |
| 模拟次数 | `--sims` | `sims` | 100 | MCTS 每个 batch 的 simulation 数 |
| 种子 | `--seed` | `seed` | 0 | PRNG 种子 |
| Batch 大小 | `--batch` | `batch` | 1 | MCTS 并行 batch 数（候选池规模） |
| Gumbel 缩放 | `--gumbel-scale` | `gumbel_scale` | 0.1 | Gumbel MuZero 探索强度 |
| OOB 惩罚 | `--oob-penalty-alpha` | `oob_penalty_alpha` | 1.0 | 见上节；紧约束可调到 2~3，松约束可用 0.3~0.5 |
| Interposer 宽 | `--width` | `width` | None | None 时从 terminal 自动推算 |
| Interposer 高 | `--height` | `height` | None | 同上 |
| 后处理搜索点 | `--search-points` | `search_points` | 20 | 平移和位置扰动的采样密度 |
| 后处理迭代 | `--annealing-phases` | `annealing_phases` | 5 | 退火相位数 |
| 输出目录 | `--output` | `output` | `.` | |
| 不存树图 | `--no-tree` | `no_tree` | False | |
| 不存布局图 | `--no-viz` | `no_viz` | False | |
| 生成 GIF | `--gif` | `gif` | False | 耗时较长，调试用 |

---

## 依赖

```bash
pip install jax jaxlib mctx numpy matplotlib pygraphviz pyyaml imageio
```

GPU 版 JAX：按 CUDA 版本选 `jax[cuda12_pip]` 等 wheel。

---

## 输出日志示例

```
总节点数: 82
可移动模块: 9
终端/固定节点: 73
网络数: 97
Interposer边界: 10500.00 x 10500.00

运行MCTS，1000次模拟...
MCTS运行时间: 37.76秒
  候选池: tree 终端 4751/7154 合法 | rollout leaves 812 合法
  998/1000 个 batch 找到合法解；其中 121 来自 rollout leaf
  最低合法 HPWL: 467178.00

开始后处理优化（201 个候选方案，合并 位置+方向，GPU 并行）...
    策略 1/4 [随机1] 完成，当前全局最优=464233
    策略 2/4 [随机2] 完成，当前全局最优=463985
    ...
后处理优化时间: 13.35秒

候选结果（按 MCTS HPWL 升序）:
  候选 1/201: MCTS=467178 → PostOpt=463985 ← 最优
  ...
```

---

## 调参建议

| 场景 | 建议 |
|---|---|
| 大 interposer / 松约束（apte、xerox） | `alpha=0.5~1.0`，`n_rollouts` 默认 256 |
| 紧约束（8die_big 等） | `alpha=1.0~3.0`，可以提 batch 到 1000+ 以增加候选池多样性 |
| 想加速但精度可让步 | 把 `mcts_placer.rollout(n_rollouts=...)` 从 256 降到 32~64 |
| 最终 HPWL 停滞不降 | 加大 `search_points` 和 `annealing_phases`，或把 `alpha` 降低以让 MCTS 探索更广 |

---

## 技术要点

- **序列对表示**：拓扑布局，避免显式坐标约束，对 MCTS 动作离散化友好
- **JAX JIT 全链路**：`recurrent_fn`、`rollout`、`_phase_step_merged` 都 vmap+jit
- **内存分块**：`_extract_per_batch_best` 对 B×N 节点做分块 bounds-check 避免 pins_dx/pdy 常驻显存 OOM
- **候选去重与合法优先**：rollout best leaf 合法优先挑选 + extraction 统一 bounds 过滤，保证 top-K 忠实于真实 HPWL
