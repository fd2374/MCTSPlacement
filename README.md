# MCTS布局算法重构版本

这是一个基于MCTS（Monte Carlo Tree Search）的VLSI布局算法，使用序列对表示和JAX进行高效计算。

## 项目结构

重构后的代码被组织为以下模块：

### 核心模块

- **`data_loader.py`** - 数据加载模块
  - `BookshelfData`: 存储Bookshelf格式数据的数据类
  - `BookshelfLoader`: 解析.blocks、.nets、.pl文件的加载器

- **`placement_state.py`** - 布局状态管理
  - `PlacementState`: 表示布局状态的命名元组
  - `StateManager`: 管理状态转换和验证

- **`sequence_pair.py`** - 序列对算法
  - `SequencePairSolver`: 将序列对转换为坐标位置

- **`hpwl_calculator.py`** - HPWL计算
  - `HPWLCalculator`: 计算半周长线长
  - `calculate_hpwl()`: 计算给定布局的HPWL值

- **`mcts_placer.py`** - MCTS布局算法
  - `MCTSPlacer`: 实现MCTS搜索和策略

- **`visualizer.py`** - 可视化模块
  - `PlacementVisualizer`: 生成布局图和搜索树图

- **`config.py`** - 配置管理
  - `PlacementConfig`: 算法参数配置

- **`main.py`** - 主程序
  - `PlacementRunner`: 协调整个算法流程

## 主要改进

### 1. 模块化设计
- 将原始的单一大文件拆分为多个功能模块
- 每个模块职责单一，便于维护和测试

### 2. 改进的命名规范
- 使用更清晰的类名和函数名
- 遵循Python命名约定

### 3. 性能优化
- 预计算常用值
- 减少重复代码
- 优化JAX函数

### 4. 类型注解和文档
- 添加完整的类型注解
- 详细的文档字符串
- 清晰的参数说明

### 5. 配置管理
- 统一的配置类
- 参数验证
- 灵活的命令行接口

### 6. 增强的可视化
- 搜索树节点显示状态信息（s1, s2, orientations）
- 英文标签
- 更清晰的图形表示

## 使用方法

```bash
# 基本用法
python main.py --blocks apte.blocks --nets apte.nets --pl apte.pl

# 自定义参数
python main.py --sims 200 --seed 42 --output results/

# 禁用某些功能
python main.py --no-tree --no-viz
```

## 配置参数

- `--blocks`: .blocks文件路径
- `--nets`: .nets文件路径  
- `--pl`: .pl文件路径
- `--sims`: MCTS模拟次数
- `--seed`: 随机种子
- `--batch`: 批处理大小
- `--output`: 输出目录
- `--gumbel-scale`: Gumbel缩放因子
- `--no-tree`: 不保存搜索树图
- `--no-viz`: 不保存可视化

## 依赖项

```bash
pip install jax jaxlib mctx numpy matplotlib pygraphviz
```

## 输出文件

- `search_tree.png`: MCTS搜索树可视化
- `placement_result.json`: 布局结果统计

## 技术特点

- **JAX加速**: 使用JAX进行高效数值计算
- **MCTS搜索**: 基于Gumbel-MuZero的强化学习
- **序列对表示**: 高效的布局表示方法
- **HPWL优化**: 最小化半周长线长
- **模块化设计**: 易于扩展和维护
