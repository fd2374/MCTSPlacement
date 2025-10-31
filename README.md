# MCTS布局算法重构版本

这是一个基于MCTS（Monte Carlo Tree Search）的VLSI布局算法，使用序列对表示和JAX进行高效计算。

## 项目结构

重构后的代码被组织为以下模块：

### 核心模块

- **`data_loader.py`** - 数据加载模块
  - `BookshelfData`: 存储Bookshelf格式数据的数据类
  - `BookshelfLoader`: 解析.blocks、.nets、.pl文件的加载器
  - `load_bookshelf_from_base_path()`: 从基础路径自动加载三个文件

- **`placement_state.py`** - 布局状态管理
  - `PlacementState`: 表示布局状态的命名元组
  - `StateManager`: 管理状态转换和验证

- **`sequence_pair.py`** - 序列对算法
  - `SequencePairSolver`: 将序列对转换为坐标位置

- **`placement_solver.py`** - 布局求解器（新增）
  - `PlacementSolver`: 统一处理状态到坐标转换和HPWL计算
  - `compute_final_positions()`: 从状态计算最终坐标
  - `compute_hpwl()`: 计算HPWL值
  - 使用`functools.partial`预绑定bench数据，提供简洁的API

- **`mcts_placer.py`** - MCTS布局算法
  - `MCTSPlacer`: 实现MCTS搜索和策略
  - 简化构造函数，直接使用bench对象
  - 委托布局计算给PlacementSolver

- **`visualizer.py`** - 可视化模块
  - `PlacementVisualizer`: 生成布局图和搜索树图
  - 支持网络连接可视化（重心连接模式）

- **`config.py`** - 配置管理
  - `PlacementConfig`: 算法参数配置

- **`main.py`** - 主程序
  - `PlacementRunner`: 协调整个算法流程

## 主要改进

### 1. 模块化设计
- 将原始的单一大文件拆分为多个功能模块
- 每个模块职责单一，便于维护和测试
- 新增`placement_solver.py`统一处理布局计算

### 2. 简化的API设计
- **统一的数据输入**：只需指定`--base-path`，自动加载三个文件
- **简化的构造函数**：MCTSPlacer和PlacementSolver只需传入bench对象
- **functools.partial优化**：预绑定bench数据，提供更简洁的调用接口

### 3. 性能优化
- 消除数据重复存储，直接使用bench对象
- 使用`functools.partial`预绑定数据，减少重复转换
- 优化JAX函数，减少内存占用

### 4. 代码清理
- 删除冗余的预计算和存储代码
- 职责分离：MCTSPlacer专注算法，PlacementSolver处理布局
- 消除数据重复，单一数据源

### 5. 类型注解和文档
- 添加完整的类型注解
- 详细的文档字符串
- 清晰的参数说明

### 6. 配置管理
- 统一的配置类
- 参数验证
- 灵活的命令行接口

### 7. 增强的可视化
- 搜索树节点显示状态信息（s1, s2, orientations）
- 英文标签
- 网络连接可视化（重心连接模式）
- 更清晰的图形表示

## 使用方法

### 新的简化方式（推荐）
```bash
# 基本用法 - 只需要指定基础路径
python main.py --base-path "data/apte"

# 自定义参数
python main.py --base-path "data/apte" --sims 200 --seed 42 --output results/

# 禁用某些功能
python main.py --base-path "data/apte" --no-tree --no-viz
```

# 自定义参数
python main.py --sims 200 --seed 42 --output results/

# 禁用某些功能
python main.py --no-tree --no-viz
```

## 配置参数

### 新的简化参数
- `--base-path`: 基础路径（会自动添加.blocks, .nets, .pl后缀）

### 通用参数
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

- `search_tree.png`: MCTS搜索树可视化（包含状态信息）
- `best_placement.png`: 最佳布局结果可视化
- 控制台输出：初始HPWL、最终HPWL、序列对信息

## 技术特点

- **JAX加速**: 使用JAX进行高效数值计算
- **MCTS搜索**: 基于Gumbel-MuZero的强化学习
- **序列对表示**: 高效的布局表示方法
- **HPWL优化**: 最小化半周长线长
- **模块化设计**: 易于扩展和维护
- **functools.partial**: 预绑定数据，提供简洁API
- **职责分离**: 清晰的模块边界和单一职责

### 性能提升
- **内存优化**: 消除数据重复存储
- **计算优化**: 预绑定减少重复转换
- **代码简化**: 减少维护成本
