"""
配置模块 - 管理应用程序配置

支持从YAML文件加载配置，命令行参数可覆盖YAML中的值。
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional

import yaml


@dataclass
class PlacementConfig:
    """布局算法配置（所有参数统一管理）"""
    # 文件路径
    base_path: str = "./data/apte"
    
    # MCTS参数
    num_simulations: int = 100
    seed: int = 0
    batch_size: int = 1
    gumbel_scale: float = 0.1

    # Interposer边界（None = 从terminal自动计算）
    boundary_width: Optional[float] = None
    boundary_height: Optional[float] = None
    
    # MCTS rollout leaf 并行数（每个 MCTS 节点 expansion 时 vmap rollout 的 leaves 数）
    rollout_leaves: int = 128

    # 后处理优化参数
    search_points: int = 20
    # annealing_phases: 退火总 phase 数（注意：新语义是“总数”，会按 n_runs 向下取整拆分）
    #   旧语义：annealing_phases 是每个 reheat 段的 phase 数，总 = annealing_phases × n_runs
    #   新语义：annealing_phases 是所有 reheat 段加起来的总 phase 数，per_run = annealing_phases // n_runs (floor)
    annealing_phases: int = 5
    # 退火 reheat 次数：n_runs=1 = 无 reheat，纯 baseline 单段；n_runs>=2 = 完整跑 N 段
    n_runs: int = 1
    # reheat 段起始温度系数：第 2 段及以后的 cur_hot = reheat_factor * initial_step
    #   1.0 = 完整 reheat；< 1.0 = 跳过 best 处的大步长，直接进中温
    reheat_factor: float = 0.9

    # 输出配置
    output_dir: str = "."
    save_visualization: bool = True
    save_tree: bool = True
    save_gif: bool = False

    # MCTS top-k 候选缓存路径（存在则 load 跳过 MCTS, 否则 dump 后跑 MCTS）
    mcts_cache_path: Optional[str] = None
    
    # YAML键名 -> dataclass字段名 映射
    _KEY_MAP = {
        'sims': 'num_simulations',
        'batch': 'batch_size',
        'width': 'boundary_height',
        'height': 'boundary_width',
        'output': 'output_dir',
        'gif': 'save_gif',
    }
    
    def validate(self) -> None:
        """验证配置参数"""
        if self.num_simulations <= 0:
            raise ValueError("模拟次数必须大于0")
        if self.batch_size <= 0:
            raise ValueError("批处理大小必须大于0")
        if self.gumbel_scale <= 0:
            raise ValueError("Gumbel缩放因子必须大于0")
    
    @classmethod
    def from_yaml(cls, path: str) -> 'PlacementConfig':
        """从YAML文件加载配置"""
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        return cls._from_dict(data)
    
    @classmethod
    def _from_dict(cls, data: dict) -> 'PlacementConfig':
        """从字典创建配置"""
        valid_fields = {f.name for f in fields(cls)}
        kwargs = {}
        no_tree = None
        no_viz = None
        
        for k, v in data.items():
            mapped = cls._KEY_MAP.get(k, k)
            if mapped == 'no_tree':
                no_tree = v
            elif mapped == 'no_viz':
                no_viz = v
            elif mapped in valid_fields:
                kwargs[mapped] = v
        
        config = cls(**kwargs)
        if no_tree is not None:
            config.save_tree = not no_tree
        if no_viz is not None:
            config.save_visualization = not no_viz
        return config
    
    def merge_cli(self, cli_args: dict) -> None:
        """用命令行参数覆盖（只覆盖用户显式指定的非None参数）"""
        for k, v in cli_args.items():
            if v is None:
                continue
            mapped = self._KEY_MAP.get(k, k)
            if mapped == 'no_tree':
                if v:
                    self.save_tree = False
            elif mapped == 'no_viz':
                if v:
                    self.save_visualization = False
            elif hasattr(self, mapped):
                setattr(self, mapped, v)
    
    def print_config(self) -> None:
        """打印当前配置"""
        print("=" * 50)
        print("当前配置")
        print("=" * 50)
        for f in fields(self):
            print(f"  {f.name}: {getattr(self, f.name)}")
        print("=" * 50)