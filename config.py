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
    gumbel_scale: float = 1.0
    
    # Interposer边界（None = 从terminal自动计算）
    boundary_width: Optional[float] = None
    boundary_height: Optional[float] = None
    
    # 后处理优化参数
    initial_step: float = 10.0
    final_step: float = 1.0
    initial_search_points: int = 20
    final_search_points: int = 5
    annealing_phases: int = 5
    
    # 输出配置
    output_dir: str = "."
    save_visualization: bool = True
    save_tree: bool = True
    
    # YAML键名 -> dataclass字段名 映射
    _KEY_MAP = {
        'sims': 'num_simulations',
        'batch': 'batch_size',
        'width': 'boundary_width',
        'height': 'boundary_height',
        'output': 'output_dir',
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