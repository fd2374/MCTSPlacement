"""
配置模块 - 管理应用程序配置
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class PlacementConfig:
    """布局算法配置"""
    # 文件路径
    base_path: str = "apte"  # 基础路径，会自动添加 .blocks, .nets, .pl 后缀
    
    # MCTS参数
    num_simulations: int = 100
    seed: int = 0
    batch_size: int = 1
    
    # 输出配置
    output_dir: str = "."
    save_visualization: bool = True
    save_tree: bool = True
    
    # 算法参数
    gumbel_scale: float = 1.0
    
    def validate(self) -> None:
        """验证配置参数"""
        if self.num_simulations <= 0:
            raise ValueError("模拟次数必须大于0")
        if self.batch_size <= 0:
            raise ValueError("批处理大小必须大于0")
        if self.gumbel_scale <= 0:
            raise ValueError("Gumbel缩放因子必须大于0")
