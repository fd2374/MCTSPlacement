"""
数据加载模块 - 处理Bookshelf格式文件的解析

该模块提供了用于加载和解析Bookshelf格式布局文件的类和方法。
Bookshelf是一种常用的VLSI布局基准测试格式，包含.blocks、.nets和.pl文件。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np


def _strip_comments(line: str) -> str:
    """移除行中的注释"""
    return line.split('#', 1)[0].strip()


@dataclass
class BookshelfData:
    """Bookshelf格式数据容器
    
    存储从Bookshelf格式文件加载的布局数据，包括模块信息、网络连接和固定位置。
    
    Attributes:
        names: 所有节点名称列表
        widths: 模块宽度数组，形状为(G,)，其中G是总节点数
        heights: 模块高度数组，形状为(G,)
        is_terminal: 终端节点标志数组，形状为(G,)，1表示终端节点
        x_fixed: 固定X坐标数组，形状为(G,)，NaN表示未固定
        y_fixed: 固定Y坐标数组，形状为(G,)，NaN表示未固定
        nets_ptr: 网络指针数组，形状为(M+1,)，用于索引网络中的引脚
        pins_nodes: 引脚节点索引数组，形状为(P,)，P是总引脚数
        pins_dx: 引脚X方向偏移数组，形状为(P,)，以百分比表示
        pins_dy: 引脚Y方向偏移数组，形状为(P,)，以百分比表示
    """
    names: List[str]                 # 所有节点名称
    widths: np.ndarray              # 宽度数组 (G,)
    heights: np.ndarray             # 高度数组 (G,)
    is_terminal: np.ndarray         # 是否为终端节点 (G,)
    x_fixed: np.ndarray             # 固定X坐标 (G,)
    y_fixed: np.ndarray             # 固定Y坐标 (G,)
    nets_ptr: np.ndarray            # 网络指针 (M+1,)
    pins_nodes: np.ndarray          # 引脚节点索引 (P,)
    pins_dx: np.ndarray             # 引脚X偏移 (P,)
    pins_dy: np.ndarray             # 引脚Y偏移 (P,)


class BookshelfLoader:
    """Bookshelf文件加载器"""
    
    @staticmethod
    def load_bookshelf(blocks_path: str, nets_path: str, pl_path: Optional[str] = None) -> BookshelfData:
        """加载完整的Bookshelf数据"""
        names, widths, heights, is_term = BookshelfLoader._parse_blocks(blocks_path)
        x_fixed, y_fixed = BookshelfLoader._parse_pl(pl_path, names) if pl_path else (
            np.full(len(names), np.nan), np.full(len(names), np.nan)
        )
        nets_ptr, pins_nodes, pins_dx, pins_dy = BookshelfLoader._parse_nets(nets_path, names)
        
        return BookshelfData(
            names=names,
            widths=np.asarray(widths, dtype=np.float32),
            heights=np.asarray(heights, dtype=np.float32),
            is_terminal=np.asarray(is_term, dtype=np.int32),
            x_fixed=np.asarray(x_fixed, dtype=np.float32),
            y_fixed=np.asarray(y_fixed, dtype=np.float32),
            nets_ptr=np.asarray(nets_ptr, dtype=np.int32),
            pins_nodes=np.asarray(pins_nodes, dtype=np.int32),
            pins_dx=np.asarray(pins_dx, dtype=np.float32),
            pins_dy=np.asarray(pins_dy, dtype=np.float32),
        )
    
    @staticmethod
    def load_bookshelf_from_base_path(base_path: str) -> BookshelfData:
        """从基础路径加载Bookshelf数据
        
        Args:
            base_path: 基础路径，如 "./data/apte"，会自动添加 .blocks, .nets, .pl 后缀
            
        Returns:
            BookshelfData对象
        """
        import os
        
        # 构建文件路径
        blocks_path = f"{base_path}.blocks"
        nets_path = f"{base_path}.nets"
        pl_path = f"{base_path}.pl"
        
        # 检查文件是否存在
        if not os.path.exists(blocks_path):
            raise FileNotFoundError(f"Blocks file not found: {blocks_path}")
        if not os.path.exists(nets_path):
            raise FileNotFoundError(f"Nets file not found: {nets_path}")
        
        # 检查.pl文件是否存在（可选）
        if not os.path.exists(pl_path):
            print(f"Warning: PL file not found: {pl_path}, using default positions")
            pl_path = None
        
        print(f"Loading Bookshelf data from base path: {base_path}")
        print(f"  Blocks: {blocks_path}")
        print(f"  Nets: {nets_path}")
        print(f"  PL: {pl_path if pl_path else 'None (using defaults)'}")
        
        return BookshelfLoader.load_bookshelf(blocks_path, nets_path, pl_path)

    @staticmethod
    def _parse_blocks(path: str) -> Tuple[List[str], List[float], List[float], List[int]]:
        """解析.blocks文件"""
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
                    w = float(parts[1])
                    h = float(parts[2])
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
                w = 0.0
                h = 0.0
                
            names.append(name)
            widths.append(w)
            heights.append(h)
            is_term.append(term)
            
        return names, widths, heights, is_term

    @staticmethod
    def _parse_pl(path: str, names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """解析.pl文件"""
        name2idx = {n: i for i, n in enumerate(names)}
        x = np.full((len(names),), np.nan, dtype=np.float32)
        y = np.full((len(names),), np.nan, dtype=np.float32)
        
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for raw in f:
                s = _strip_comments(raw)
                if not s or 'UCLA' in s:
                    continue
                parts = s.split()
                if len(parts) < 3:
                    continue
                nm = parts[0]
                if nm not in name2idx:
                    continue
                try:
                    xi = float(parts[1])
                    yi = float(parts[2])
                except ValueError:
                    continue
                idx = name2idx[nm]
                x[idx] = xi
                y[idx] = yi
                
        return x, y

    @staticmethod
    def _parse_nets(path: str, names: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """解析.nets文件"""
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
            np.asarray(nets_ptr, dtype=np.int32),
            np.asarray(pins_nodes, dtype=np.int32),
            np.asarray(pins_dx_pct, dtype=np.float32),
            np.asarray(pins_dy_pct, dtype=np.float32),
        )
