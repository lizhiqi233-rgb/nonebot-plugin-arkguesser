"""
题库管理器
负责管理干员抽取的星级范围设置
"""

import re
import json
from typing import List, Tuple, Dict, Any
import csv
from pathlib import Path
from nonebot import require

# 加载 localstore 插件
require("nonebot_plugin_localstore")
import nonebot_plugin_localstore as store

class PoolManager:
    """题库管理器"""
    
    def __init__(self):
        self.data_file = store.get_plugin_data_file("pool_settings.json")
        # 动态星级数量统计缓存，由 characters.csv 加载
        self._rarity_counts: Dict[int, int] = {}
        self._load_rarity_counts()
        self._load_settings()
    
    def _load_settings(self):
        """加载设置文件"""
        try:
            if self.data_file.exists():
                self.settings = json.loads(self.data_file.read_text(encoding='utf-8'))
            else:
                self.settings = {}
        except Exception as e:
            print(f"加载题库设置失败: {e}")
            self.settings = {}
    
    def _save_settings(self):
        """保存设置文件"""
        try:
            self.data_file.write_text(json.dumps(self.settings, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception as e:
            print(f"保存题库设置失败: {e}")

    def _load_rarity_counts(self) -> None:
        """
        从 update_simple.py 生成的 characters.csv 动态统计各星级干员数量。
        若文件不存在或解析失败，不抛异常，保留为空以便使用兜底估算。
        """
        try:
            characters_csv: Path = store.get_plugin_data_file("characters.csv")
            if not characters_csv.exists():
                self._rarity_counts = {}
                return
            counts: Dict[int, int] = {}
            with characters_csv.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rarity_str = row.get("rarity")
                    if rarity_str is None:
                        continue
                    try:
                        rarity = int(rarity_str)
                    except ValueError:
                        continue
                    counts[rarity] = counts.get(rarity, 0) + 1
            self._rarity_counts = counts
        except Exception:
            # 任何异常都不影响主流程，使用兜底估算
            self._rarity_counts = {}
    
    def refresh_rarity_counts(self) -> None:
        """对外暴露的刷新接口：重新从 characters.csv 统计星级数量"""
        self._load_rarity_counts()
    
    def parse_rarity_range(self, range_str: str) -> List[int]:
        """
        解析星级范围字符串
        
        Args:
            range_str: 星级范围字符串，如 "6", "5-6", "4-6", "1-6"
            
        Returns:
            星级列表，如 [6] 或 [5, 6] 或 [4, 5, 6]
            
        Raises:
            ValueError: 范围格式错误或超出有效范围
        """
        range_str = range_str.strip()
        
        # 单个数字
        if range_str.isdigit():
            rarity = int(range_str)
            if 1 <= rarity <= 6:
                return [rarity]
            else:
                raise ValueError(f"星级必须在1-6之间，得到: {rarity}")
        
        # 范围格式 x-y
        range_match = re.match(r'^(\d+)-(\d+)$', range_str)
        if range_match:
            start, end = int(range_match.group(1)), int(range_match.group(2))
            
            if not (1 <= start <= 6 and 1 <= end <= 6):
                raise ValueError(f"星级必须在1-6之间，得到: {start}-{end}")
            
            if start > end:
                raise ValueError(f"起始星级不能大于结束星级: {start}-{end}")
            
            return list(range(start, end + 1))
        
        raise ValueError(f"无效的星级范围格式: {range_str}，支持格式: '6' 或 '4-6'")
    
    def set_pool_range(self, user_id: str, group_id: str, range_str: str) -> Dict[str, Any]:
        """
        设置用户/群组的题库范围
        
        设计理念:
        - 群聊: 所有成员共享一个题库设置，任何成员都可以修改
        - 私聊: 每个用户有独立的个人设置
        - 优先级: 群聊设置 > 个人设置 > 默认设置(6星)
        
        Args:
            user_id: 用户ID
            group_id: 群组ID (私聊时为None)
            range_str: 星级范围字符串
            
        Returns:
            设置结果字典，包含成功状态、星级列表、干员数量等信息
        """
        try:
            rarity_list = self.parse_rarity_range(range_str)
            
            # 构建设置键 (群聊优先)
            key = f"group_{group_id}" if group_id else f"user_{user_id}"
            
            # 保存设置
            self.settings[key] = {
                "rarity_range": rarity_list,
                "range_str": range_str
            }
            self._save_settings()
            
            # 统计对应星级的干员数量
            operator_count = self._count_operators_by_rarity(rarity_list)
            
            return {
                "success": True,
                "rarity_list": rarity_list,
                "range_str": range_str,
                "operator_count": operator_count,
                "scope": "群聊" if group_id else "个人"
            }
            
        except ValueError as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_pool_range(self, user_id: str, group_id: str) -> List[int]:
        """
        获取用户/群组的题库范围
        
        Args:
            user_id: 用户ID
            group_id: 群组ID (私聊时为None)
            
        Returns:
            星级列表，默认为 [6]
        """
        # 群聊设置优先于个人设置
        key = f"group_{group_id}" if group_id else f"user_{user_id}"
        
        if key in self.settings:
            return self.settings[key]["rarity_range"]
        
        # 返回默认设置 (只选6星)
        return [6]
    
    def get_pool_info(self, user_id: str, group_id: str) -> Dict[str, Any]:
        """
        获取当前题库设置信息
        
        Args:
            user_id: 用户ID  
            group_id: 群组ID (私聊时为None)
            
        Returns:
            题库信息字典
        """
        rarity_list = self.get_pool_range(user_id, group_id)
        
        # 构建范围字符串显示
        if len(rarity_list) == 1:
            range_display = str(rarity_list[0])
        else:
            range_display = f"{min(rarity_list)}-{max(rarity_list)}"
        
        # 统计干员数量
        operator_count = self._count_operators_by_rarity(rarity_list)
        
        # 判断设置来源
        group_key = f"group_{group_id}" if group_id else None
        user_key = f"user_{user_id}"
        
        if group_key and group_key in self.settings:
            source = "群聊设置"
        elif user_key in self.settings:
            source = "个人设置"
        else:
            source = "默认设置"
        
        return {
            "rarity_list": rarity_list,
            "range_display": range_display,
            "operator_count": operator_count,
            "source": source
        }
    
    def reset_pool_range(self, user_id: str, group_id: str) -> Dict[str, Any]:
        """
        重置题库范围为默认设置
        
        Args:
            user_id: 用户ID
            group_id: 群组ID (私聊时为None)
            
        Returns:
            重置结果字典
        """
        key = f"group_{group_id}" if group_id else f"user_{user_id}"
        
        if key in self.settings:
            del self.settings[key]
            self._save_settings()
            
        return {
            "success": True,
            "rarity_list": [6],
            "range_str": "6",
            "operator_count": self._count_operators_by_rarity([6]),
            "scope": "群聊" if group_id else "个人"
        }
    
    def _count_operators_by_rarity(self, rarity_list: List[int]) -> int:
        """
        统计指定星级范围内的干员数量
        
        Args:
            rarity_list: 星级列表
            
        Returns:
            干员数量
        """
        # 仅使用动态统计结果；无数据返回0
        if not self._rarity_counts:
            return 0
        return sum(self._rarity_counts.get(r, 0) for r in rarity_list)

# 全局题库管理器实例
pool_manager = PoolManager()
