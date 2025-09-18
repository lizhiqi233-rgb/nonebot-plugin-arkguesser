import random
import secrets
import json
import csv
import difflib
from pypinyin import lazy_pinyin
from pathlib import Path
def _safe_get_data_dir():
    try:
        from nonebot_plugin_localstore import get_plugin_data_dir as _get_dir  # type: ignore
        return _get_dir()
    except Exception:
        return Path.home() / ".local" / "share" / "nonebot2" / "nonebot_plugin_arkguesser"
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from nonebot_plugin_uninfo import Uninfo
from .config import get_plugin_config
from .pool_manager import pool_manager
from .mode_manager import mode_manager
import zipfile

# 初始化随机数生成器，确保每次运行都有不同的随机性
def _init_random():
    """初始化随机数生成器"""
    # 使用当前时间和进程ID作为种子，确保随机性
    import time
    import os
    seed = int(time.time() * 1000000) + os.getpid()
    random.seed(seed)
    
    # 预热随机数生成器
    for _ in range(100):
        random.random()

# 在模块导入时初始化随机数
_init_random()

class OperatorGuesser:
    def __init__(self):
        self.games: Dict[str, Dict] = {}
        # 数据文件路径：使用 localstore 插件数据目录
        self.data_path = _safe_get_data_dir()
        
        # 立绘路径：使用 localstore 插件数据目录
        self.illustrations_path = self.data_path / "illustrations"
        
        # 初始化ZIP文件路径 - 兼容旧资源存档，仍放置于数据目录
        self.illustrations_zip_path = self.data_path / "images" / "illustrations.zip"
        self._zip_cache = {}  # 缓存ZIP文件内容
        
        # 检查数据文件是否存在
        if not self._check_data_files():
            # 数据文件不存在，初始化空数据
            self.operators = []
            self.career_map = {}
            self.camp_map = {}
            self.operator_names = []
            self.pinyin_operators = []
            self.max_attempts = get_plugin_config().arkguesser_max_attempts
            self._data_available = False
        else:
            # 数据文件存在，正常加载
            self.operators = self._load_data()
            self.career_map = self._load_career_map()
            self.camp_map = self._load_camp_map()
            self.max_attempts = get_plugin_config().arkguesser_max_attempts
            self.operator_names = [o["name"] for o in self.operators]  # 预加载干员名称列表
            self.pinyin_operators = [''.join(lazy_pinyin(operator)) for operator in self.operator_names]  # 预加载干员名称拼音列表
            self._data_available = True

    def _load_data(self) -> List[Dict]:
        """从CSV文件加载干员数据"""
        operators = []
        csv_path = self.data_path / "characters.csv"
        
        with open(csv_path, "r", encoding="utf-8-sig") as f:  # 使用utf-8-sig处理BOM
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader, 1):
                # 适配新的字段结构
                operator_name = row["name"]  # CSV中的name字段是干员名称
                operator = {
                    "id": idx,
                    "name": operator_name,
                    "enName": row["id"],  # CSV中的id字段是英文ID
                    "profession": row["career"],  # 职业
                    "subProfession": row["subcareer"],  # 子职业
                    "rarity": int(row["rarity"]),  # 星级
                    "origin": row["birthplace"],  # 出身地
                    "race": row["race"],  # 种族
                    "gender": row["gender"],  # 性别
                    "parentFaction": row["camp"],  # 上级势力
                    "faction": row["subcamp"],  # 下级势力
                    "position": row["position"],  # 部署位置
                    "tags": [row.get("tag1", ""), row.get("tag2", ""), row.get("tag3", ""), row.get("tag4", "")],  # 标签
                    "illustration": self._get_illustration_path(operator_name, int(row["rarity"]), row["career"]),  # 立绘路径
                    # 数值属性
                    "attack": int(row.get("max_atk", 0)),  # 攻击
                    "defense": int(row.get("max_def", 0)),  # 防御
                    "hp": int(row.get("max_hp", 0)),  # 生命值上限
                    "res": int(row.get("max_magic_res", 0)),  # 法抗
                    "interval": self._parse_attack_speed(row.get("attack_speed", "0")),  # 攻击间隔
                    "cost": int(row.get("deploy_cost", 0))  # 初始费用
                }
                # 过滤空标签
                operator["tags"] = [tag for tag in operator["tags"] if tag.strip()]
                
                operators.append(operator)
        
        return operators
    
    def _check_data_files(self) -> bool:
        """检查必要的数据文件是否存在"""
        required_files = [
            "characters.csv",
            "career.json", 
            "camp.json"
        ]
        
        for filename in required_files:
            file_path = self.data_path / filename
            if not file_path.exists():
                print(f"数据文件缺失: {file_path}")
                return False
        
        return True
    
    def _parse_attack_speed(self, speed_str: str) -> float:
        """解析攻击间隔字符串，提取数值部分"""
        try:
            # 移除单位 's' 并转换为浮点数
            if speed_str and isinstance(speed_str, str):
                # 提取数字部分（可能包含小数点）
                import re
                match = re.search(r'(\d+\.?\d*)', speed_str)
                if match:
                    return float(match.group(1))
            return 0.0
        except (ValueError, TypeError):
            return 0.0
    
    def _load_career_map(self) -> Dict:
        """加载职业映射数据"""
        career_path = self.data_path / "career.json"
        with open(career_path, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    
    def _load_camp_map(self) -> Dict:
        """加载势力映射数据"""
        camp_path = self.data_path / "camp.json"
        with open(camp_path, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    
    def _get_illustration_path(self, name: str, rarity: int, career: str, mode: str = "大头") -> str:
        """获取干员立绘路径"""
        # 新的目录结构：稀有度X/职业/半身像/干员名_半身像_精英X.png
        rarity_folder = f"稀有度{rarity}"
        
        # 根据稀有度选择立绘类型
        # 稀有度4及以上使用精英2立绘，稀有度3及以下使用精英1立绘
        if rarity >= 4:
            illustration_type = "半身像_精英2"
        else:
            illustration_type = "半身像_精英1"
        
        # 构建新的文件路径
        # 首先尝试从新的目录结构获取
        new_path = f"{rarity_folder}/{career}/半身像/{name}_{illustration_type}.png"
        
        # 检查新路径是否存在
        if self._check_new_illustration_exists(new_path):
            return new_path
        
        # 如果新路径不存在，回退到旧的ZIP文件方式
        folder = str(rarity) if 1 <= rarity <= 6 else "6"
        webp_path = f"{folder}/{name}.webp"
        png_path = f"{folder}/{name}.png"
        
        # 检查ZIP文件中是否存在WebP文件
        if self._check_zip_file_exists(webp_path):
            return webp_path
        else:
            return png_path
    
    def check_illustration_availability(self, name: str, rarity: int, career: str) -> tuple[bool, str]:
        """
        检查干员立绘是否可用
        
        Returns:
            (是否可用, 提示信息)
        """
        # 检查新的目录结构
        rarity_folder = f"稀有度{rarity}"
        # 根据稀有度选择立绘类型
        if rarity >= 4:
            illustration_type = "半身像_精英2"
        else:
            illustration_type = "半身像_精英1"
        
        new_path = f"{rarity_folder}/{career}/半身像/{name}_{illustration_type}.png"
        if self._check_new_illustration_exists(new_path):
            return True, ""
        
        # 检查ZIP文件
        folder = str(rarity) if 1 <= rarity <= 6 else "6"
        webp_path = f"{folder}/{name}.webp"
        png_path = f"{folder}/{name}.png"
        
        if self._check_zip_file_exists(webp_path) or self._check_zip_file_exists(png_path):
            return True, ""
        
        # 立绘不可用，返回提示信息
        missing_msg = f"⚠️ 立绘资源缺失\n"
        missing_msg += f"干员：{name}\n"
        missing_msg += f"稀有度：{rarity}星\n"
        missing_msg += f"职业：{career}\n"
        missing_msg += f"💡 请使用 /arkstart 更新 来下载立绘资源"
        
        return False, missing_msg

    def _check_new_illustration_exists(self, file_path: str) -> bool:
        """检查新的立绘文件是否存在"""
        full_path = self.illustrations_path / file_path
        return full_path.exists()

    def _check_zip_file_exists(self, file_path: str) -> bool:
        """检查ZIP文件中是否存在指定文件"""
        if not self.illustrations_zip_path.exists():
            return False
            
        try:
            with zipfile.ZipFile(self.illustrations_zip_path, 'r') as zip_file:
                return file_path in zip_file.namelist()
        except Exception as e:
            print(f"检查ZIP文件失败: {e}")
            return False

    def _get_zip_file_content(self, file_path: str) -> Optional[bytes]:
        """从ZIP文件中获取文件内容"""
        if not self.illustrations_zip_path.exists():
            return None
            
        try:
            with zipfile.ZipFile(self.illustrations_zip_path, 'r') as zip_file:
                if file_path in zip_file.namelist():
                    return zip_file.read(file_path)
        except Exception as e:
            print(f"读取ZIP文件失败: {e}")
        return None

    def get_session_id(self, uninfo) -> str:
        return f"{uninfo.scope}_{uninfo.self_id}_{uninfo.scene_path}"

    def is_data_available(self) -> bool:
        """检查数据是否可用"""
        return getattr(self, '_data_available', False)
    
    def reload_data(self) -> bool:
        """重新加载数据文件"""
        try:
            # 重新检查数据文件
            if self._check_data_files():
                # 数据文件存在，重新加载
                self.operators = self._load_data()
                self.career_map = self._load_career_map()
                self.camp_map = self._load_camp_map()
                self.operator_names = [o["name"] for o in self.operators]
                self.pinyin_operators = [''.join(lazy_pinyin(operator)) for operator in self.operator_names]
                self._data_available = True
                print("✅ 数据重新加载成功")
                return True
            else:
                # 数据文件仍然缺失
                self.operators = []
                self.career_map = {}
                self.camp_map = {}
                self.operator_names = []
                self.pinyin_operators = []
                self._data_available = False
                print("❌ 数据文件仍然缺失")
                return False
        except Exception as e:
            print(f"❌ 重新加载数据失败: {e}")
            self._data_available = False
            return False

    def get_game(self, uninfo: Uninfo) -> Optional[Dict]:
        return self.games.get(self.get_session_id(uninfo))

    def start_new_game(self, uninfo: Uninfo) -> Dict:
        """开始新游戏"""
        # 检查数据是否可用
        if not self.is_data_available():
            raise ValueError("数据文件不可用，请先使用 [arkstart 更新] 来下载干员数据")
        
        session_id = self.get_session_id(uninfo)
        
        # 获取当前题库设置
        from .pool_manager import pool_manager
        user_id = str(uninfo.user.id) if uninfo.user else None
        group_id = str(uninfo.group.id) if uninfo.group else None
        pool_info = pool_manager.get_pool_info(user_id, group_id)
        allowed_rarities = pool_info["rarity_list"]
        
        # 获取当前游戏模式设置
        from .mode_manager import mode_manager
        mode_info = mode_manager.get_mode_info(user_id, group_id)
        current_mode = mode_info["mode"]
        
        # 获取连战模式设置
        from .continuous_manager import ContinuousManager
        continuous_manager = ContinuousManager()
        continuous_enabled = continuous_manager.get_continuous_mode(user_id, group_id)
        
        # 从题库中随机选择干员
        available_operators = [o for o in self.operators if o["rarity"] in allowed_rarities]
        if not available_operators:
            raise ValueError("当前题库中没有可用干员")
        
        # 使用更安全的随机选择方法
        if len(available_operators) == 1:
            selected_operator = available_operators[0]
        else:
            # 使用secrets模块进行更安全的随机选择
            selected_index = secrets.randbelow(len(available_operators))
            selected_operator = available_operators[selected_index]
        
        # 检查选中干员的立绘是否可用
        operator_name = selected_operator["name"]
        operator_rarity = selected_operator["rarity"]
        operator_career = selected_operator["profession"]
        
        is_illustration_available, missing_msg = self.check_illustration_availability(
            operator_name, operator_rarity, operator_career
        )
        
        if not is_illustration_available:
            # 立绘不可用，抛出异常
            raise ValueError(f"无法开始游戏：{missing_msg}")
        
        # 创建游戏数据
        game_data = {
            "operator": selected_operator,
            "guesses": [],
            "start_time": datetime.now(),
            "allowed_rarities": allowed_rarities,
            "current_mode": current_mode,
            "continuous_mode": continuous_enabled,  # 保存连战模式状态
            "continuous_count": 0,  # 连战次数计数
            "session_id": session_id,
            "user_id": user_id,  # 保存用户ID用于后续连战模式检查
            "group_id": group_id  # 保存群组ID用于后续连战模式检查
        }
        
        self.games[session_id] = game_data
        return game_data

    def guess(self, uninfo: Uninfo, name: str) -> Tuple[bool, Optional[Dict], Dict]:
        # 检查数据是否可用
        if not self.is_data_available():
            raise ValueError("数据文件不可用，请先使用 [arkstart 更新] 来下载干员数据")
        
        game = self.get_game(uninfo)
        if not game or len(game["guesses"]) >= self.max_attempts:
            raise ValueError("游戏已结束")

        guessed = next((o for o in self.operators if o["name"] == name), None)
        if not guessed:
            return False, None, {}

        game["guesses"].append(guessed)
        current = game["operator"]

        # 势力比较逻辑
        faction_comparison = self._compare_factions(
            guessed.get("parentFaction", ""),
            guessed.get("faction", ""),
            current.get("parentFaction", ""),
            current.get("faction", "")
        )
        
        # 标签比较逻辑
        tags_comparison = self._compare_tags(
            guessed.get("tags", []),
            current.get("tags", [])
        )
        
        # 获取当前游戏模式
        current_mode = game.get("current_mode", "大头")
        
        if current_mode == "兔头":
            # 兔头模式：比较游戏数值属性
            comparison = {
                "attack": self._compare_numeric_value(guessed.get("attack", 0), current.get("attack", 0)),
                "defense": self._compare_numeric_value(guessed.get("defense", 0), current.get("defense", 0)),
                "hp": self._compare_numeric_value(guessed.get("hp", 0), current.get("hp", 0)),
                "res": self._compare_numeric_value(guessed.get("res", 0), current.get("res", 0)),
                "rarity": self._compare_rarity(guessed["rarity"], current["rarity"]),
                "gender": guessed["gender"] == current["gender"],
                "interval": self._compare_numeric_value(guessed.get("interval", 0), current.get("interval", 0)),
                "cost": self._compare_numeric_value(guessed.get("cost", 0), current.get("cost", 0)),
                "tags": tags_comparison
            }
            
            # 检查是否所有属性都正确
            all_correct = (
                comparison["attack"]["correct"] and
                comparison["defense"]["correct"] and
                comparison["hp"]["correct"] and
                comparison["res"]["correct"] and
                comparison["rarity"]["correct"] and
                comparison["gender"] and
                comparison["interval"]["correct"] and
                comparison["cost"]["correct"]
            )
            comparison["all_correct"] = all_correct
        else:
            # 大头模式：比较原有属性
            comparison = {
                "profession": guessed["profession"] == current["profession"],
                "subProfession": guessed["subProfession"] == current["subProfession"],
                "rarity": "higher" if guessed["rarity"] > current["rarity"]
                else "lower" if guessed["rarity"] < current["rarity"]
                else "same",
                "origin": guessed["origin"] == current["origin"],
                "race": guessed["race"] == current["race"],
                "gender": guessed["gender"] == current["gender"],
                "position": guessed["position"] == current["position"],
                "faction": faction_comparison,
                "tags": tags_comparison
            }
        return guessed["name"] == current["name"], guessed, comparison

    def find_similar_operators(self, name: str, n: int = 3) -> List[str]:
        # 检查数据是否可用
        if not self.is_data_available():
            return []
        
        # 使用difflib找到相似的干员名称
        difflib_matches = difflib.get_close_matches(
            name,
            self.operator_names,
            n=n,
            cutoff=0.6  # 相似度阈值（0-1之间）
        )
        # 通过拼音精确匹配读音一样的干员名称
        name_pinyin = ''.join(lazy_pinyin(name))  # 转换输入名称为拼音
        pinyin_matches = [self.operator_names[i] for i, pinyin in enumerate(self.pinyin_operators) if
                          pinyin == name_pinyin]

        all_matches = list(dict.fromkeys(pinyin_matches + difflib_matches))
        return all_matches

    def _compare_tags(self, guessed_tags: List[str], target_tags: List[str]) -> Dict:
        """
        比较标签信息，支持乱序匹配
        
        Args:
            guessed_tags: 猜测干员的标签列表
            target_tags: 目标干员的标签列表
        
        Returns:
            包含匹配结果的字典，包括状态信息
        """
        # 清理空标签
        guessed_tags = [tag.strip() for tag in guessed_tags if tag.strip()]
        target_tags = [tag.strip() for tag in target_tags if tag.strip()]
        
        # 找出匹配的标签
        matched_tags = []
        for tag in guessed_tags:
            if tag in target_tags:
                matched_tags.append(tag)
        
        # 确定匹配状态
        match_count = len(matched_tags)
        total_guessed = len(guessed_tags)
        total_target = len(target_tags)
        
        if match_count == total_guessed and match_count == total_target:
            # 完全匹配：所有标签都一致
            status = "exact_match"
        elif match_count > 0:
            # 部分匹配：有些标签匹配
            status = "partial_match"
        else:
            # 无匹配：没有标签匹配
            status = "no_match"
        
        return {
            "matched_tags": matched_tags,  # 匹配的标签列表
            "total_guessed": total_guessed,  # 猜测干员的标签总数
            "total_target": total_target,  # 目标干员的标签总数
            "match_count": match_count,  # 匹配的标签数量
            "status": status  # 匹配状态
        }

    def _compare_numeric_value(self, guessed_value: int, target_value: int) -> Dict:
        """
        比较数值属性，返回详细的比较信息
        
        Args:
            guessed_value: 猜测的数值
            target_value: 目标数值
        
        Returns:
            包含比较结果的字典
        """
        if target_value == 0:
            # 避免除零错误
            return {
                "correct": guessed_value == target_value,
                "direction": "same" if guessed_value == target_value else "unknown",
                "percentage_diff": 0 if guessed_value == target_value else float('inf'),
                "within_20_percent": guessed_value == target_value
            }
        
        # 计算百分比差异
        percentage_diff = abs(guessed_value - target_value) / target_value * 100
        
        # 判断是否在20%范围内
        within_20_percent = percentage_diff <= 20
        
        # 确定差距方向
        if guessed_value == target_value:
            direction = "same"
        elif guessed_value < target_value:
            direction = "up"  # 答案大于猜测，显示↑
        else:
            direction = "down"  # 答案小于猜测，显示↓
        
        return {
            "correct": guessed_value == target_value,
            "direction": direction,
            "percentage_diff": percentage_diff,
            "within_20_percent": within_20_percent
        }

    def _compare_rarity(self, guessed_rarity: int, target_rarity: int) -> Dict:
        """
        比较星级，返回详细的比较信息
        
        Args:
            guessed_rarity: 猜测的星级
            target_rarity: 目标星级
        
        Returns:
            包含比较结果的字典
        """
        if guessed_rarity == target_rarity:
            return {
                "correct": True,
                "direction": "same",
                "percentage_diff": 0,
                "within_20_percent": True
            }
        elif guessed_rarity > target_rarity:
            return {
                "correct": False,
                "direction": "down",
                "percentage_diff": (guessed_rarity - target_rarity) / target_rarity * 100,
                "within_20_percent": False
            }
        else:
            return {
                "correct": False,
                "direction": "up",
                "percentage_diff": (target_rarity - guessed_rarity) / target_rarity * 100,
                "within_20_percent": False
            }

    def _compare_factions(self, guess_parent: str, guess_faction: str, 
                         target_parent: str, target_faction: str) -> Dict:
        """
        比较势力信息，支持分层势力系统
        
        Args:
            guess_parent: 猜测的上级势力
            guess_faction: 猜测的下级势力
            target_parent: 目标的上级势力
            target_faction: 目标的下级势力
        
        Returns:
            包含比较结果的字典
        """
        # 完全匹配
        if (guess_parent == target_parent and guess_faction == target_faction):
            return {
                "status": "exact_match",
                "parent_match": True,
                "faction_match": True,
                "display_parent": guess_parent,
                "display_faction": guess_faction
            }
        
        # 上级势力匹配
        elif guess_parent == target_parent:
            return {
                "status": "parent_match",
                "parent_match": True,
                "faction_match": False,
                "display_parent": guess_parent,
                "display_faction": guess_faction
            }
        
        # 完全不匹配
        else:
            return {
                "status": "no_match",
                "parent_match": False,
                "faction_match": False,
                "display_parent": guess_parent,
                "display_faction": guess_faction
            }

    def end_game(self, uninfo: Uninfo):
        try:
            self.games.pop(self.get_session_id(uninfo))
        except (AttributeError, KeyError):
            pass
    
    def update_continuous_count(self, uninfo: Uninfo, increment: bool = True) -> int:
        """更新连战计数"""
        session_id = self.get_session_id(uninfo)
        if session_id in self.games:
            if increment:
                self.games[session_id]["continuous_count"] += 1
            return self.games[session_id]["continuous_count"]
        return 0
    
    def get_continuous_count(self, uninfo: Uninfo) -> int:
        """获取当前连战计数"""
        session_id = self.get_session_id(uninfo)
        if session_id in self.games:
            return self.games[session_id].get("continuous_count", 0)
        return 0
    
    def is_continuous_mode(self, uninfo: Uninfo) -> bool:
        """检查当前是否处于连战模式"""
        session_id = self.get_session_id(uninfo)
        if session_id in self.games:
            return self.games[session_id].get("continuous_mode", False)
        return False
    
    def get_random_quality_info(self) -> Dict:
        """获取随机数质量信息，用于调试和验证"""
        import time
        
        # 生成一些随机数样本
        random_samples = [random.random() for _ in range(100)]
        
        # 计算随机数的分布情况
        min_val = min(random_samples)
        max_val = max(random_samples)
        avg_val = sum(random_samples) / len(random_samples)
        
        # 检查是否有重复值（理论上不应该有）
        unique_count = len(set(random_samples))
        
        return {
            "seed_time": time.time(),
            "samples_count": len(random_samples),
            "unique_count": unique_count,
            "min_value": min_val,
            "max_value": max_val,
            "average_value": avg_val,
            "randomness_score": unique_count / len(random_samples),  # 越接近1越好
            "using_secrets": True
        }