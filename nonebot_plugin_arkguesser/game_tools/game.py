import json
import secrets
import csv
import difflib
from pypinyin import lazy_pinyin
from pathlib import Path

import nonebot_plugin_localstore as store
from typing import Any
from datetime import datetime
from nonebot import logger
from nonebot_plugin_uninfo import Uninfo
from .config import get_plugin_config
from .pool_manager import pool_manager
from .mode_manager import mode_manager
import re

RECENT_DRAWN_FILENAME = "recent_drawn.json"


class OperatorGuesser:
    def __init__(self):
        self.games: dict[str, dict[str, Any]] = {}
        # 最近已出干员记录，key 为 context_key（群组/用户），value 为干员名称列表（最近的在前面）
        self.recent_drawn: dict[str, list[str]] = {}
        # 数据文件路径：使用 localstore 插件数据目录
        self.data_path = store.get_plugin_data_dir()
        
        # 检查数据文件是否存在
        if not self._check_data_files():
            # 数据文件不存在，初始化空数据
            self.operators = []
            self.operator_names = []
            self.pinyin_operators = []
            self.alias_to_operator: dict[str, str] = {}
            self.max_attempts = get_plugin_config().max_attempts
            self._data_available = False
        else:
            # 数据文件存在，正常加载
            self.operators = self._load_data()
            self.max_attempts = get_plugin_config().max_attempts
            self.operator_names = [o["name"] for o in self.operators]  # 预加载干员名称列表
            self.pinyin_operators = [''.join(lazy_pinyin(operator)) for operator in self.operator_names]  # 预加载干员名称拼音列表
            self.alias_to_operator = self._load_aliases()
            self._data_available = True

        self._load_recent_drawn()

    def _load_recent_drawn(self) -> None:
        """从 JSON 恢复「最近已出干员」记录，重启后不丢失。"""
        path = self.data_path / RECENT_DRAWN_FILENAME
        if not path.exists():
            self.recent_drawn = {}
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as e:
            logger.warning("[arkguesser] 读取 {} 失败: {}", RECENT_DRAWN_FILENAME, e)
            self.recent_drawn = {}
            return
        if not isinstance(raw, dict):
            self.recent_drawn = {}
            return
        out: dict[str, list[str]] = {}
        for k, v in raw.items():
            if not isinstance(k, str) or not k.strip():
                continue
            if not isinstance(v, list):
                continue
            out[k] = [x.strip() for x in v if isinstance(x, str) and x.strip()]
        self.recent_drawn = out
        if self._apply_recent_drawn_sanitize():
            self._save_recent_drawn()

    def _save_recent_drawn(self) -> None:
        path = self.data_path / RECENT_DRAWN_FILENAME
        try:
            self.data_path.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.recent_drawn, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("[arkguesser] 写入 {} 失败: {}", RECENT_DRAWN_FILENAME, e)

    def _apply_recent_drawn_sanitize(self) -> bool:
        """
        按当前题库干员名与 recent_exclude_count 裁剪/清理内存中的记录。
        若数据有变化返回 True。
        """
        n = get_plugin_config().recent_exclude_count
        valid = set(self.operator_names) if self.operator_names else set()
        new_map: dict[str, list[str]] = {}
        changed = False
        for k, names in list(self.recent_drawn.items()):
            if not isinstance(names, list):
                changed = True
                continue
            if n <= 0:
                if names:
                    changed = True
                continue
            filtered = [x for x in names if x in valid][:n]
            if filtered:
                new_map[k] = filtered
            if names != filtered:
                changed = True
        if new_map != self.recent_drawn:
            self.recent_drawn = new_map
            changed = True
        return changed

    def _load_aliases(self) -> dict[str, str]:
        """
        从 aliases.csv 加载别称映射：alias -> operator_name
        aliases.csv 格式：operator_name, aliases（aliases 单元格内可包含多个别称）
        """
        aliases_path = self.data_path / "aliases.csv"
        if not aliases_path.exists():
            return {}

        mapping: dict[str, str] = {}
        try:
            with open(aliases_path, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames or "operator_name" not in reader.fieldnames:
                    # 兼容无表头情况
                    f.seek(0)
                    raw = csv.reader(f)
                    for row in raw:
                        if not row:
                            continue
                        if row[0].strip() == "operator_name":
                            continue
                        op_name = (row[0].strip() if len(row) >= 1 else "")
                        cell = (row[1].strip() if len(row) >= 2 else "")
                        self._merge_alias_cell(mapping, op_name, cell)
                    return mapping

                for row in reader:
                    op_name = (row.get("operator_name") or "").strip()
                    cell = (row.get("aliases") or "").strip()
                    self._merge_alias_cell(mapping, op_name, cell)
        except Exception:
            return {}

        return mapping

    def reload_aliases(self) -> None:
        """仅刷新别称映射，不影响正在进行的游戏状态。"""
        if not self.is_data_available():
            self.alias_to_operator = {}
            return
        self.alias_to_operator = self._load_aliases()

    def _merge_alias_cell(self, mapping: dict[str, str], operator_name: str, cell: str) -> None:
        if not operator_name or operator_name not in getattr(self, "operator_names", []):
            # 仅接受数据库里真实存在的干员名，防止脏数据污染映射
            return
        if not cell:
            return
        parts = [p.strip() for p in re.split(r"[|,，;；/\\\s]+", cell) if p.strip()]
        for alias in parts:
            if alias and alias != operator_name:
                mapping.setdefault(alias, operator_name)

    def resolve_operator_name(self, name: str) -> str | None:
        """将用户输入解析为标准干员名（支持别称）。"""
        name = (name or "").strip()
        if not name:
            return None
        if name in self.operator_names:
            return name
        mapped = self.alias_to_operator.get(name)
        if mapped:
            return mapped
        return None

    def _load_data(self) -> list[dict[str, Any]]:
        """从CSV文件加载干员数据"""
        operators = []
        csv_path = self.data_path / "characters.csv"
        
        with open(csv_path, "r", encoding="utf-8-sig") as f:  # 使用utf-8-sig处理BOM
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader, 1):
                # 适配新的字段结构
                operator_name = row["name"]  # CSV中的name字段是干员名称
                char_id = row["id"]  # CSV中的id字段是英文ID（如 char_XXXX）
                rarity_int = int(row["rarity"])
                operator = {
                    "id": idx,
                    "name": operator_name,
                    "enName": char_id,  # CSV中的id字段是英文ID
                    "profession": row["career"],  # 职业
                    "subProfession": row["subcareer"],  # 子职业
                    "rarity": rarity_int,  # 星级
                    "origin": row["birthplace"],  # 出身地
                    "race": row["race"],  # 种族
                    "gender": row["gender"],  # 性别
                    "parentFaction": row["camp"],  # 上级势力
                    "faction": row["subcamp"],  # 下级势力
                    "position": row["position"],  # 部署位置
                    "tags": [row.get("tag1", ""), row.get("tag2", ""), row.get("tag3", ""), row.get("tag4", "")],  # 标签
                    # 立绘：优先使用 PRTS 静态资源 URL（不依赖本地下载）
                    "illustration": self._get_illustration_url(char_id, rarity_int),
                    "charArt": self._get_char_art_url(char_id, rarity_int),
                    "charArtE2": self._get_char_art_e2_url(char_id),
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
    
    def _get_illustration_url(self, char_id: str, rarity: int) -> str:
        """根据 char_id 生成 PRTS 静态资源立绘 URL（半身像）"""
        # 稀有度4及以上使用精英2立绘，稀有度3及以下使用精英1立绘
        level = 2 if rarity >= 4 else 1
        return f"https://torappu.prts.wiki/assets/char_portrait/{char_id}_{level}.png"

    def _get_char_art_url(self, char_id: str, rarity: int) -> str:
        """整身立绘 URL（char_arts），用于 new_base 等全幅底图"""
        level = 2 if rarity >= 4 else 1
        return f"https://torappu.prts.wiki/assets/char_arts/{char_id}_{level}.png"

    def _get_char_art_e2_url(self, char_id: str) -> str:
        """精二 char_arts URL，用于 new_base 左侧立绘"""
        return f"https://torappu.prts.wiki/assets/char_arts/{char_id}_2.png"

    def check_illustration_availability(self, char_id: str, rarity: int) -> tuple[bool, str]:
        """
        检查干员立绘是否可用
        
        Returns:
            (是否可用, 提示信息)
        """
        # 方案1：直接使用网络立绘，不依赖本地缓存。
        # 这里不做网络连通性探测，避免更新/开始游戏时额外阻塞。
        if char_id:
            return True, ""
        return False, "⚠️ 干员缺少 char_id，无法生成立绘 URL"

    def get_session_id(self, uninfo) -> str:
        return f"{uninfo.scope}_{uninfo.self_id}_{uninfo.scene_path}"

    def _update_recent_drawn(self, context_key: str, operator_name: str, max_count: int) -> None:
        """将本次抽中的干员加入最近记录，并保持列表长度不超过 max_count"""
        if max_count <= 0:
            return
        # 新建列表，避免对 dict 内 list 原地 insert 再 slice 时与其它引用纠缠
        prev = self.recent_drawn.get(context_key, [])
        self.recent_drawn[context_key] = [operator_name] + prev[: max_count - 1]
        self._save_recent_drawn()

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
                self.operator_names = [o["name"] for o in self.operators]
                self.pinyin_operators = [''.join(lazy_pinyin(operator)) for operator in self.operator_names]
                self.alias_to_operator = self._load_aliases()
                self._data_available = True
                print("✅ 数据重新加载成功")
                if self._apply_recent_drawn_sanitize():
                    self._save_recent_drawn()
                return True
            else:
                # 数据文件仍然缺失
                self.operators = []
                self.operator_names = []
                self.pinyin_operators = []
                self.alias_to_operator = {}
                self._data_available = False
                print("❌ 数据文件仍然缺失")
                if self._apply_recent_drawn_sanitize():
                    self._save_recent_drawn()
                return False
        except Exception as e:
            print(f"❌ 重新加载数据失败: {e}")
            self._data_available = False
            if self._apply_recent_drawn_sanitize():
                self._save_recent_drawn()
            return False

    def get_game(self, uninfo: Uninfo) -> dict[str, Any] | None:
        return self.games.get(self.get_session_id(uninfo))

    def start_new_game(self, uninfo: Uninfo) -> dict[str, Any]:
        """开始新游戏"""
        # 检查数据是否可用
        if not self.is_data_available():
            raise ValueError("数据文件不可用，请先使用 [arkstart 更新] 来下载干员数据")
        
        session_id = self.get_session_id(uninfo)
        # 与 get_game 使用同一套会话标识。若仅用 group_id/user_id，Alconna 与 on_message
        # 注入的 uninfo 字段可能不一致，但 scene_path 仍相同，会导致「有游戏进度但最近记录键错位」、短期内重复抽中。
        recent_context_key = session_id

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

        # 最近已出干员排除：保证最近 N 名不重复（列表顺序：下标 0 为最近一次开局抽中的干员）
        exclude_count = get_plugin_config().recent_exclude_count
        if exclude_count <= 0:
            recent_ordered: list[str] = []
            recent_names: set[str] = set()
        else:
            recent_ordered = list(self.recent_drawn.get(recent_context_key, []))
            recent_names = set(recent_ordered)
        pool_for_draw = [o for o in available_operators if o["name"] not in recent_names]
        used_full_pool_fallback = False
        if not pool_for_draw:
            pool_for_draw = available_operators  # 排除后无可用干员时回退到全池
            used_full_pool_fallback = True
            logger.warning(
                "[arkguesser] 排除近N次后星级池为空，已回退为未排除的星级池；"
                "近N次列表={} | N={} | 星级={}",
                recent_ordered,
                exclude_count,
                allowed_rarities,
            )

        # 使用更安全的随机选择方法
        selected_index: int
        if len(pool_for_draw) == 1:
            selected_index = 0
            selected_operator = pool_for_draw[0]
        else:
            selected_index = secrets.randbelow(len(pool_for_draw))
            selected_operator = pool_for_draw[selected_index]

        logger.debug(
            "[arkguesser] 开局随机 | 星级题库={} | 星级筛后人数={} | 排除窗口N={} | "
            "本局排除列表(近N次,新→旧)={} | 实际随机池人数={} | 是否因池空回退全星级池={} | "
            "randbelow上界(不含)={} | 选中下标={} | 选中干员={}",
            allowed_rarities,
            len(available_operators),
            exclude_count,
            recent_ordered,
            len(pool_for_draw),
            used_full_pool_fallback,
            len(pool_for_draw),
            selected_index,
            selected_operator["name"],
        )

        # 更新最近已出记录
        self._update_recent_drawn(recent_context_key, selected_operator["name"], exclude_count)
        
        # 检查选中干员的立绘是否可用
        operator_name = selected_operator["name"]
        operator_rarity = selected_operator["rarity"]
        operator_char_id = selected_operator.get("enName", "")
        
        is_illustration_available, missing_msg = self.check_illustration_availability(
            operator_char_id, operator_rarity
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

    def guess(self, uninfo: Uninfo, name: str) -> tuple[bool, dict[str, Any] | None, dict[str, Any]]:
        # 检查数据是否可用
        if not self.is_data_available():
            raise ValueError("数据文件不可用，请先使用 [arkstart 更新] 来下载干员数据")
        
        game = self.get_game(uninfo)
        if not game or len(game["guesses"]) >= self.max_attempts:
            raise ValueError("游戏已结束")

        resolved = self.resolve_operator_name(name)
        guessed_name = resolved or name
        guessed = next((o for o in self.operators if o["name"] == guessed_name), None)
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
        
        # 仅保留大头模式比较逻辑
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

    def find_similar_operators(self, name: str, n: int = 3) -> list[str]:
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

    def _compare_tags(self, guessed_tags: list[str], target_tags: list[str]) -> dict[str, Any]:
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

    def _compare_numeric_value(self, guessed_value: int, target_value: int) -> dict[str, Any]:
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

    def _compare_rarity(self, guessed_rarity: int, target_rarity: int) -> dict[str, Any]:
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
                         target_parent: str, target_faction: str) -> dict[str, Any]:
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
        # 答案无势力数据：不参与对错配色（模板 row-neutral，「无」不高亮）
        if target_parent == "无" and target_faction == "无":
            return {
                "status": "neutral",
                "parent_match": False,
                "faction_match": False,
                "display_parent": guess_parent,
                "display_faction": guess_faction,
            }

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