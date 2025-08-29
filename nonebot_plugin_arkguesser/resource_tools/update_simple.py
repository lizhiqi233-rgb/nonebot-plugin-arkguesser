#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
明日方舟猜谜游戏数据自动更新程序 - 简化版本
基于 arknights-toolkit 的 main.py 原理
"""

import re
import json
import csv
import asyncio
import sys
from pathlib import Path
def _safe_get_data_dir() -> Path:
    try:
        from nonebot_plugin_localstore import get_plugin_data_dir as _get_dir  # type: ignore
        return _get_dir()
    except Exception:
        return Path.home() / ".local" / "share" / "nonebot2" / "nonebot_plugin_arkguesser"
from typing import Dict, List, Optional

import httpx
from nonebot import logger

# 强制设置系统编码为UTF-8
if sys.platform.startswith('win'):
    import locale
    locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')

# 插件不应修改用户日志配置，移除自定义日志处理器

# 数据文件路径 - 使用 localstore 插件数据目录（根目录）
DATA_DIR = _safe_get_data_dir()
CAMP_FILE = DATA_DIR / "camp.json"
CAREER_FILE = DATA_DIR / "career.json"
CHARACTERS_FILE = DATA_DIR / "characters.csv"

# 维基百科API
WIKI_API = "https://prts.wiki/api.php"

# 正则表达式（基于arknights-toolkit）
ID_PATTERN = re.compile(r"\|干员id=char_([^|]+?)\n\|")
RARITY_PATTERN = re.compile(r"\|稀有度=(\d+?)\n\|")
CHAR_PATTERN = re.compile(r"\|职业=([^|]+?)\n\|")
SUB_CHAR_PATTERN = re.compile(r"\|分支=([^|]+?)\n\|")
RACE_PATTERN = re.compile(r"\|种族=([^|]+?)\n\|")
ORG_PATTERN = re.compile(r"\|所属国家=([^|]+?)\n\|")
ORG_PATTERN1 = re.compile(r"\|所属组织=([^|]+?)\n\|")
ORG_PATTERN2 = re.compile(r"\|所属团队=([^|]+?)\n\|")
ART_PATTERN = re.compile(r"\|画师=([^|]+?)\n\|")
NAME_PATTERN = re.compile(r"\|干员名=([^|]+?)\n\|")
POSITION_PATTERN = re.compile(r"\|位置=([^|]+?)\n\|")
TAG_PATTERN = re.compile(r"\|标签=([^|]+?)\n\|")
JAPANESE_VOICE_PATTERN = re.compile(r"\|日文配音=([^|]+?)\n\|")
OBTAIN_METHOD_PATTERN = re.compile(r"\|获得方式=([^|]+?)\n\|")
ONLINE_TIME_PATTERN = re.compile(r"\|上线时间=([^|}]+?)(?:\n\||\}|\n)")
# 修改部署费用正则表达式，提取最后一个阶段的数字
DEPLOY_COST_PATTERN = re.compile(r"\|部署费用=(?:\d+→)*(\d+)\n\|")
# 添加更多可能的部署费用字段名，都提取最后一个阶段的数字
DEPLOY_COST_PATTERN2 = re.compile(r"\|费用=(?:\d+→)*(\d+)\n\|")
DEPLOY_COST_PATTERN3 = re.compile(r"\|cost=(?:\d+→)*(\d+)\n\|")
DEPLOY_COST_PATTERN4 = re.compile(r"\|部署费用_精英0=(\d+)\n\|")
DEPLOY_COST_PATTERN5 = re.compile(r"\|费用_精英0=(\d+)\n\|")
# 阻挡数正则表达式，支持多阶段格式（如：2→3→3）
BLOCK_COUNT_PATTERN = re.compile(r"\|阻挡数=(?:\d+→)*(\d+)\n\|")
ATTACK_SPEED_PATTERN = re.compile(r"\|攻击速度=([^|]+?)\n\|")

# 多阶段属性正则表达式（用于获取干员的满级数据，按精英阶段优先级）
ELITE0_HP_PATTERN = re.compile(r"\|精英0_满级_生命上限=(\d+?)\n\|")
ELITE0_ATK_PATTERN = re.compile(r"\|精英0_满级_攻击=(\d+?)\n\|")
ELITE0_DEF_PATTERN = re.compile(r"\|精英0_满级_防御=(\d+?)\n\|")
ELITE0_MAGIC_RES_PATTERN = re.compile(r"\|精英0_满级_法术抗性=(\d+?)\n\|")

ELITE1_HP_PATTERN = re.compile(r"\|精英1_满级_生命上限=(\d+?)\n\|")
ELITE1_ATK_PATTERN = re.compile(r"\|精英1_满级_攻击=(\d+?)\n\|")
ELITE1_DEF_PATTERN = re.compile(r"\|精英1_满级_防御=(\d+?)\n\|")
ELITE1_MAGIC_RES_PATTERN = re.compile(r"\|精英1_满级_法术抗性=(\d+?)\n\|")

ELITE2_HP_PATTERN = re.compile(r"\|精英2_满级_生命上限=(\d+?)\n\|")
ELITE2_ATK_PATTERN = re.compile(r"\|精英2_满级_攻击=(\d+?)\n\|")
ELITE2_DEF_PATTERN = re.compile(r"\|精英2_满级_防御=(\d+?)\n\|")
ELITE2_MAGIC_RES_PATTERN = re.compile(r"\|精英2_满级_法术抗性=(\d+?)\n\|")

GENDER_PATTERN = re.compile(r"\|性别=([^|]+?)\n\|")
BIRTHPLACE_PATTERN = re.compile(r"\|出身地=([^|]+?)\n\|")
BIRTHDAY_PATTERN = re.compile(r"\|生日=([^|]+?)\n\|")
HEIGHT_PATTERN = re.compile(r"\|身高=([^|]+?)\n\|")

# 技能相关正则表达式（支持两种格式：传统{{技能模板和新{{技能2模板）
# 技能1相关
SKILL1_NAME_PATTERN = re.compile(r"'''技能1[（(]精英0开放[）)]'''\s*{{技能[^}]*?\|技能名=([^|]+?)\n\|")
SKILL1_TYPE1_PATTERN = re.compile(r"'''技能1[（(]精英0开放[）)]'''\s*{{技能[^}]*?\|技能名=[^|]+?\n\|[^}]*?\|技能类型1=([^|]+?)\n\|")
SKILL1_TYPE2_PATTERN = re.compile(r"'''技能1[（(]精英0开放[）)]'''\s*{{技能[^}]*?\|技能名=[^|]+?\n\|[^}]*?\|技能类型2=([^|]+?)\n\|")
SKILL1_SPEC3_INITIAL_PATTERN = re.compile(r"\|技能专精3初始=(\d+?)\n\|")
SKILL1_SPEC3_COST_PATTERN = re.compile(r"\|技能专精3消耗=(\d+?)\n\|")
SKILL1_SPEC3_DURATION_PATTERN = re.compile(r"\|技能专精3持续=([^|}]+?)(?:\n\||\}|\n)")

# 技能2相关
SKILL2_NAME_PATTERN = re.compile(r"'''技能2[（(]精英1开放[）)]'''\s*{{技能[^}]*?\|技能名=([^|]+?)\n\|")
SKILL2_TYPE1_PATTERN = re.compile(r"'''技能2[（(]精英1开放[）)]'''\s*{{技能[^}]*?\|技能名=[^|]+?\n\|[^}]*?\|技能类型1=([^|]+?)\n\|")
SKILL2_TYPE2_PATTERN = re.compile(r"'''技能2[（(]精英1开放[）)]'''\s*{{技能[^}]*?\|技能名=[^|]+?\n\|[^}]*?\|技能类型2=([^|]+?)\n\|")
SKILL2_SPEC3_INITIAL_PATTERN = re.compile(r"\|技能专精3初始=(\d+?)\n\|")
SKILL2_SPEC3_COST_PATTERN = re.compile(r"\|技能专精3消耗=(\d+?)\n\|")
SKILL2_SPEC3_DURATION_PATTERN = re.compile(r"\|技能专精3持续=([^|}]+?)(?:\n\||\}|\n)")

# 技能3相关
SKILL3_NAME_PATTERN = re.compile(r"'''技能3[（(]精英2开放[）)]'''\s*{{技能[^}]*?\|技能名=([^|]+?)\n\|")
SKILL3_TYPE1_PATTERN = re.compile(r"'''技能3[（(]精英2开放[）)]'''\s*{{技能[^}]*?\|技能名=[^|]+?\n\|[^}]*?\|技能类型1=([^|]+?)\n\|")
SKILL3_TYPE2_PATTERN = re.compile(r"'''技能3[（(]精英2开放[）)]'''\s*{{技能[^}]*?\|技能名=[^|]+?\n\|[^}]*?\|技能类型2=([^|]+?)\n\|")
SKILL3_SPEC3_INITIAL_PATTERN = re.compile(r"\|技能专精3初始=(\d+?)\n\|")
SKILL3_SPEC3_COST_PATTERN = re.compile(r"\|技能专精3消耗=(\d+?)\n\|")
SKILL3_SPEC3_DURATION_PATTERN = re.compile(r"\|技能专精3持续=([^|}]+?)(?:\n\||\}|\n)")

# 新增：支持{{技能2模板的技能专精3相关正则表达式
# 技能1相关（{{技能2模板）
SKILL1_NAME_PATTERN2 = re.compile(r"'''技能1[（(]精英0开放[）)]'''\s*{{技能2[^}]*?\|技能名=([^|]+?)\n\|")
SKILL1_TYPE1_PATTERN2 = re.compile(r"'''技能1[（(]精英0开放[）)]'''\s*{{技能2[^}]*?\|技能名=[^|]+?\n\|[^}]*?\|技能类型1=([^|]+?)\n\|")
SKILL1_TYPE2_PATTERN2 = re.compile(r"'''技能1[（(]精英0开放[）)]'''\s*{{技能2[^}]*?\|技能名=[^|]+?\n\|[^}]*?\|技能类型2=([^|]+?)\n\|")
SKILL1_SPEC3_INITIAL_PATTERN2 = re.compile(r"\|技能专精3初始=(\d+?)\n\|")
SKILL1_SPEC3_COST_PATTERN2 = re.compile(r"\|技能专精3消耗=(\d+?)\n\|")
SKILL1_SPEC3_DURATION_PATTERN2 = re.compile(r"\|技能专精3持续=([^|}]+?)(?:\n\||\}|\n)")

# 技能2相关（{{技能2模板）
SKILL2_NAME_PATTERN2 = re.compile(r"'''技能2[（(]精英1开放[）)]'''\s*{{技能2[^}]*?\|技能名=([^|]+?)\n\|")
SKILL2_TYPE1_PATTERN2 = re.compile(r"'''技能2[（(]精英1开放[）)]'''\s*{{技能2[^}]*?\|技能名=[^|]+?\n\|[^}]*?\|技能类型1=([^|]+?)\n\|")
SKILL2_TYPE2_PATTERN2 = re.compile(r"'''技能2[（(]精英1开放[）)]'''\s*{{技能2[^}]*?\|技能名=[^|]+?\n\|[^}]*?\|技能类型2=([^|]+?)\n\|")
SKILL2_SPEC3_INITIAL_PATTERN2 = re.compile(r"\|技能专精3初始=(\d+?)\n\|")
SKILL2_SPEC3_COST_PATTERN2 = re.compile(r"\|技能专精3消耗=(\d+?)\n\|")
SKILL2_SPEC3_DURATION_PATTERN2 = re.compile(r"\|技能专精3持续=([^|}]+?)(?:\n\||\}|\n)")

# 技能3相关（{{技能2模板）
SKILL3_NAME_PATTERN2 = re.compile(r"'''技能3[（(]精英2开放[）)]'''\s*{{技能2[^}]*?\|技能名=([^|]+?)\n\|")
SKILL3_TYPE1_PATTERN2 = re.compile(r"'''技能3[（(]精英2开放[）)]'''\s*{{技能2[^}]*?\|技能名=[^|]+?\n\|[^}]*?\|技能类型1=([^|]+?)\n\|")
SKILL3_TYPE2_PATTERN2 = re.compile(r"'''技能3[（(]精英2开放[）)]'''\s*{{技能2[^}]*?\|技能名=[^|]+?\n\|[^}]*?\|技能类型2=([^|]+?)\n\|")
SKILL3_SPEC3_INITIAL_PATTERN2 = re.compile(r"\|技能专精3初始=(\d+?)\n\|")
SKILL3_SPEC3_COST_PATTERN2 = re.compile(r"\|技能专精3消耗=(\d+?)\n\|")
SKILL3_SPEC3_DURATION_PATTERN2 = re.compile(r"\|技能专精3持续=([^|}]+?)(?:\n\||\}|\n)")

def extract_skill_spec3_initial(content: str, skill_num: int) -> str:
    """
    提取技能专精3初始值
    
    Args:
        content: 页面内容
        skill_num: 技能编号（1、2或3）
    
    Returns:
        技能专精3初始值，如果未找到则返回"未知"
    """
    field_name = f"|技能专精3初始="
    start_pos = content.find(field_name)
    
    if start_pos == -1:
        return "未知"
    
    start_pos += len(field_name)
    
    # 查找换行符
    end_pos = content.find('\n', start_pos)
    if end_pos == -1:
        end_pos = len(content)
    
    value = content[start_pos:end_pos].strip()
    return value if value else "未知"

def extract_skill_spec3_cost(content: str, skill_num: int) -> str:
    """
    提取技能专精3消耗值
    
    Args:
        content: 页面内容
        skill_num: 技能编号（1、2或3）
    
    Returns:
        技能专精3消耗值，如果未找到则返回"未知"
    """
    field_name = f"|技能专精3消耗="
    start_pos = content.find(field_name)
    
    if start_pos == -1:
        return "未知"
    
    start_pos += len(field_name)
    
    # 查找换行符
    end_pos = content.find('\n', start_pos)
    if end_pos == -1:
        end_pos = len(content)
    
    value = content[start_pos:end_pos].strip()
    return value if value else "未知"

def extract_skill_spec3_duration(content: str, skill_num: int) -> str:
    """
    提取技能专精3持续值
    
    Args:
        content: 页面内容
        skill_num: 技能编号（1、2或3）
    
    Returns:
        技能专精3持续值，如果未找到则返回"未知"
    """
    field_name = f"|技能专精3持续="
    start_pos = content.find(field_name)
    
    if start_pos == -1:
        return "未知"
    
    start_pos += len(field_name)
    
    # 查找换行符或管道符
    end_pos = content.find('\n', start_pos)
    pipe_pos = content.find('|', start_pos)
    
    if end_pos == -1 and pipe_pos == -1:
        end_pos = len(content)
    elif end_pos == -1:
        end_pos = pipe_pos
    elif pipe_pos == -1:
        pass  # 使用end_pos
    else:
        end_pos = min(end_pos, pipe_pos)
    
    value = content[start_pos:end_pos].strip()
    return value if value else "未知"

async def get_operator_info(name: str, client: httpx.AsyncClient) -> Optional[Dict]:
    """获取干员信息（基于arknights-toolkit的逻辑）"""
    try:
        # 使用查询API获取页面内容
        query_url = (
            f"{WIKI_API}?action=query&format=json&prop=revisions"
            f"&titles={name}&rvprop=content&rvslots=main"
        )
        
        response = await client.get(query_url)
        response.raise_for_status()
        data = response.json()
        
        # 检查页面是否存在
        if "query" in data and "pages" in data["query"]:
            pages = data["query"]["pages"]
            
            for page_id, page_info in pages.items():
                if page_id != "-1":  # 页面存在
                    if "revisions" in page_info:
                        revisions = page_info["revisions"]
                        if revisions and "slots" in revisions[0]:
                            content = revisions[0]["slots"]["main"]["*"]
                            
                            # 检查获得方式，过滤掉集成战略专属干员
                            obtain_match = re.search(r"\|获得方式=([^|]+?)\n\|", content)
                            if obtain_match:
                                obtain_method = obtain_match.group(1)
                                # 过滤掉获得方式为"无"的干员（集成战略专属）
                                if obtain_method == "无":
                                    logger.info(f"跳过集成战略专属干员: {name} (获得方式: {obtain_method})")
                                    return None
                            
                            # 提取信息（基于arknights-toolkit的逻辑）
                            char_id = ID_PATTERN.search(content)
                            if not char_id:
                                return None
                            
                            # 星级在获取数据的基础上+1
                            base_rarity = int(RARITY_PATTERN.search(content).group(1)) if RARITY_PATTERN.search(content) else 3
                            rarity = base_rarity + 1
                            career = CHAR_PATTERN.search(content).group(1) if CHAR_PATTERN.search(content) else "未知"
                            subcareer = SUB_CHAR_PATTERN.search(content).group(1) if SUB_CHAR_PATTERN.search(content) else "未知"
                            race = RACE_PATTERN.search(content).group(1) if RACE_PATTERN.search(content) else "未知"
                            
                            # 提取组织信息（按优先级）
                            org = "未知"
                            for pattern in [ORG_PATTERN2, ORG_PATTERN1, ORG_PATTERN]:
                                match = pattern.search(content)
                                if match and match.group(1) != "无信息":
                                    org = match.group(1)
                                    break
                            
                            # 确定主组织和子组织
                            main_camp, sub_camp = determine_camp(org)
                            
                            artist = ART_PATTERN.search(content).group(1) if ART_PATTERN.search(content) else "未知"
                            
                            # 提取新增字段
                            operator_name = NAME_PATTERN.search(content).group(1) if NAME_PATTERN.search(content) else name
                            position = POSITION_PATTERN.search(content).group(1) if POSITION_PATTERN.search(content) else "未知"
                            
                            # 提取标签并分割
                            tags_raw = TAG_PATTERN.search(content).group(1) if TAG_PATTERN.search(content) else ""
                            tags = [tag.strip() for tag in tags_raw.split()] if tags_raw else []
                            tag1 = tags[0] if len(tags) > 0 else ""
                            tag2 = tags[1] if len(tags) > 1 else ""
                            tag3 = tags[2] if len(tags) > 2 else ""
                            tag4 = tags[3] if len(tags) > 3 else ""
                            
                            japanese_voice = JAPANESE_VOICE_PATTERN.search(content).group(1) if JAPANESE_VOICE_PATTERN.search(content) else "未知"
                            obtain_method = OBTAIN_METHOD_PATTERN.search(content).group(1) if OBTAIN_METHOD_PATTERN.search(content) else "未知"
                            online_time = ONLINE_TIME_PATTERN.search(content).group(1) if ONLINE_TIME_PATTERN.search(content) else "未知"
                            
                            # 提取属性信息
                            # 尝试多种部署费用字段
                            deploy_cost = "未知"
                            for pattern in [DEPLOY_COST_PATTERN, DEPLOY_COST_PATTERN2, DEPLOY_COST_PATTERN3, DEPLOY_COST_PATTERN4, DEPLOY_COST_PATTERN5]:
                                match = pattern.search(content)
                                if match:
                                    deploy_cost = match.group(1)
                                    break
                            
                            block_count = BLOCK_COUNT_PATTERN.search(content).group(1) if BLOCK_COUNT_PATTERN.search(content) else "未知"
                            attack_speed = ATTACK_SPEED_PATTERN.search(content).group(1) if ATTACK_SPEED_PATTERN.search(content) else "未知"
                            
                            # 智能提取最高阶段的满级属性
                            # 优先获取精英2属性，如果没有则获取精英1，最后获取精英0
                            elite2_hp = ELITE2_HP_PATTERN.search(content).group(1) if ELITE2_HP_PATTERN.search(content) else None
                            elite2_atk = ELITE2_ATK_PATTERN.search(content).group(1) if ELITE2_ATK_PATTERN.search(content) else None
                            elite2_def = ELITE2_DEF_PATTERN.search(content).group(1) if ELITE2_DEF_PATTERN.search(content) else None
                            elite2_magic_res = ELITE2_MAGIC_RES_PATTERN.search(content).group(1) if ELITE2_MAGIC_RES_PATTERN.search(content) else None
                            
                            elite1_hp = ELITE1_HP_PATTERN.search(content).group(1) if ELITE1_HP_PATTERN.search(content) else None
                            elite1_atk = ELITE1_ATK_PATTERN.search(content).group(1) if ELITE1_ATK_PATTERN.search(content) else None
                            elite1_def = ELITE1_DEF_PATTERN.search(content).group(1) if ELITE1_DEF_PATTERN.search(content) else None
                            elite1_magic_res = ELITE1_MAGIC_RES_PATTERN.search(content).group(1) if ELITE1_MAGIC_RES_PATTERN.search(content) else None
                            
                            elite0_hp = ELITE0_HP_PATTERN.search(content).group(1) if ELITE0_HP_PATTERN.search(content) else None
                            elite0_atk = ELITE0_ATK_PATTERN.search(content).group(1) if ELITE0_ATK_PATTERN.search(content) else None
                            elite0_def = ELITE0_DEF_PATTERN.search(content).group(1) if ELITE0_DEF_PATTERN.search(content) else None
                            elite0_magic_res = ELITE0_MAGIC_RES_PATTERN.search(content).group(1) if ELITE0_MAGIC_RES_PATTERN.search(content) else None
                            
                            # 确定最终使用的属性值（优先使用最高阶段）
                            final_hp = elite2_hp or elite1_hp or elite0_hp or "未知"
                            final_atk = elite2_atk or elite1_atk or elite0_atk or "未知"
                            final_def = elite2_def or elite1_def or elite0_def or "未知"
                            final_magic_res = elite2_magic_res or elite1_magic_res or elite0_magic_res or "未知"
                            
                            # 提取档案信息
                            gender = GENDER_PATTERN.search(content).group(1) if GENDER_PATTERN.search(content) else "未知"
                            birthplace = BIRTHPLACE_PATTERN.search(content).group(1) if BIRTHPLACE_PATTERN.search(content) else "未知"
                            birthday = BIRTHDAY_PATTERN.search(content).group(1) if BIRTHDAY_PATTERN.search(content) else "未知"
                            height = HEIGHT_PATTERN.search(content).group(1) if HEIGHT_PATTERN.search(content) else "未知"
                            # 技能1相关
                            skill1_name = SKILL1_NAME_PATTERN.search(content).group(1) if SKILL1_NAME_PATTERN.search(content) else (SKILL1_NAME_PATTERN2.search(content).group(1) if SKILL1_NAME_PATTERN2.search(content) else "未知")
                            skill1_type1 = SKILL1_TYPE1_PATTERN.search(content).group(1) if SKILL1_TYPE1_PATTERN.search(content) else (SKILL1_TYPE1_PATTERN2.search(content).group(1) if SKILL1_TYPE1_PATTERN2.search(content) else "未知")
                            skill1_type2 = SKILL1_TYPE2_PATTERN.search(content).group(1) if SKILL1_TYPE2_PATTERN.search(content) else (SKILL1_TYPE2_PATTERN2.search(content).group(1) if SKILL1_TYPE2_PATTERN2.search(content) else "未知")
                            # 使用智能提取函数获取技能专精3信息（除了描述）
                            skill1_spec3_initial = extract_skill_spec3_initial(content, 1)
                            skill1_spec3_cost = extract_skill_spec3_cost(content, 1)
                            skill1_spec3_duration = extract_skill_spec3_duration(content, 1)
                            
                            # 技能2相关
                            skill2_name = SKILL2_NAME_PATTERN.search(content).group(1) if SKILL2_NAME_PATTERN.search(content) else (SKILL2_NAME_PATTERN2.search(content).group(1) if SKILL2_NAME_PATTERN2.search(content) else "未知")
                            skill2_type1 = SKILL2_TYPE1_PATTERN.search(content).group(1) if SKILL2_TYPE1_PATTERN.search(content) else (SKILL2_TYPE1_PATTERN2.search(content).group(1) if SKILL2_TYPE1_PATTERN2.search(content) else "未知")
                            skill2_type2 = SKILL2_TYPE2_PATTERN.search(content).group(1) if SKILL2_TYPE2_PATTERN.search(content) else (SKILL2_TYPE2_PATTERN2.search(content).group(1) if SKILL2_TYPE2_PATTERN2.search(content) else "未知")
                            # 使用智能提取函数获取技能专精3信息（除了描述）
                            skill2_spec3_initial = extract_skill_spec3_initial(content, 2)
                            skill2_spec3_cost = extract_skill_spec3_cost(content, 2)
                            skill2_spec3_duration = extract_skill_spec3_duration(content, 2)
                            
                            # 技能3相关
                            skill3_name = SKILL3_NAME_PATTERN.search(content).group(1) if SKILL3_NAME_PATTERN.search(content) else (SKILL3_NAME_PATTERN2.search(content).group(1) if SKILL3_NAME_PATTERN2.search(content) else "未知")
                            skill3_type1 = SKILL3_TYPE1_PATTERN.search(content).group(1) if SKILL3_TYPE1_PATTERN.search(content) else (SKILL3_TYPE1_PATTERN2.search(content).group(1) if SKILL3_TYPE1_PATTERN2.search(content) else "未知")
                            skill3_type2 = SKILL3_TYPE2_PATTERN.search(content).group(1) if SKILL3_TYPE2_PATTERN.search(content) else (SKILL3_TYPE2_PATTERN2.search(content).group(1) if SKILL3_TYPE2_PATTERN2.search(content) else "未知")
                            # 使用智能提取函数获取技能专精3信息（除了描述）
                            skill3_spec3_initial = extract_skill_spec3_initial(content, 3)
                            skill3_spec3_cost = extract_skill_spec3_cost(content, 3)
                            skill3_spec3_duration = extract_skill_spec3_duration(content, 3)
                            
                            return {
                                "id": f"char_{char_id.group(1)}",
                                "name": operator_name,
                                "rarity": rarity,
                                "career": career,
                                "subcareer": subcareer,
                                "camp": main_camp,
                                "subcamp": sub_camp,
                                "race": race,
                                "artist": artist,
                                "position": position,
                                "tag1": tag1,
                                "tag2": tag2,
                                "tag3": tag3,
                                "tag4": tag4,
                                "japanese_voice": japanese_voice,
                                "obtain_method": obtain_method,
                                "online_time": online_time,
                                "deploy_cost": deploy_cost,
                                "block_count": block_count,
                                "attack_speed": attack_speed,
                                "max_hp": final_hp,
                                "max_atk": final_atk,
                                "max_def": final_def,
                                "max_magic_res": final_magic_res,
                                "gender": gender,
                                "birthplace": birthplace,
                                "birthday": birthday,
                                "height": height,
                                "skill1_name": skill1_name,
                                "skill1_type1": skill1_type1,
                                "skill1_type2": skill1_type2,
                                "skill1_spec3_initial": skill1_spec3_initial,
                                "skill1_spec3_cost": skill1_spec3_cost,
                                "skill1_spec3_duration": skill1_spec3_duration,
                                "skill2_name": skill2_name,
                                "skill2_type1": skill2_type1,
                                "skill2_type2": skill2_type2,
                                "skill2_spec3_initial": skill2_spec3_initial,
                                "skill2_spec3_cost": skill2_spec3_cost,
                                "skill2_spec3_duration": skill2_spec3_duration,
                                "skill3_name": skill3_name,
                                "skill3_type1": skill3_type1,
                                "skill3_type2": skill3_type2,
                                "skill3_spec3_initial": skill3_spec3_initial,
                                "skill3_spec3_cost": skill3_spec3_cost,
                                "skill3_spec3_duration": skill3_spec3_duration
                            }
        return None
        
    except Exception as e:
        logger.error(f"获取干员 {name} 信息失败: {e}")
        return None

def determine_camp(org: str) -> tuple[str, str]:
    """确定主组织和子组织（基于arknights-toolkit的逻辑）"""
    if not org or org == "未知":
        return "未知", "未知"
    
    # 特殊处理深池
    if org == "深池":
        return "维多利亚", "深池"
    
    # 根据组织关系确定主组织
    camp_mapping = {
        "罗德岛": "罗德岛",
        "行动组A4": "罗德岛",
        "行动预备组A1": "罗德岛",
        "行动预备组A4": "罗德岛",
        "行动预备组A6": "罗德岛",
        "S.W.E.E.P.": "罗德岛",
        "罗德岛-精英干员": "罗德岛",
        "龙门近卫局": "炎",
        "企鹅物流": "炎",
        "鲤氏侦探事务所": "炎",
        "黑钢国际": "哥伦比亚",
        "莱茵生命": "哥伦比亚",
        "汐斯塔": "哥伦比亚",
        "喀兰贸易": "谢拉格",
        "深海猎人": "阿戈尔",
        "红松骑士团": "卡西米尔",
        "格拉斯哥帮": "维多利亚",
        "塔拉": "维多利亚",
        "乌萨斯学生自治团": "乌萨斯",
        "贾维团伙": "叙拉古"
    }
    
    main_camp = camp_mapping.get(org, org)
    return main_camp, org

async def get_operator_list(client: httpx.AsyncClient) -> List[str]:
    """自动获取最新的干员列表"""
    logger.info("正在获取最新干员列表...")
    
    try:
        # 方法1：通过分类获取干员列表
        category_query = (
            f"{WIKI_API}?action=query&format=json&list=categorymembers"
            "&cmtitle=Category:干员&cmlimit=500&utf8=1"
        )
        
        response = await client.get(category_query)
        response.raise_for_status()
        data = response.json()
        
        if "query" in data and "categorymembers" in data["query"]:
            operators = []
            
            for index, item in enumerate(data["query"]["categorymembers"], 1):
                title = item["title"]
                # 移除"Category:"前缀
                if title.startswith("Category:"):
                    title = title[9:]
                
                # 过滤掉非干员页面
                if any(keyword in title for keyword in [
                    "密录", "剧情一览", "升级数值", "等级上限", 
                    "黑话", "梗", "成句", "资料相关", "特性一览",
                    "预备-", "spine", "编号相关", "首页", "亮点",
                    "卫星", "职业分支", "模组一览", "分支", "一览",
                    "预告", "公测前", "兑换券", "甄选", "邀请函", "装置",
                    "寻访模拟", "轮换卡池", "资深调用凭证", "凭证", "庆典",
                    "集成战略", "专属", "原型"  # 添加集成战略相关过滤
                ]):
                    continue
                
                # 如果页面名称包含"干员"，提取干员名称
                if "干员" in title:
                    operator_name = title.replace("干员", "").strip()
                    if operator_name and len(operator_name) <= 15:
                        operators.append(operator_name)
                # 如果页面名称不包含"干员"但看起来像干员名称
                elif (len(title) <= 15 and 
                      not any(char in title for char in ["/", "/", "\\"]) and
                      not any(keyword in title for keyword in ["真名", "海报", "凭证", "庆典"])):
                    operators.append(title)
                # 特殊处理：阿米娅的多形态（包含括号的职业标识）
                elif (len(title) <= 20 and 
                      "阿米娅(" in title and 
                      title.endswith(")") and
                      any(career in title for career in ["医疗", "近卫", "术师", "狙击", "重装", "先锋", "特种", "辅助"])):
                    operators.append(title)
                # 特殊处理：带中点的干员名称（如"维娜·维多利亚"）
                elif (len(title) <= 20 and 
                      "·" in title and
                      not any(char in title for char in ["/", "/", "\\"]) and
                      not any(keyword in title for keyword in ["真名", "海报", "凭证", "庆典", "spine", "语音记录"])):
                    operators.append(title)
            
            if operators:
                logger.success(f"通过分类API成功获取 {len(operators)} 个干员")
                return operators
        
        # 方法2：如果分类API失败，尝试搜索特定格式的干员页面
        search_query = (
            f"{WIKI_API}?action=query&format=json&list=search"
            "&srsearch=干员&srlimit=200&utf8=1"
        )
        
        response = await client.get(search_query)
        response.raise_for_status()
        data = response.json()
        
        if "query" in data and "search" in data["query"]:
            operators = []
            
            for index, item in enumerate(data["query"]["search"], 1):
                title = item["title"]
                
                # 过滤掉明显的非干员页面
                if any(keyword in title for keyword in [
                    "密录", "剧情一览", "升级数值", "等级上限", 
                    "黑话", "梗", "成句", "资料相关", "特性一览",
                    "预备-", "spine", "编号相关", "首页", "亮点",
                    "卫星", "职业分支", "模组一览", "分支", "一览",
                    "预告", "公测前", "兑换券", "甄选", "邀请函", "装置",
                    "寻访模拟", "轮换卡池", "资深调用凭证", "凭证", "庆典",
                    "集成战略", "专属", "原型"  # 添加集成战略相关过滤
                ]):
                    continue
                
                # 如果页面名称包含"干员"，提取干员名称
                if "干员" in title:
                    operator_name = title.replace("干员", "").strip()
                    if operator_name and len(operator_name) <= 15:
                        operators.append(operator_name)
                # 如果页面名称不包含"干员"但看起来像干员名称
                elif (len(title) <= 15 and 
                      not any(char in title for char in ["/", "/", "\\"]) and
                      not any(keyword in title for keyword in ["真名", "海报", "凭证", "庆典"])):
                    operators.append(title)
                # 特殊处理：阿米娅的多形态（包含括号的职业标识）
                elif (len(title) <= 20 and 
                      "阿米娅(" in title and 
                      title.endswith(")") and
                      any(career in title for career in ["医疗", "近卫", "术师", "狙击", "重装", "先锋", "特种", "辅助"])):
                    operators.append(title)
                # 特殊处理：带中点的干员名称（如"维娜·维多利亚"）
                elif (len(title) <= 20 and 
                      "·" in title and
                      not any(char in title for char in ["/", "/", "\\"]) and
                      not any(keyword in title for keyword in ["真名", "海报", "凭证", "庆典", "spine", "语音记录"])):
                    operators.append(title)
            
            if operators:
                logger.success(f"通过搜索API成功获取 {len(operators)} 个干员")
                return operators
        
        # 方法3：如果都失败了，尝试获取一些基础干员
        logger.warning("所有自动获取方法都失败了，使用基础干员列表")
        logger.info("尝试方法3：使用基础干员列表...")
        return []
        
    except Exception as e:
        logger.error(f"获取干员列表失败: {e}")
        return []

async def update_data():
    """更新数据"""
    global DATA_DIR, CAMP_FILE, CAREER_FILE, CHARACTERS_FILE
    logger.info("开始更新明日方舟猜谜游戏数据...")
    
    # 确保数据目录存在，若权限异常则回退到用户目录
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        fallback = Path.home() / ".arkguesser" / "data"
        logger.warning(f"数据目录无写权限，切换到 {fallback}")
        fallback.mkdir(parents=True, exist_ok=True)
        DATA_DIR = fallback
        CAMP_FILE = DATA_DIR / "camp.json"
        CAREER_FILE = DATA_DIR / "career.json"
        CHARACTERS_FILE = DATA_DIR / "characters.csv"
    
    async with httpx.AsyncClient(verify=False, timeout=30) as client:
        # 自动获取最新的干员列表
        operators = await get_operator_list(client)
        
        if not operators:
            logger.error("无法获取干员列表，更新终止")
            return False
        
        logger.info(f"开始获取 {len(operators)} 个干员的信息...")
        
        # 获取干员信息
        operator_data = []
        total_operators = len(operators)
        
        logger.info(f"开始获取 {total_operators} 个干员的信息...")
        
        for index, name in enumerate(operators, 1):
            info = await get_operator_info(name, client)
            if info:
                operator_data.append(info)
            # 只在每50个干员时显示进度，减少日志输出
            if index % 50 == 0:
                logger.info(f"已处理 {index}/{total_operators} 个干员...")
        
        logger.info(f"成功获取 {len(operator_data)} 个干员信息")
        
        if not operator_data:
            logger.error("没有获取到任何干员信息，更新终止")
            return False
        
        # 构建阵营数据
        camps = {}
        for op in operator_data:
            if op["camp"] not in camps:
                camps[op["camp"]] = []
            if op["subcamp"] not in camps[op["camp"]]:
                camps[op["camp"]].append(op["subcamp"])
        
        # 构建职业数据
        careers = {}
        for op in operator_data:
            if op["career"] not in careers:
                careers[op["career"]] = []
            if op["subcareer"] not in careers[op["career"]]:
                careers[op["career"]].append(op["subcareer"])
        
        # 保存文件
        logger.info("开始保存数据文件...")
        
        # 保存阵营数据
        with open(CAMP_FILE, 'w', encoding='utf-8-sig', newline='') as f:
            json.dump(camps, f, ensure_ascii=False, indent=2)
        logger.success(f"阵营数据已保存到 {CAMP_FILE}")
        
        # 保存职业数据
        with open(CAREER_FILE, 'w', encoding='utf-8-sig', newline='') as f:
            json.dump(careers, f, ensure_ascii=False, indent=2)
        logger.success(f"职业数据已保存到 {CAREER_FILE}")
        
        # 保存干员数据
        with open(CHARACTERS_FILE, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            
            # 写入表头
            headers = [
                'id', 'name', 'rarity', 'career', 'subcareer', 'camp', 'subcamp', 'race', 'artist',
                'position', 'tag1', 'tag2', 'tag3', 'tag4', 'japanese_voice', 'obtain_method', 'online_time',
                'deploy_cost', 'block_count', 'attack_speed', 'max_hp', 'max_atk', 'max_def', 'max_magic_res',
                'gender', 'birthplace', 'birthday', 'height',
                'skill1_name', 'skill1_type1', 'skill1_type2', 'skill1_spec3_initial', 'skill1_spec3_cost', 'skill1_spec3_duration',
                'skill2_name', 'skill1_type1', 'skill2_type2', 'skill2_spec3_initial', 'skill2_spec3_cost', 'skill2_spec3_duration',
                'skill3_name', 'skill3_type1', 'skill3_type2', 'skill3_spec3_initial', 'skill3_spec3_cost', 'skill3_spec3_duration'
            ]
            writer.writerow(headers)
            
            # 写入数据
            for index, op in enumerate(operator_data, 1):
                row = [
                    op["id"],           # id
                    op["name"],         # name
                    op["rarity"],       # rarity
                    op["career"],       # career
                    op["subcareer"],    # subcareer
                    op["camp"],         # camp
                    op["subcamp"],      # subcamp
                    op["race"],         # race
                    op["artist"],       # artist
                    op["position"],     # position
                    op["tag1"],         # tag1
                    op["tag2"],         # tag2
                    op["tag3"],         # tag3
                    op["tag4"],         # tag4
                    op["japanese_voice"], # japanese_voice
                    op["obtain_method"],  # obtain_method
                    op["online_time"],    # online_time
                    op["deploy_cost"],    # deploy_cost
                    op["block_count"],    # block_count
                    op["attack_speed"],   # attack_speed
                    op["max_hp"],
                    op["max_atk"],
                    op["max_def"],
                    op["max_magic_res"],
                    op["gender"],         # gender
                    op["birthplace"],     # birthplace
                    op["birthday"],       # birthday
                    op["height"],         # height
                    op["skill1_name"],    # skill1_name
                    op["skill1_type1"],   # skill1_type1
                    op["skill1_type2"],   # skill1_type2
                    op["skill1_spec3_initial"], # skill1_spec3_initial
                    op["skill1_spec3_cost"],   # skill1_spec3_cost
                    op["skill1_spec3_duration"], # skill1_spec3_duration
                    op["skill2_name"],    # skill2_name
                    op["skill2_type1"],   # skill2_type1
                    op["skill2_type2"],   # skill2_type2
                    op["skill2_spec3_initial"], # skill2_spec3_initial
                    op["skill2_spec3_cost"],   # skill2_spec3_cost
                    op["skill2_spec3_duration"], # skill2_spec3_duration
                    op["skill3_name"],    # skill3_name
                    op["skill3_type1"],   # skill3_type1
                    op["skill3_type2"],   # skill3_type2
                    op["skill3_spec3_initial"], # skill3_spec3_initial
                    op["skill3_spec3_cost"],   # skill3_spec3_cost
                    op["skill3_spec3_duration"] # skill3_spec3_duration
                ]
                writer.writerow(row)
        
        logger.success(f"干员数据已保存到 {CHARACTERS_FILE}")
        logger.info(f"总共保存了 {len(operator_data)} 个干员的数据")
        # 尝试刷新题库的星级统计缓存（pool_manager）
        try:
            # 优先使用包内相对导入（在包上下文中运行时）
            try:
                from ..game_tools.pool_manager import pool_manager  # type: ignore
            except Exception:
                # 退回到绝对导入（作为独立脚本运行时）
                from nonebot_plugin_arkguesser.game_tools.pool_manager import pool_manager  # type: ignore
            try:
                pool_manager.refresh_rarity_counts()
                logger.success("题库星级统计已刷新")
            except Exception as e:
                logger.warning(f"刷新题库星级统计失败: {e}")
        except Exception as e:
            logger.warning(f"无法导入题库管理器以刷新星级统计: {e}")

        logger.success("数据更新完成！")
        return True

async def main():
    """主函数"""
    try:
        success = await update_data()
        return 0 if success else 1
    except KeyboardInterrupt:
        logger.warning("用户中断更新过程")
        return 1
    except Exception as e:
        logger.error(f"更新过程中发生错误: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
