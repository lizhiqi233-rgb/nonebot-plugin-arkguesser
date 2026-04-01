#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRTS 维基拉取干员数据，写入 characters.csv / aliases.csv，并串联立绘下载与头像对齐表生成。
"""

import re
import csv
import asyncio
import os
import sys
from pathlib import Path

import nonebot_plugin_localstore as store
from typing import Any

import httpx
from nonebot import logger

from .char_art_update import (
    CHAR_ART_LOCAL_MIN_BYTES,
    CHAR_AVATAR_LOCAL_MIN_BYTES,
    sync_six_star_elite2_arts,
    sync_six_star_elite2_avatars,
)

# 强制设置系统编码为UTF-8
if sys.platform.startswith('win'):
    import locale
    locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')

# 插件不应修改用户日志配置，移除自定义日志处理器

# 数据文件路径 - 使用 localstore 插件数据目录（根目录）
DATA_DIR = store.get_plugin_data_dir()
CHARACTERS_FILE = DATA_DIR / "characters.csv"
ALIASES_FILE = DATA_DIR / "aliases.csv"
# 六星精二：头像在整身立绘中的偏移与 slice 缩放，供 new_base.html / 渲染层对齐头部锚点
CHAR_E2_HEAD_ALIGN_FILE = DATA_DIR / "char_e2_head_align.csv"

# 维基百科API
WIKI_API = "https://prts.wiki/api.php"

# 正则表达式（基于arknights-toolkit）
ID_PATTERN = re.compile(r"\|干员id=char_([^|]+?)\n\|")
RARITY_PATTERN = re.compile(r"\|稀有度=(\d+?)\n\|")
CHAR_PATTERN = re.compile(r"\|职业=([^|]+?)\n\|")
SUB_CHAR_PATTERN = re.compile(r"\|分支=([^|]+?)\n\|")
RACE_PATTERN = re.compile(r"\|种族=([^|]+?)\n\|")
# 仅从 PRTS「所属势力」解析：多段为「国家/地区,组织」；单段只填国家/地区（含联动类），子势力为无
ORG_PATTERN_FACTION = re.compile(r"\|所属势力=([^\n]+?)\n\|")
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


def _strip_wiki_line_value(raw: str) -> str:
    """模板行内字段值：去空白及首尾引号（部分页面坐标会写成 \"-226\"）。"""
    s = raw.strip().strip('"').strip("'").strip()
    return s


async def get_operator_info(name: str, client: httpx.AsyncClient) -> dict[str, Any] | None:
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
                                    logger.debug(
                                        f"跳过集成战略专属干员: {name} (获得方式: {obtain_method})"
                                    )
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
                            
                            # 主/子势力：仅 |所属势力=
                            main_camp, sub_camp = parse_camp_from_wiki(content)
                            
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

def parse_camp_from_wiki(content: str) -> tuple[str, str]:
    """
    仅从 |所属势力= 解析。
    两段及以上：第一段为国家/地区，第二段为组织（多段时仍取前两段）。
    单段（含联动等仅写一名称）：只填国家/地区字段，子势力为「无」。
    无该字段或无效：均为「无」（猜测侧不参与底色判定，见游戏模板）。
    """
    m = ORG_PATTERN_FACTION.search(content)
    if not m:
        return "无", "无"
    raw = (m.group(1) or "").strip()
    if not raw or raw == "无信息":
        return "无", "无"
    parts = [p.strip() for p in re.split(r"[,，]", raw) if p.strip()]
    if not parts:
        return "无", "无"
    if len(parts) >= 2:
        return parts[0], parts[1]
    return parts[0], "无"

async def get_operator_list(client: httpx.AsyncClient) -> list[str]:
    """自动获取最新的干员列表"""
    logger.debug("正在获取最新干员列表…")
    
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
                logger.debug(f"通过分类 API 获取 {len(operators)} 个干员")
                logger.info(f"干员列表已就绪，共 {len(operators)} 名")
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
                logger.debug(f"通过搜索 API 获取 {len(operators)} 个干员")
                logger.info(f"干员列表已就绪，共 {len(operators)} 名")
                return operators
        
        # 方法3：如果都失败了，尝试获取一些基础干员
        logger.warning("所有自动获取方法都失败了，使用基础干员列表")
        logger.debug("尝试方法3：使用基础干员列表…")
        return []
        
    except Exception as e:
        logger.error(f"获取干员列表失败: {e}")
        return []


async def update_data():
    """更新数据"""
    global DATA_DIR, CHARACTERS_FILE, ALIASES_FILE, CHAR_E2_HEAD_ALIGN_FILE
    logger.info("开始更新猜谜数据…")
    
    # 确保数据目录存在，若权限异常则回退到用户目录
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        fallback = Path.home() / ".arkguesser" / "data"
        logger.warning(f"数据目录无写权限，切换到 {fallback}")
        fallback.mkdir(parents=True, exist_ok=True)
        DATA_DIR = fallback
        CHARACTERS_FILE = DATA_DIR / "characters.csv"
        ALIASES_FILE = DATA_DIR / "aliases.csv"
        CHAR_E2_HEAD_ALIGN_FILE = DATA_DIR / "char_e2_head_align.csv"
    
    async with httpx.AsyncClient(verify=False, timeout=30) as client:
        # 自动获取最新的干员列表
        operators = await get_operator_list(client)
        
        if not operators:
            logger.error("无法获取干员列表，更新终止")
            return False
        
        operator_data = []
        total_operators = len(operators)
        logger.info(f"拉取干员详情（{total_operators} 名）…")

        for name in operators:
            info = await get_operator_info(name, client)
            if info:
                operator_data.append(info)

        logger.info(f"干员详情：有效 {len(operator_data)} / {total_operators}")
        
        if not operator_data:
            logger.error("没有获取到任何干员信息，更新终止")
            return False
        
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
        
        logger.success(f"干员 CSV 已写入 {CHARACTERS_FILE}（{len(operator_data)} 条）")

        try:
            await sync_six_star_elite2_arts(CHARACTERS_FILE, DATA_DIR, client)
        except Exception as e:
            logger.warning(f"六星立绘同步失败（不影响 CSV）: {e}")

        try:
            await sync_six_star_elite2_avatars(CHARACTERS_FILE, DATA_DIR, client)
        except Exception as e:
            logger.warning(f"六星精二头像同步失败（不影响 CSV）: {e}")

        try:
            try:
                from .char_art_match import (
                    CHAR_AVATAR_ALIGN_DEBUG_REL_DIR,
                    rebuild_char_e2_head_align_csv,
                )
            except ImportError:
                # 脚本方式运行（__main__）或 cwd 不在 resource_tools 时：保证同目录可导入
                _rt_dir = Path(__file__).resolve().parent
                if str(_rt_dir) not in sys.path:
                    sys.path.insert(0, str(_rt_dir))
                from char_art_match import (  # type: ignore
                    CHAR_AVATAR_ALIGN_DEBUG_REL_DIR,
                    rebuild_char_e2_head_align_csv,
                )

            # 默认不生成 char_avatar_align_debug；设 ARKGUESSER_HEAD_ALIGN_DEBUG=1 开启
            _head_dbg = os.environ.get("ARKGUESSER_HEAD_ALIGN_DEBUG", "0").strip().lower() in (
                "1",
                "true",
                "yes",
            )
            # 默认增量：char_e2_head_align.csv 已有 char_id 则跳过识别；设 ARKGUESSER_HEAD_ALIGN_REBUILD_ALL=1 全员重算
            _head_align_full = os.environ.get(
                "ARKGUESSER_HEAD_ALIGN_REBUILD_ALL", "0"
            ).strip().lower() in ("1", "true", "yes")

            def _run_head_align() -> tuple[int, int, int, int]:
                return rebuild_char_e2_head_align_csv(
                    DATA_DIR,
                    CHARACTERS_FILE,
                    CHAR_E2_HEAD_ALIGN_FILE,
                    min_art_bytes=CHAR_ART_LOCAL_MIN_BYTES,
                    min_avatar_bytes=CHAR_AVATAR_LOCAL_MIN_BYTES,
                    max_workers=4,
                    write_debug_images=_head_dbg,
                    reuse_existing_rows=not _head_align_full,
                )

            ok_h, skip_h, fail_h, reused_h = await asyncio.to_thread(_run_head_align)
            logger.success(
                f"精二头像对齐表已写入 {CHAR_E2_HEAD_ALIGN_FILE}（匹配成功 {ok_h}，跳过 {skip_h}，失败 {fail_h}，复用已有 {reused_h}）"
            )
            if _head_dbg:
                logger.info(
                    f"头像匹配调试图：{DATA_DIR / CHAR_AVATAR_ALIGN_DEBUG_REL_DIR}（默认关闭；仅调试时开启）"
                )
            if _head_align_full:
                logger.info("已使用全量重算（ARKGUESSER_HEAD_ALIGN_REBUILD_ALL=1）")
        except ImportError as e:
            msg = str(e).lower()
            if "cv2" in msg or "opencv" in msg:
                hint = "需安装 opencv-python、numpy"
            elif "parent package" in msg or "relative import" in msg:
                hint = "导入路径异常（请从项目根目录执行 python resource_tools/data_update.py）"
            else:
                hint = "依赖或导入失败"
            logger.warning(f"跳过精二头像对齐表（{hint}）: {e}")
        except Exception as e:
            logger.warning(f"精二头像对齐表生成失败（不影响 CSV）: {e}")

        # 生成/更新别称表：operator_name, aliases（同一干员多个别称存同一单元格）
        # - 仅负责“生成骨架 + 继承旧值”，不自动抓取别称，方便你自行维护
        existing_aliases: dict[str, str] = {}
        try:
            if ALIASES_FILE.exists():
                with open(ALIASES_FILE, "r", encoding="utf-8-sig", newline="") as f:
                    # 优先按表头读取，避免手工编辑导致列错位
                    # 期望表头：operator_name, aliases
                    dict_reader = csv.DictReader(f)
                    if dict_reader.fieldnames and "operator_name" in dict_reader.fieldnames:
                        for row in dict_reader:
                            operator_name = (row.get("operator_name") or "").strip()
                            aliases = (row.get("aliases") or "").strip()
                            if operator_name:
                                existing_aliases[operator_name] = aliases
                    else:
                        # 无表头时回退到两列读取
                        f.seek(0)
                        reader = csv.reader(f)
                        for row in reader:
                            if not row:
                                continue
                            operator_name = (row[0].strip() if len(row) >= 1 else "")
                            aliases = (row[1].strip() if len(row) >= 2 else "")
                            if operator_name and operator_name != "operator_name":
                                existing_aliases[operator_name] = aliases
        except Exception as e:
            logger.warning(f"读取旧别称表失败，将重新生成: {e}")

        try:
            # 原子写入：先写临时文件，再替换，避免更新中断导致 aliases.csv 变成半截
            tmp_file = ALIASES_FILE.with_suffix(ALIASES_FILE.suffix + ".tmp")
            with open(tmp_file, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["operator_name", "aliases"])
                # 以 operator_name 键控对齐，避免“新增干员导致错位”
                for op_name in sorted({op["name"] for op in operator_data}):
                    writer.writerow([op_name, existing_aliases.get(op_name, "")])
            tmp_file.replace(ALIASES_FILE)
            logger.debug(f"别称表已保存到 {ALIASES_FILE}")
        except Exception as e:
            logger.warning(f"保存别称表失败: {e}")
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
                logger.debug("题库星级统计已刷新")
            except Exception as e:
                logger.warning(f"刷新题库星级统计失败: {e}")
        except Exception as e:
            logger.warning(f"无法导入题库管理器以刷新星级统计: {e}")

        logger.success("数据更新完成")
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
