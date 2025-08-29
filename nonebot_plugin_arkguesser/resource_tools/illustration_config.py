#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
明日方舟立绘下载配置文件
"""

import os
from pathlib import Path

# 在独立脚本环境中，导入 nonebot_plugin_localstore 可能因 NoneBot 未初始化而失败
def _safe_get_data_dir() -> Path:
    try:
        from nonebot_plugin_localstore import get_plugin_data_dir as _get_dir  # type: ignore
        return _get_dir()
    except Exception:
        # 回退到 localstore 的默认路径结构
        return Path.home() / ".local" / "share" / "nonebot2" / "nonebot_plugin_arkguesser"

# 基础配置：使用 localstore 提供的插件数据目录
DATA_DIR = _safe_get_data_dir()
OUTPUT_DIR = DATA_DIR / "illustrations"

# 干员数据文件路径（与更新脚本保持一致，位于 DATA_DIR 根目录）
CHARACTERS_CSV_PATH = DATA_DIR / "characters.csv"

# 下载配置
DOWNLOAD_CONFIG = {
    "max_concurrent": 3,  # 最大并发下载数
    "timeout": 60,         # 请求超时时间（秒）
    "retry_times": 3,      # 重试次数
    "delay_between_requests": 0.5,  # 请求间隔（秒）
}

# PRTS Wiki API 配置
PRTS_CONFIG = {
    "base_url": "https://prts.wiki/api.php",
    "assets_url": "https://torappu.prts.wiki/assets",
    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# 立绘类型配置
ILLUSTRATION_TYPES = {
    "半身像": {
        "folder": "char_portrait",
        "suffixes": ["_1", "_2"],  # 精英化等级后缀
        "extensions": [".png", ".jpg", ".webp"]
    },
    "头像": {
        "folder": "char_avatar",
        "suffixes": [""],  # 无后缀
        "extensions": [".png", ".jpg", ".webp"]
    },
    "皮肤立绘": {
        "folder": "char_portrait",
        "suffixes": ["_skin1", "_skin2"],  # 皮肤后缀
        "extensions": [".png", ".jpg", ".webp"]
    }
}

# 日志配置
LOG_CONFIG = {
    "level": "INFO",
    "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    "file": "illustration_download.log",
    "rotation": "1 day",
    "retention": "7 days"
}

# 文件命名配置
NAMING_CONFIG = {
    "separator": "_",
    "include_rarity": False,  # 不包含稀有度信息
    "include_career": False,
    "include_level": True,
    "filename_format": "{name}_{type}_{level}"
}

# 过滤配置
FILTER_CONFIG = {
    "min_rarity": 1,      # 最小稀有度
    "max_rarity": 6,      # 最大稀有度
    "include_careers": [   # 包含的职业
        "先锋", "近卫", "重装", "狙击", "术师", "医疗", "辅助", "特种"
    ],
    "exclude_operators": [  # 排除的干员ID
        # 可以在这里添加不需要下载的干员ID
    ]
}

# 输出配置
OUTPUT_CONFIG = {
    "create_subdirectories": True,  # 是否创建子目录
    "subdirectory_structure": {     # 子目录结构
        "by_rarity": True,          # 按稀有度分类
        "by_career": False,         # 按职业分类
        "by_type": True             # 按立绘类型分类
    },
    "save_metadata": True,          # 是否保存元数据
    "metadata_format": "json"       # 元数据格式
}

# 验证配置
VALIDATION_CONFIG = {
    "verify_image_integrity": True,  # 验证图片完整性
    "min_file_size": 1024,           # 最小文件大小（字节）
    "max_file_size": 10 * 1024 * 1024,  # 最大文件大小（字节）
    "allowed_formats": ["png", "jpg", "jpeg", "webp"]  # 允许的图片格式
}

# 网络配置
NETWORK_CONFIG = {
    "use_proxy": False,              # 是否使用代理
    "proxy": {                       # 代理配置
        "http": None,
        "https": None
    },
    "verify_ssl": False,             # 是否验证SSL证书
    "follow_redirects": True,        # 是否跟随重定向
    "max_redirects": 5               # 最大重定向次数
}

# 错误处理配置
ERROR_HANDLING_CONFIG = {
    "continue_on_error": True,       # 遇到错误时是否继续
    "log_errors": True,              # 是否记录错误
    "save_failed_list": True,        # 是否保存失败列表
    "retry_failed": True,            # 是否重试失败的下载
    "max_retry_attempts": 3          # 最大重试次数
}

# 性能配置
PERFORMANCE_CONFIG = {
    "chunk_size": 8192,              # 下载块大小
    "buffer_size": 1024 * 1024,     # 缓冲区大小
    "use_async_io": True,            # 是否使用异步IO
    "memory_limit": 100 * 1024 * 1024  # 内存限制（字节）
}
