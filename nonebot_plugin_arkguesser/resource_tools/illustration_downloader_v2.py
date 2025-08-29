#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
明日方舟干员立绘自动下载程序 V2
基于 nonebot-plugin-arkguesser-main 的干员数据
使用 PRTS Wiki API 获取干员立绘图片
增强版本，支持更多配置选项和功能
"""

import asyncio
import csv
import json
import re
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from urllib.parse import quote
from dataclasses import dataclass

import httpx
from PIL import Image
from nonebot import logger

# 兼容相对/绝对导入，支持脚本独立运行
try:
    from .illustration_config import *  # type: ignore
except Exception:
    from illustration_config import *  # type: ignore


@dataclass
class Operator:
    """干员信息数据类"""
    charid: str
    name: str
    original_name: str
    rarity: str
    career: str
    subcareer: str = ""
    camp: str = ""
    race: str = ""


@dataclass
class Illustration:
    """立绘信息数据类"""
    type: str
    filename: str
    charid: str
    level: Optional[int] = None
    url: Optional[str] = None
    file_size: Optional[int] = None
    md5_hash: Optional[str] = None


class ArknightsIllustrationDownloaderV2:
    """明日方舟干员立绘下载器 V2"""
    
    def __init__(self, output_dir: Optional[str] = None, config: Optional[Dict] = None):
        self.output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
        self.output_dir.mkdir(exist_ok=True)
        
        # 使用配置文件
        self.config = config or {}
        self.download_config = {**DOWNLOAD_CONFIG, **self.config.get('download', {})}
        self.prts_config = {**PRTS_CONFIG, **self.config.get('prts', {})}
        self.illustration_types = {**ILLUSTRATION_TYPES, **self.config.get('illustration_types', {})}
        
        # 干员数据文件路径
        self.characters_csv_path = CHARACTERS_CSV_PATH
        
        # 下载统计
        self.stats = {
            "total_operators": 0,
            "successful_downloads": 0,
            "failed_downloads": 0,
            "skipped_downloads": 0,
            "total_file_size": 0,
            "start_time": None,
            "end_time": None
        }
        
        # 失败列表
        self.failed_downloads = []
        
        # 不修改用户日志配置
    
    def _setup_logging(self):
        """兼容方法：不再添加自定义日志处理器"""
        return
    
    def load_operators_from_csv(self) -> List[Operator]:
        """从CSV文件加载干员数据"""
        logger.info("正在从CSV文件加载干员数据...")
        
        operators = []
        
        try:
            with open(self.characters_csv_path, 'r', encoding='utf-8-sig') as f:  # 使用utf-8-sig处理BOM
                reader = csv.DictReader(f)
                for row in reader:
                    # 清理数据，去除多余的字符
                    char_id = row['id'].strip()
                    name = row['name'].strip()
                    rarity = row['rarity'].strip()
                    career = row['career'].strip()
                    subcareer = row.get('subcareer', '').strip()
                    camp = row.get('camp', '').strip()
                    race = row.get('race', '').strip()
                    
                    # 跳过无效数据（包括表头行）
                    if not char_id or not name or char_id == 'id' or name == 'name':
                        continue
                    
                    # 应用过滤条件
                    if not self._should_include_operator(char_id, rarity, career):
                        continue
                    
                    # 清理名称中的特殊字符
                    clean_name = re.sub(r'[\(\)（）]', '', name)
                    clean_name = clean_name.strip()
                    
                    operator = Operator(
                        charid=char_id,
                        name=clean_name,
                        original_name=name,
                        rarity=rarity,
                        career=career,
                        subcareer=subcareer,
                        camp=camp,
                        race=race
                    )
                    
                    operators.append(operator)
            
            logger.success(f"成功加载 {len(operators)} 个干员信息")
            self.stats["total_operators"] = len(operators)
            return operators
            
        except Exception as e:
            logger.error(f"加载CSV文件失败: {e}")
            return []
    
    def _should_include_operator(self, char_id: str, rarity: str, career: str) -> bool:
        """判断是否应该包含该干员"""
        filter_config = FILTER_CONFIG.copy()
        filter_config.update(self.config.get('filter', {}))
        
        # 检查是否在排除列表中
        if char_id in filter_config.get('exclude_operators', []):
            return False
        
        # 检查稀有度范围
        try:
            rarity_num = int(rarity)
            if not (filter_config['min_rarity'] <= rarity_num <= filter_config['max_rarity']):
                return False
        except (ValueError, TypeError):
            pass
        
        # 检查职业
        if career and filter_config.get('include_careers'):
            if career not in filter_config['include_careers']:
                return False
        
        return True
    
    def _get_output_path(self, operator: Operator, illustration: Illustration) -> Path:
        """获取输出文件路径"""
        output_config = OUTPUT_CONFIG.copy()
        output_config.update(self.config.get('output', {}))
        
        if not output_config['create_subdirectories']:
            return self.output_dir / f"{operator.name}_{illustration.type}_{illustration.level or '1'}.png"
        
        # 创建子目录结构
        subdirs = []
        
        if output_config['subdirectory_structure'].get('by_rarity'):
            subdirs.append(f"稀有度{operator.rarity}")
        
        if output_config['subdirectory_structure'].get('by_career'):
            subdirs.append(operator.career)
        
        if output_config['subdirectory_structure'].get('by_type'):
            subdirs.append(illustration.type)
        
        output_path = self.output_dir
        for subdir in subdirs:
            output_path = output_path / subdir
            output_path.mkdir(exist_ok=True)
        
        # 构建文件名
        naming_config = NAMING_CONFIG.copy()
        naming_config.update(self.config.get('naming', {}))
        
        filename_parts = [operator.name]
        
        if naming_config['include_rarity']:
            filename_parts.append(f"稀有度{operator.rarity}")
        
        if naming_config['include_career']:
            filename_parts.append(operator.career)
        
        filename_parts.append(illustration.type)
        
        if naming_config['include_level'] and illustration.level:
            filename_parts.append(f"精英{illustration.level}")
        
        filename = naming_config['separator'].join(filename_parts) + ".png"
        
        return output_path / filename
    
    async def get_operator_illustrations(self, client: httpx.AsyncClient, operator: Operator) -> List[Illustration]:
        """获取干员的立绘信息"""
        logger.debug(f"正在获取 {operator.name}({operator.charid}) 的立绘信息...")
        
        illustrations = []
        
        # 直接使用默认的立绘路径，跳过页面解析
        try:
            # 标准半身像路径
            for level in [1, 2]:  # 精英化等级
                illustrations.append(Illustration(
                    type="半身像",
                    filename=f"{operator.charid}_{level}.png",
                    charid=operator.charid,
                    level=level
                ))
            
            # 头像路径
            illustrations.append(Illustration(
                type="头像", 
                filename=f"{operator.charid}.png",
                charid=operator.charid
            ))
            
            logger.debug(f"找到 {len(illustrations)} 个立绘文件")
            return illustrations
            
        except Exception as e:
            logger.error(f"获取 {operator.name} 立绘信息失败: {e}")
            return []
    
    async def download_illustration(self, client: httpx.AsyncClient, operator: Operator, illustration: Illustration) -> bool:
        """下载立绘图片"""
        try:
            # 构建下载URL
            if illustration.type == "半身像":
                level = illustration.level or 1
                url = f"{self.prts_config['assets_url']}/char_portrait/{operator.charid}_{level}.png"
                illustration.url = url
            elif illustration.type == "头像":
                url = f"{self.prts_config['assets_url']}/char_avatar/{operator.charid}.png"
                illustration.url = url
            else:
                # 如果有自定义文件名，尝试直接下载
                url = f"{self.prts_config['assets_url']}/char_portrait/{illustration.filename}"
                illustration.url = url
            
            logger.debug(f"正在下载: {url}")
            
            # 添加请求头
            headers = {
                "User-Agent": self.prts_config['user_agent']
            }
            
            response = await client.get(url, headers=headers, timeout=20.0)
            if response.status_code != 200:
                logger.warning(f"下载失败 {operator.name}: HTTP {response.status_code}")
                return False
            
            # 获取输出路径
            output_path = self._get_output_path(operator, illustration)
            
            # 保存图片
            with open(output_path, "wb") as f:
                f.write(response.content)
            
            # 更新立绘信息
            illustration.file_size = len(response.content)
            illustration.md5_hash = hashlib.md5(response.content).hexdigest()
            
            # 验证图片
            if not self._validate_image(output_path, illustration):
                output_path.unlink(missing_ok=True)
                return False
            
            logger.success(f"成功下载: {output_path.name}")
            return True
                
        except Exception as e:
            logger.error(f"下载立绘失败 {operator.name}: {e}")
            return False
    
    def _validate_image(self, file_path: Path, illustration: Illustration) -> bool:
        """验证图片文件"""
        validation_config = VALIDATION_CONFIG.copy()
        validation_config.update(self.config.get('validation', {}))
        
        try:
            # 检查文件大小
            file_size = file_path.stat().st_size
            if file_size < validation_config['min_file_size']:
                logger.warning(f"文件过小: {file_path.name} ({file_size} bytes)")
                return False
            
            if file_size > validation_config['max_file_size']:
                logger.warning(f"文件过大: {file_path.name} ({file_size} bytes)")
                return False
            
            # 验证图片完整性
            if validation_config['verify_image_integrity']:
                with Image.open(file_path) as img:
                    img.verify()
                    
                    # 检查图片格式
                    if img.format.lower() not in validation_config['allowed_formats']:
                        logger.warning(f"不支持的图片格式: {file_path.name} ({img.format})")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"图片验证失败 {file_path.name}: {e}")
            return False
    
    async def download_operator_illustrations(self, client: httpx.AsyncClient, operator: Operator) -> Dict:
        """下载单个干员的所有立绘"""
        logger.info(f"正在处理干员: {operator.name} ({operator.charid})")
        
        # 获取立绘信息
        illustrations = await self.get_operator_illustrations(client, operator)
        
        if not illustrations:
            logger.warning(f"{operator.name}: 没有找到立绘信息")
            return {"name": operator.name, "success": 0, "total": 0, "skipped": True}
        
        # 下载立绘
        success_count = 0
        for illustration in illustrations:
            if await self.download_illustration(client, operator, illustration):
                success_count += 1
                self.stats["successful_downloads"] += 1
                if illustration.file_size:
                    self.stats["total_file_size"] += illustration.file_size
            else:
                self.stats["failed_downloads"] += 1
                self.failed_downloads.append({
                    "operator": operator.name,
                    "charid": operator.charid,
                    "illustration": illustration.type,
                    "error": "下载失败"
                })
        
        if success_count > 0:
            logger.info(f"{operator.name}: 成功下载 {success_count}/{len(illustrations)} 个立绘")
        else:
            logger.warning(f"{operator.name}: 没有成功下载任何立绘")
        
        return {"name": operator.name, "success": success_count, "total": len(illustrations), "skipped": False}
    
    async def download_all_illustrations(self, operators: List[Operator]):
        """下载所有干员的立绘"""
        logger.info(f"开始下载 {len(operators)} 个干员的立绘...")
        
        semaphore = asyncio.Semaphore(self.download_config['max_concurrent'])
        
        async def download_operator(operator: Operator):
            async with semaphore:
                return await self.download_operator_illustrations(client, operator)
        
        # 网络配置
        network_config = NETWORK_CONFIG.copy()
        network_config.update(self.config.get('network', {}))
        
        client_config = {
            "verify": network_config['verify_ssl'],
            "follow_redirects": network_config['follow_redirects'],
            "timeout": self.download_config['timeout']
        }
        
        if network_config['use_proxy'] and network_config['proxy']:
            client_config['proxies'] = network_config['proxy']
        
        async with httpx.AsyncClient(**client_config) as client:
            tasks = [download_operator(op) for op in operators]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"下载过程中出现异常: {result}")
                elif result.get("skipped"):
                    self.stats["skipped_downloads"] += 1
        
        logger.success("所有立绘下载完成！")
    
    def save_download_stats(self):
        """保存下载统计信息"""
        stats_file = self.output_dir / "download_stats.json"
        
        stats_data = {
            "download_time": {
                "start": self.stats["start_time"],
                "end": self.stats["end_time"],
                "duration": self.stats["end_time"] - self.stats["start_time"] if self.stats["start_time"] and self.stats["end_time"] else None
            },
            "statistics": self.stats,
            "failed_downloads": self.failed_downloads,
            "output_directory": str(self.output_dir.absolute()),
            "configuration": {
                "download": self.download_config,
                "filter": FILTER_CONFIG,
                "output": OUTPUT_CONFIG
            }
        }
        
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"下载统计已保存到: {stats_file}")
    
    def print_summary(self):
        """打印下载总结"""
        logger.info("=" * 60)
        logger.info("下载完成总结")
        logger.info("=" * 60)
        logger.info(f"总干员数: {self.stats['total_operators']}")
        logger.info(f"成功下载: {self.stats['successful_downloads']}")
        logger.info(f"下载失败: {self.stats['failed_downloads']}")
        logger.info(f"跳过下载: {self.stats['skipped_downloads']}")
        logger.info(f"总文件大小: {self.stats['total_file_size'] / (1024*1024):.2f} MB")
        
        if self.stats['start_time'] and self.stats['end_time']:
            duration = self.stats['end_time'] - self.stats['start_time']
            logger.info(f"总耗时: {duration}")
        
        logger.info(f"输出目录: {self.output_dir.absolute()}")
        
        if self.failed_downloads:
            logger.info(f"失败列表已保存到统计文件中")
        
        logger.info("=" * 60)
    
    async def run(self):
        """运行下载器"""
        logger.info("明日方舟干员立绘下载器 V2 启动")
        
        self.stats["start_time"] = time.time()
        
        # 加载干员数据
        operators = self.load_operators_from_csv()
        
        if not operators:
            logger.error("无法加载干员数据，程序退出")
            return
        
        # 保存干员信息到JSON文件
        if OUTPUT_CONFIG['save_metadata']:
            info_file = self.output_dir / "operators_info.json"
            operators_data = [
                {
                    "charid": op.charid,
                    "name": op.name,
                    "original_name": op.original_name,
                    "rarity": op.rarity,
                    "career": op.career,
                    "subcareer": op.subcareer,
                    "camp": op.camp,
                    "race": op.race
                }
                for op in operators
            ]
            
            with open(info_file, "w", encoding="utf-8") as f:
                json.dump(operators_data, f, ensure_ascii=False, indent=2)
            logger.info(f"干员信息已保存到: {info_file}")
        
        # 下载立绘
        await self.download_all_illustrations(operators)
        
        self.stats["end_time"] = time.time()
        
        # 保存统计信息
        self.save_download_stats()
        
        # 打印总结
        self.print_summary()


async def main():
    """主函数"""
    # 自定义配置示例 - 下载所有职业和星级的干员立绘
    custom_config = {
        'download': {
            'max_concurrent': 5,  # 增加并发数
            'timeout': 30,         # 减少超时时间
        },
        'filter': {
            'min_rarity': 1,       # 下载1星以上干员（包含所有星级）
            'max_rarity': 6,       # 下载6星干员
            'include_careers': [   # 包含所有职业
                "先锋", "近卫", "重装", "狙击", "术师", "医疗", "辅助", "特种"
            ]
        },
        'output': {
            'create_subdirectories': True,
            'subdirectory_structure': {
                'by_rarity': True,
                'by_career': True,
                'by_type': True
            }
        }
    }
    
    downloader = ArknightsIllustrationDownloaderV2(config=custom_config)
    await downloader.run()


if __name__ == "__main__":
    asyncio.run(main())
