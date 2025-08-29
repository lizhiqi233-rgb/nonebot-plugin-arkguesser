#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
明日方舟立绘下载程序启动脚本
提供简单的命令行界面来运行下载程序
"""

import asyncio
import argparse
# 兼容相对/绝对导入，支持脚本独立运行
try:
    from .illustration_downloader_v2 import ArknightsIllustrationDownloaderV2  # type: ignore
except Exception:
    from illustration_downloader_v2 import ArknightsIllustrationDownloaderV2  # type: ignore


def create_basic_config():
    """创建基础配置"""
    return {
        'download': {
            'max_concurrent': 3,
            'timeout': 60,
        },
        'filter': {
            'min_rarity': 1,
            'max_rarity': 6,
        },
        'output': {
            'create_subdirectories': True,
            'subdirectory_structure': {
                'by_rarity': True,
                'by_career': False,
                'by_type': True
            }
        }
    }


def create_high_quality_config():
    """创建高质量配置（只下载高星干员）"""
    return {
        'download': {
            'max_concurrent': 5,
            'timeout': 60,
        },
        'filter': {
            'min_rarity': 4,  # 只下载4星以上干员
            'max_rarity': 6,
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


def create_specific_career_config(careers):
    """创建特定职业配置"""
    return {
        'download': {
            'max_concurrent': 3,
            'timeout': 60,
        },
        'filter': {
            'min_rarity': 1,
            'max_rarity': 6,
            'include_careers': careers
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


def create_test_config():
    """创建测试配置（只下载少量干员进行测试）"""
    return {
        'download': {
            'max_concurrent': 2,
            'timeout': 30,
        },
        'filter': {
            'min_rarity': 6,  # 只下载6星干员
            'max_rarity': 6,
            'include_careers': ['近卫']  # 只下载近卫职业
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


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="明日方舟干员立绘自动下载程序",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python run_illustration_download.py                    # 使用默认配置
  python run_illustration_download.py --basic            # 基础配置
  python run_illustration_download.py --high-quality     # 高质量配置（4星以上）
  python run_illustration_download.py --careers 近卫 狙击  # 特定职业
  python run_illustration_download.py --test             # 测试配置
  python run_illustration_download.py --output my_illustrations  # 自定义输出目录
        """
    )
    
    # 配置选项
    parser.add_argument('--basic', action='store_true', 
                       help='使用基础配置（推荐新手）')
    parser.add_argument('--high-quality', action='store_true',
                       help='使用高质量配置（只下载4星以上干员）')
    parser.add_argument('--careers', nargs='+', metavar='CAREER',
                       help='指定要下载的职业（如：近卫 狙击 术师）')
    parser.add_argument('--test', action='store_true',
                       help='使用测试配置（只下载少量干员进行测试）')
    
    # 输出选项
    parser.add_argument('--output', '-o', metavar='DIR',
                       help='指定输出目录（默认：illustrations）')
    
    # 下载选项
    parser.add_argument('--concurrent', '-c', type=int, metavar='NUM',
                       help='设置并发下载数（默认：3）')
    parser.add_argument('--timeout', '-t', type=int, metavar='SECONDS',
                       help='设置超时时间（默认：60秒）')
    
    # 过滤选项
    parser.add_argument('--min-rarity', type=int, metavar='RARITY',
                       help='设置最小稀有度（1-6）')
    parser.add_argument('--max-rarity', type=int, metavar='RARITY',
                       help='设置最大稀有度（1-6）')
    
    args = parser.parse_args()
    
    # 选择配置
    if args.test:
        config = create_test_config()
        print("使用测试配置：只下载6星近卫干员")
    elif args.high_quality:
        config = create_high_quality_config()
        print("使用高质量配置：只下载4星以上干员")
    elif args.careers:
        config = create_specific_career_config(args.careers)
        print(f"使用特定职业配置：只下载 {', '.join(args.careers)} 职业干员")
    elif args.basic:
        config = create_basic_config()
        print("使用基础配置")
    else:
        config = create_basic_config()
        print("使用默认配置")
    
    # 应用命令行参数
    if args.concurrent:
        config['download']['max_concurrent'] = args.concurrent
        print(f"设置并发数：{args.concurrent}")
    
    if args.timeout:
        config['download']['timeout'] = args.timeout
        print(f"设置超时时间：{args.timeout}秒")
    
    if args.min_rarity:
        config['filter']['min_rarity'] = args.min_rarity
        print(f"设置最小稀有度：{args.min_rarity}")
    
    if args.max_rarity:
        config['filter']['max_rarity'] = args.max_rarity
        print(f"设置最大稀有度：{args.max_rarity}")
    
    # 显示配置信息
    print("\n当前配置:")
    print(f"  并发数: {config['download']['max_concurrent']}")
    print(f"  超时时间: {config['download']['timeout']}秒")
    print(f"  稀有度范围: {config['filter']['min_rarity']}-{config['filter']['max_rarity']}")
    if 'include_careers' in config['filter']:
        print(f"  职业限制: {', '.join(config['filter']['include_careers'])}")
    print(f"  输出目录: {args.output or 'illustrations'}")
    
    # 确认开始下载
    print("\n准备开始下载...")
    try:
        input("按回车键开始下载，或按 Ctrl+C 取消...")
    except KeyboardInterrupt:
        print("\n下载已取消")
        return
    
    # 创建下载器并运行
    try:
        downloader = ArknightsIllustrationDownloaderV2(
            output_dir=args.output,
            config=config
        )
        await downloader.run()
    except KeyboardInterrupt:
        print("\n下载被用户中断")
    except Exception as e:
        print(f"\n下载过程中出现错误: {e}")
        print("请检查日志文件获取详细信息")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n程序运行出错: {e}")
        sys.exit(1)
