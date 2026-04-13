"""
资源工具子包：
- data_update：PRTS 维基 CSV、别称表与完整更新流水线
- char_art_update：六星精二立绘/头像下载
- char_art_match：头像在立绘中的匹配；char_e2_head_align.csv 仅存 id/name/绿框 xywh/scale（两位小数）

避免在包导入时加载 char_art_match（依赖 OpenCV），仅导出 update_data。
"""

from .data_update import update_data

__all__ = ["update_data"]
