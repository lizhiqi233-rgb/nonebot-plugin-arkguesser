from pydantic import BaseModel, Field
from nonebot import get_plugin_config as nb_get_plugin_config

class ArkGuesserConfig(BaseModel):
    """插件配置类"""
    
    # 最大尝试次数
    arkguesser_max_attempts: int = Field(default=10, description="最大尝试次数")
    
    # 默认星级范围
    arkguesser_default_rarity_range: str = Field(default="6", description="默认星级范围")
    
    # 默认游戏模式
    arkguesser_default_mode: str = Field(default="大头", description="默认游戏模式")
    
    class Config:
        extra = "ignore"

# 获取插件配置实例（使用 nonebot 官方 API）
def get_plugin_config() -> ArkGuesserConfig:
    try:
        return nb_get_plugin_config(ArkGuesserConfig)
    except Exception:
        return ArkGuesserConfig()