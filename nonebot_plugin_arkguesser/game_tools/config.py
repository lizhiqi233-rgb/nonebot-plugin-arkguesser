from pydantic import BaseModel, Field
from nonebot import get_driver

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

# 获取插件配置实例
def get_plugin_config() -> ArkGuesserConfig:
    """获取插件配置实例"""
    try:
        driver = get_driver()
        # 尝试从driver.config获取配置
        if hasattr(driver.config, 'arkguesser'):
            return driver.config.arkguesser
        else:
            # 如果没有配置，返回默认配置
            return ArkGuesserConfig()
    except Exception:
        # 如果出现任何错误，返回默认配置
        return ArkGuesserConfig()