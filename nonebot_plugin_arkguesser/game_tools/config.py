from pydantic import BaseModel, Field
try:
    # pydantic v2
    from pydantic import ConfigDict  # type: ignore
    _PD_V2 = True
except Exception:
    _PD_V2 = False
from nonebot import get_plugin_config as nb_get_plugin_config

class ArkGuesserConfig(BaseModel):
    """插件配置类"""
    
    # 最大尝试次数
    arkguesser_max_attempts: int = Field(default=10, description="最大尝试次数")
    
    # 默认星级范围
    arkguesser_default_rarity_range: str = Field(default="6", description="默认星级范围")
    
    # 默认游戏模式
    arkguesser_default_mode: str = Field(default="大头", description="默认游戏模式")
    
    # 最近已出干员排除数量，保证该数量内不重复抽取（0 表示不排除）
    arkguesser_recent_exclude_count: int = Field(default=40, ge=0, description="最近已出干员排除数量")

    # 猜干员结果图相对设计稿 540×540 的缩放（1 为默认；1.5/2 即放大输出分辨率，字体与版面同比缩放）
    arkguesser_render_scale: float = Field(
        default=1.0,
        ge=0.25,
        le=8.0,
        description="渲染面板缩放倍数（基准 540×540）",
    )
    
    if _PD_V2:
        # pydantic v2 配置
        model_config = ConfigDict(extra="ignore")  # type: ignore
    else:
        # pydantic v1 配置
        class Config:
            extra = "ignore"

# 获取插件配置实例（使用 nonebot 官方 API）
def get_plugin_config() -> ArkGuesserConfig:
    try:
        return nb_get_plugin_config(ArkGuesserConfig)
    except Exception:
        return ArkGuesserConfig()