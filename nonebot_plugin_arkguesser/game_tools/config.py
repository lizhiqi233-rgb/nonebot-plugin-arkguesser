import nonebot
from pydantic import BaseModel, Field

try:
    from pydantic import ConfigDict  # type: ignore

    _PD_V2 = True
except Exception:
    _PD_V2 = False


class ArkGuesserScoped(BaseModel):
    """插件配置本体。.env 使用嵌套键名，例如 ARKGUESSER__MAX_ATTEMPTS（见 NoneBot 文档「插件配置」scope）。"""

    max_attempts: int = Field(default=10, description="最大尝试次数")
    default_rarity_range: str = Field(default="6", description="默认星级范围")
    default_mode: str = Field(default="大头", description="默认游戏模式")
    recent_exclude_count: int = Field(default=40, ge=0, description="最近已出干员排除数量")
    render_scale: float = Field(
        default=1.0,
        ge=0.25,
        le=8.0,
        description="渲染面板缩放倍数（基准 540×540）",
    )

    if _PD_V2:
        model_config = ConfigDict(extra="ignore")  # type: ignore
    else:

        class Config:
            extra = "ignore"


class ArkGuesserConfig(BaseModel):
    """供 PluginMetadata 注册；与 get_plugin_config(ArkGuesserConfig) 配合解析全局配置中的 arkguesser 段。"""

    arkguesser: ArkGuesserScoped = Field(default_factory=ArkGuesserScoped)

    if _PD_V2:
        model_config = ConfigDict(extra="ignore")  # type: ignore
    else:

        class Config:
            extra = "ignore"


def get_plugin_config() -> ArkGuesserScoped:
    return nonebot.get_plugin_config(ArkGuesserConfig).arkguesser
