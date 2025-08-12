from pydantic import BaseModel, Field

class Config(BaseModel):
    """插件配置类"""
    
    # 最大尝试次数
    arkguesser_max_attempts: int = Field(default=10, description="最大尝试次数")
    
    # 默认星级范围
    arkguesser_default_rarity_range: str = Field(default="6", description="默认星级范围")
    
    # 默认游戏模式
    arkguesser_default_mode: str = Field(default="大头", description="默认游戏模式")
    
    class Config:
        extra = "ignore"