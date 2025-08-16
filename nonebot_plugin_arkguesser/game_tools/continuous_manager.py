"""
连战模式管理器
负责管理连战模式的开启/关闭状态
"""

import json
from typing import Dict, Any
from nonebot import require

# 加载 localstore 插件
require("nonebot_plugin_localstore")
import nonebot_plugin_localstore as store

class ContinuousManager:
    """连战模式管理器"""
    
    def __init__(self):
        self.data_file = store.get_plugin_data_file("continuous_settings.json")
        self._load_settings()
    
    def _load_settings(self):
        """加载设置文件"""
        try:
            if self.data_file.exists():
                self.settings = json.loads(self.data_file.read_text(encoding='utf-8'))
            else:
                self.settings = {}
            
            # 确保默认值存在
            if "default" not in self.settings:
                self.settings["default"] = False
            if "users" not in self.settings:
                self.settings["users"] = {}
            if "groups" not in self.settings:
                self.settings["groups"] = {}
                
        except Exception as e:
            print(f"加载连战模式设置失败: {e}")
            self.settings = {"default": False, "users": {}, "groups": {}}
    
    def _save_settings(self):
        """保存设置文件"""
        try:
            self.data_file.write_text(json.dumps(self.settings, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception as e:
            print(f"保存连战模式设置失败: {e}")
    
    def get_continuous_mode(self, user_id: str = None, group_id: str = None) -> bool:
        """获取当前连战模式设置"""
        # 优先级：群聊设置 > 个人设置 > 默认设置
        if group_id and group_id in self.settings.get("groups", {}):
            return self.settings["groups"][group_id]
        
        if user_id and user_id in self.settings.get("users", {}):
            return self.settings["users"][user_id]
        
        return self.settings.get("default", False)
    
    def set_continuous_mode(self, enabled: bool, user_id: str = None, group_id: str = None) -> Dict[str, Any]:
        """设置连战模式"""
        try:
            if group_id:
                # 群聊设置
                if "groups" not in self.settings:
                    self.settings["groups"] = {}
                self.settings["groups"][group_id] = enabled
                scope = "群聊"
            elif user_id:
                # 个人设置
                if "users" not in self.settings:
                    self.settings["users"] = {}
                self.settings["users"][user_id] = enabled
                scope = "个人"
            else:
                # 默认设置
                self.settings["default"] = enabled
                scope = "全局"
            
            self._save_settings()
            
            status = "开启" if enabled else "关闭"
            return {
                "success": True,
                "enabled": enabled,
                "status": status,
                "scope": scope,
                "message": f"连战模式已{status}"
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"设置失败：{str(e)}"
            }
    
    def reset_continuous_mode(self, user_id: str = None, group_id: str = None) -> Dict[str, Any]:
        """重置连战模式设置"""
        try:
            if group_id and "groups" in self.settings:
                if group_id in self.settings["groups"]:
                    del self.settings["groups"][group_id]
                    scope = "群聊"
                else:
                    return {
                        "success": False,
                        "message": "群聊未设置自定义连战模式"
                    }
            elif user_id and "users" in self.settings:
                if user_id in self.settings["users"]:
                    del self.settings["users"][user_id]
                    scope = "个人"
                else:
                    return {
                        "success": False,
                        "message": "个人未设置自定义连战模式"
                    }
            else:
                return {
                    "success": False,
                    "message": "未设置自定义连战模式"
                }
            
            self._save_settings()
            
            return {
                "success": True,
                "enabled": self.settings.get("default", False),
                "status": "开启" if self.settings.get("default", False) else "关闭",
                "scope": scope,
                "message": f"连战模式已重置为默认设置"
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"重置失败：{str(e)}"
            }
    
    def get_continuous_info(self, user_id: str = None, group_id: str = None) -> Dict[str, Any]:
        """获取连战模式信息"""
        enabled = self.get_continuous_mode(user_id, group_id)
        
        if group_id and group_id in self.settings.get("groups", {}):
            source = "群聊设置"
        elif user_id and user_id in self.settings.get("users", {}):
            source = "个人设置"
        else:
            source = "默认设置"
        
        return {
            "enabled": enabled,
            "status": "开启" if enabled else "关闭",
            "source": source,
            "description": self._get_continuous_description(enabled)
        }
    
    def _get_continuous_description(self, enabled: bool) -> str:
        """获取连战模式描述"""
        if enabled:
            return "猜对后自动开始下一轮，无需重新输入开始指令"
        else:
            return "猜对后游戏结束，需要重新输入开始指令"
