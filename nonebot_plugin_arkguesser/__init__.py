from nonebot import on_message, get_driver, require
from nonebot.plugin import PluginMetadata, inherit_supported_adapters
from nonebot.adapters import Event
from nonebot.matcher import Matcher
from nonebot.rule import Rule
from nonebot.exception import FinishedException
require("nonebot_plugin_alconna")
require("nonebot_plugin_uninfo")
require("nonebot_plugin_htmlrender")
require("nonebot_plugin_localstore")

from nonebot_plugin_uninfo import Uninfo
from nonebot_plugin_alconna import UniMessage, Image, on_alconna, Args
from arclet.alconna import Alconna, Option
from .config import Config

from .game import OperatorGuesser
from .render import render_guess_result, render_correct_answer
from .pool_manager import pool_manager
from .mode_manager import mode_manager
from .continuous_manager import ContinuousManager

# 创建管理器实例
pool_manager = pool_manager
mode_manager = mode_manager
continuous_manager = ContinuousManager()

# 导出插件元数据，确保 NoneBot 能正确识别
__all__ = ["__plugin_meta__"]

__plugin_meta__ = PluginMetadata(
    name="明日方舟猜干员游戏",
    description="明日方舟猜干员游戏 - 支持多种游戏模式和题库设置",
    usage="""🎮 游戏指令:
arkstart - 开始游戏
结束 - 结束游戏
直接输入干员名即可开始猜测

📚 题库设置:
/arkstart 题库 6 - 设置题库为6星干员
/arkstart 题库 4-6 - 设置题库为4-6星干员
/arkstart 题库 查看 - 查看当前题库设置
/arkstart 题库 重置 - 重置为默认设置

🎭 模式设置:
/arkstart 模式 大头 - 设置为大头模式
/arkstart 模式 兔头 - 设置为兔头模式
/arkstart 模式 查看 - 查看当前模式设置
/arkstart 模式 重置 - 重置为默认模式

🔄 连战模式设置:
/arkstart 连战 开启 - 开启连战模式
/arkstart 连战 关闭 - 关闭连战模式
/arkstart 连战 查看 - 查看当前连战模式设置
/arkstart 连战 重置 - 重置为默认连战模式设置""",
    type="application",
    homepage="https://github.com/lizhiqi233-rgb/nonebot-plugin-arkguesser",
    config=Config,
    supported_adapters={"~onebot.v11", "~onebot.v12", "~telegram", "~discord"},
)
game = OperatorGuesser()
driver = get_driver()

def is_playing() -> Rule:
    async def _checker(uninfo: Uninfo) -> bool:
        return bool(game.get_game(uninfo))
    return Rule(_checker)

# 简化 Alconna 结构，使用 Option 而不是 Subcommand
start_cmd = on_alconna(
    Alconna(
        "arkstart",
        Option("题库", Args["range_str;?", str]),
        Option("模式", Args["mode;?", str]),
        Option("连战", Args["action;?", str])
    ),
    aliases={"明日方舟开始"}
)

guess_matcher = on_message(rule=is_playing(), priority=15, block=False)

@start_cmd.handle()
async def handle_start(uninfo: Uninfo, matcher: Matcher, event: Event):
    try:
        # 获取原始消息文本
        message_text = str(event.get_message()) if hasattr(event, 'get_message') else ""
        
        # 检查是否是设置子命令
        if "题库" in message_text:
            await handle_pool_settings(uninfo, matcher, message_text)
            return
        
        if "模式" in message_text:
            await handle_mode_settings(uninfo, matcher, message_text)
            return
        
        if "连战" in message_text:
            await handle_continuous_settings(uninfo, matcher, message_text)
            return
        
        # 处理开始游戏
        if game.get_game(uninfo):
            await matcher.send("🎮 游戏已在进行中！\n💬 请继续猜测或输入「结束」来结束游戏")
            return
        
        game_data = game.start_new_game(uninfo)
        
        # 获取连战模式设置
        user_id = str(uninfo.user.id) if uninfo.user else None
        group_id = str(uninfo.group.id) if uninfo.group else None
        continuous_enabled = continuous_manager.get_continuous_mode(user_id, group_id)
        
        # 设置连战模式状态
        game_data["continuous_mode"] = continuous_enabled
        
        # 显示游戏开始信息，包含当前题库范围和模式
        allowed_rarities = game_data.get("allowed_rarities", [6])
        current_mode = game_data.get("current_mode", "大头")
        
        if len(allowed_rarities) == 1:
            range_display = f"{allowed_rarities[0]}星"
        else:
            range_display = f"{min(allowed_rarities)}-{max(allowed_rarities)}星"
        
        mode_description = mode_manager._get_mode_description(current_mode)
        
        # 构建简洁的游戏开始提示
        start_msg = f"🎮 游戏开始！\n"
        start_msg += f"📚 {range_display} | 🎭 {current_mode}"
        
        # 添加连战模式状态
        if continuous_enabled:
            start_msg += f" | 🔄 连战"
        
        start_msg += f"\n🎯 {game.max_attempts}次机会 | 💬 直接输入干员名"
        
        await matcher.send(start_msg)
    
    except FinishedException:
        # FinishedException 是正常的流程控制，直接返回
        return
    except Exception as e:
        # 添加错误处理
        import traceback
        error_msg = f"处理命令时出错: {str(e)}\n{traceback.format_exc()}"
        await matcher.send(f"❌ 插件运行出错，请检查日志: {str(e)}")

async def handle_continuous_settings(uninfo: Uninfo, matcher: Matcher, message_text: str):
    """统一的连战模式设置处理"""
    try:
        user_id = str(uninfo.user.id) if uninfo.user else None
        group_id = str(uninfo.group.id) if uninfo.group else None
        
        # 解析子命令
        if "查看" in message_text:
            info = continuous_manager.get_continuous_info(user_id, group_id)
            msg = f"🔄 当前连战模式设置\n"
            msg += f"状态：{info['status']}\n"
            msg += f"描述：{info['description']}\n"
            msg += f"设置来源：{info['source']}"
            
            # 添加当前连战统计信息
            current_game = game.get_game(uninfo)
            if current_game and current_game.get("continuous_mode", False):
                continuous_count = game.get_continuous_count(uninfo)
                if continuous_count > 0:
                    msg += f"\n\n📊 当前连战统计\n"
                    msg += f"连战轮数：{continuous_count}轮\n"
                    msg += f"剩余尝试：{game.max_attempts - len(current_game['guesses'])}次"
            
            await matcher.send(msg)
            return
        
        elif "重置" in message_text:
            reset_result = continuous_manager.reset_continuous_mode(user_id, group_id)
            if reset_result["success"]:
                msg = f"✅ 连战模式已重置\n"
                msg += f"当前状态：{reset_result['status']}\n"
                msg += f"作用范围：{reset_result['scope']}"
                await matcher.send(msg)
            else:
                await matcher.send(f"❌ 重置失败：{reset_result['message']}")
            return
        
        # 处理设置连战模式
        elif "开启" in message_text:
            set_result = continuous_manager.set_continuous_mode(True, user_id, group_id)
            if set_result["success"]:
                msg = f"✅ 连战模式已开启\n"
                msg += f"作用范围：{set_result['scope']}\n"
                msg += f"💡 猜对后会自动开始下一轮，无需重新输入开始指令"
                await matcher.send(msg)
            else:
                await matcher.send(f"❌ 开启失败：{set_result['message']}")
            return
        
        elif "关闭" in message_text:
            set_result = continuous_manager.set_continuous_mode(False, user_id, group_id)
            if set_result["success"]:
                msg = f"⏹️ 连战模式已关闭\n"
                msg += f"作用范围：{set_result['scope']}\n"
                msg += f"💡 猜对后游戏结束，需要重新输入开始指令"
                await matcher.send(msg)
            else:
                await matcher.send(f"❌ 关闭失败：{set_result['message']}")
            return
        
        # 如果没有提供参数，显示当前设置和帮助
        else:
            info = continuous_manager.get_continuous_info(user_id, group_id)
            msg = f"🔄 当前连战模式设置\n"
            msg += f"状态：{info['status']}\n"
            msg += f"描述：{info['description']}\n"
            msg += f"设置来源：{info['source']}\n\n"
            msg += f"💡 连战模式说明：\n"
            msg += f"🔄 开启：猜对后自动开始下一轮，无需重新输入开始指令\n"
            msg += f"⏹️ 关闭：猜对后游戏结束，需要重新输入开始指令\n\n"
            msg += f"🔧 使用方法：\n"
            msg += f"/arkstart 连战 开启 - 开启连战模式\n"
            msg += f"/arkstart 连战 关闭 - 关闭连战模式\n"
            msg += f"/arkstart 连战 查看 - 查看当前设置\n"
            msg += f"/arkstart 连战 重置 - 重置为默认设置\n\n"
            msg += f"💡 提示：连战模式设置会影响游戏体验"
            await matcher.send(msg)
    
    except FinishedException:
        return
    except Exception as e:
        import traceback
        error_msg = f"处理连战模式设置时出错: {str(e)}\n{traceback.format_exc()}"
        await matcher.send(f"❌ 连战模式设置出错，请检查日志: {str(e)}")

async def handle_pool_settings(uninfo: Uninfo, matcher: Matcher, message_text: str):
    """统一的题库设置处理"""
    try:
        user_id = str(uninfo.user.id) if uninfo.user else None
        group_id = str(uninfo.group.id) if uninfo.group else None
        
        # 解析子命令
        if "查看" in message_text:
            info = pool_manager.get_pool_info(user_id, group_id)
            msg = f"📚 当前题库设置\n"
            msg += f"星级范围：{info['range_display']}星\n"
            msg += f"可选干员：{info['operator_count']}个\n"
            msg += f"设置来源：{info['source']}"
            await matcher.send(msg)
            return
        
        elif "重置" in message_text:
            reset_result = pool_manager.reset_pool_range(user_id, group_id)
            if reset_result["success"]:
                msg = f"✅ 题库已重置\n"
                msg += f"星级范围：{reset_result['range_str']}星\n"
                msg += f"可选干员：{reset_result['operator_count']}个\n"
                msg += f"作用范围：{reset_result['scope']}"
                await matcher.send(msg)
            else:
                await matcher.send("❌ 重置失败")
            return
        
        # 处理设置星级范围
        else:
            # 从消息中提取参数
            import re
            range_match = re.search(r'题库\s+([0-9]+(?:-[0-9]+)?)', message_text)
            range_str = range_match.group(1) if range_match else None
            
            # 如果没有提供参数，显示当前设置和帮助
            if not range_str:
                info = pool_manager.get_pool_info(user_id, group_id)
                msg = f"📚 当前题库设置\n"
                msg += f"星级范围：{info['range_display']}星\n"
                msg += f"可选干员：{info['operator_count']}个\n"
                msg += f"设置来源：{info['source']}\n\n"
                msg += f"💡 题库说明：\n"
                msg += f"• 6星：仅包含6星干员，难度较高\n"
                msg += f"• 4-6星：包含4-6星干员，难度适中\n"
                msg += f"• 1-6星：包含所有星级，难度较低\n\n"
                msg += f"🔧 使用方法：\n"
                msg += f"/arkstart 题库 6 - 设置为6星\n"
                msg += f"/arkstart 题库 4-6 - 设置为4-6星\n"
                msg += f"/arkstart 题库 查看 - 查看当前设置\n"
                msg += f"/arkstart 题库 重置 - 重置为默认设置"
                await matcher.send(msg)
                return
            
            # 设置新的星级范围
            set_result = pool_manager.set_pool_range(user_id, group_id, range_str)
            if set_result["success"]:
                rarity_display = f"{min(set_result['rarity_list'])}-{max(set_result['rarity_list'])}" if len(set_result['rarity_list']) > 1 else str(set_result['rarity_list'][0])
                msg = f"✅ 题库设置成功\n"
                msg += f"星级范围：{rarity_display}星\n"
                msg += f"可选干员：{set_result['operator_count']}个\n"
                msg += f"作用范围：{set_result['scope']}"
                
                # 如果是群聊设置，添加说明
                if group_id:
                    msg += f"\n💡 群聊题库已更新，对本群所有成员生效"
                
                await matcher.send(msg)
            else:
                msg = f"❌ 设置失败\n"
                msg += f"错误：{set_result['error']}\n\n"
                msg += f"💡 正确格式：\n"
                msg += f"6 - 仅6星干员\n"
                msg += f"5-6 - 5至6星干员\n"
                msg += f"1-6 - 全部星级"
                await matcher.send(msg)
    
    except FinishedException:
        return
    except Exception as e:
        import traceback
        error_msg = f"处理题库设置时出错: {str(e)}\n{traceback.format_exc()}"
        await matcher.send(f"❌ 题库设置出错，请检查日志: {str(e)}")

async def handle_mode_settings(uninfo: Uninfo, matcher: Matcher, message_text: str):
    """统一的模式设置处理"""
    try:
        user_id = str(uninfo.user.id) if uninfo.user else None
        group_id = str(uninfo.group.id) if uninfo.group else None
        
        # 解析子命令
        if "查看" in message_text:
            info = mode_manager.get_mode_info(user_id, group_id)
            msg = f"🎭 当前模式设置\n"
            msg += f"模式：{info['mode']}\n"
            msg += f"描述：{info['description']}\n"
            msg += f"设置来源：{info['source']}"
            await matcher.send(msg)
            return
        
        elif "重置" in message_text:
            reset_result = mode_manager.reset_mode(user_id, group_id)
            if reset_result["success"]:
                msg = f"✅ 模式已重置\n"
                msg += f"当前模式：{reset_result['mode']}\n"
                msg += f"作用范围：{reset_result['scope']}"
                await matcher.send(msg)
            else:
                await matcher.send(f"❌ 重置失败：{reset_result['message']}")
            return
        
        # 处理设置模式
        else:
            # 从消息中提取参数
            import re
            mode_match = re.search(r'模式\s+(兔头|大头)', message_text)
            mode = mode_match.group(1) if mode_match else None
            
            # 如果没有提供参数，显示当前设置和帮助
            if not mode:
                info = mode_manager.get_mode_info(user_id, group_id)
                msg = f"🎭 当前模式设置\n"
                msg += f"模式：{info['mode']}\n"
                msg += f"描述：{info['description']}\n"
                msg += f"设置来源：{info['source']}\n\n"
                msg += f"💡 模式说明：\n"
                msg += f"🐰 兔头模式：增加游戏趣味性\n"
                msg += f"👤 大头模式：适合正常游戏体验\n\n"
                msg += f"🔧 使用方法：\n"
                msg += f"/arkstart 模式 大头 - 设置为大头模式\n"
                msg += f"/arkstart 模式 兔头 - 设置为兔头模式\n"
                msg += f"/arkstart 模式 查看 - 查看当前设置\n"
                msg += f"/arkstart 模式 重置 - 重置为默认模式\n\n"
                msg += f"💡 提示：模式设置会影响游戏体验"
                await matcher.send(msg)
                return
            
            # 设置新的模式
            set_result = mode_manager.set_mode(mode, user_id, group_id)
            if set_result["success"]:
                msg = f"✅ 模式设置成功\n"
                msg += f"模式：{set_result['mode']}\n"
                msg += f"作用范围：{set_result['scope']}\n"
                msg += f"描述：{mode_manager._get_mode_description(set_result['mode'])}\n\n"
                
                # 添加模式切换说明
                if set_result['mode'] == "兔头":
                    msg += f"🐰 已切换到兔头模式\n"
                    msg += f"💡 下次开始游戏时将使用兔头模式\n"
                    msg += f"🎨 兔头模式增加了游戏的趣味性"
                else:
                    msg += f"👤 已切换到大头模式\n"
                    msg += f"💡 下次开始游戏时将使用大头模式\n"
                    msg += f"🎯 大头模式适合正常的游戏体验"
                
                if set_result['mode'] == "兔头":
                    # 兔头模式：发送消息和图片
                    from pathlib import Path
                    image_path = Path(__file__).parent / "resources" / "images" / "xlpj" / "血狼破军_B站头像.webp"
                    if image_path.exists():
                        # 读取图片文件内容
                        with open(image_path, 'rb') as f:
                            image_data = f.read()
                        await UniMessage([
                            msg,
                            Image(raw=image_data)
                        ]).send()
                    else:
                        # 如果图片不存在，只发送消息
                        await matcher.send(msg)
                else:
                    # 大头模式：只发送消息
                    await matcher.send(msg)
            else:
                await matcher.send(f"❌ 设置失败：{set_result['message']}")
    
    except FinishedException:
        return
    except Exception as e:
        import traceback
        error_msg = f"处理模式设置时出错: {str(e)}\n{traceback.format_exc()}"
        await matcher.send(f"❌ 模式设置出错，请检查日志: {str(e)}")

async def handle_end(uninfo: Uninfo):
    game_data = game.get_game(uninfo)
    operator = game_data["operator"]
    current_mode = game_data.get("current_mode", "大头")
    game.end_game(uninfo)
    img = await render_correct_answer(operator, current_mode)
    await UniMessage(Image(raw=img)).send()

@guess_matcher.handle()
async def handle_guess(uninfo: Uninfo, event: Event):
    guess_name = event.get_plaintext().strip()
    if guess_name in ("", "结束", "arkstart"):
        if guess_name == "结束":
            # 检查是否在连战模式中
            game_data = game.get_game(uninfo)
            if game_data and game_data.get("continuous_mode", False):
                continuous_count = game.get_continuous_count(uninfo)
                if continuous_count > 0:
                    # 连战模式退出
                    operator = game_data["operator"]
                    current_mode = game_data.get("current_mode", "大头")
                    game.end_game(uninfo)
                    img = await render_correct_answer(operator, current_mode)
                    await UniMessage([
                        f"🔄 连战模式已退出\n🎯 正确答案：",
                        Image(raw=img),
                        f"\n📊 本次连战共完成{continuous_count}轮"
                    ]).send()
                else:
                    # 普通游戏结束
                    await handle_end(uninfo)
            else:
                # 普通游戏结束
                await handle_end(uninfo)
        return
    # 检查游戏状态
    game_data = game.get_game(uninfo)
    if not game_data:
        return
    # 检查重复猜测
    if any(g["name"] == guess_name for g in game_data["guesses"]):
        await UniMessage.text(f"🤔 已经猜过【{guess_name}】了，请尝试其他干员！").send()
        return
        
    correct, guessed, comparison = game.guess(uninfo, guess_name)
    
    if correct:
        # 检查连战模式
        continuous_mode = game_data.get("continuous_mode", False)
        
        if continuous_mode:
            # 连战模式：自动开始新游戏
            # 更新连战计数
            continuous_count = game.update_continuous_count(uninfo)
            
            # 结束当前游戏
            game.end_game(uninfo)
            
            # 开始新游戏
            new_game = game.start_new_game(uninfo)
            
            # 显示答案并提示连战模式
            current_mode = game_data.get("current_mode", "大头")
            img = await render_correct_answer(guessed, current_mode)
            
            # 构建连战模式提示消息
            continuous_msg = f"🎉 恭喜你猜对了！\n🎯 正确答案："
            
            if continuous_count > 1:
                continuous_msg += f"\n🔄 连战进度：第{continuous_count}轮"
            else:
                continuous_msg += f"\n🔄 连战模式已启动"
            
            continuous_msg += f"\n💡 直接输入干员名即可开始下一轮猜测"
            continuous_msg += f"\n⏹️ 输入「结束」可退出连战模式"
            
            await UniMessage([
                continuous_msg,
                Image(raw=img)
            ]).send()
        else:
            # 普通模式：正常结束
            game.end_game(uninfo)
            current_mode = game_data.get("current_mode", "大头")
            img = await render_correct_answer(guessed, current_mode)
            await UniMessage([
                "🎉 恭喜你猜对了！\n🎯 正确答案：",
                Image(raw=img)
            ]).send()
        return
    
    if not guessed:
        similar = game.find_similar_operators(guess_name)
        if not similar:
            return
        err_msg = f"❓ 未找到干员【{guess_name}】！\n💡 尝试以下相似结果：" + "、".join(similar)
        await UniMessage.text(err_msg).send()
        return

    attempts_left = game.max_attempts - len(game_data["guesses"])
    # 检查尝试次数
    if attempts_left <= 0:
        operator = game_data["operator"]
        current_mode = game_data.get("current_mode", "大头")
        game.end_game(uninfo)
        img = await render_correct_answer(operator, current_mode)
        await UniMessage([
            "😅 尝试次数已用尽！\n🎯 正确答案：",
            Image(raw=img)
        ]).send()
        return
    
    current_mode = game_data.get("current_mode", "大头")
    img = await render_guess_result(guessed, comparison, attempts_left, current_mode)
    
    # 添加连战模式进度显示
    if game.is_continuous_mode(uninfo):
        continuous_count = game.get_continuous_count(uninfo)
        if continuous_count > 0:
            # 在图片下方添加连战进度提示
            progress_msg = f"\n🔄 连战进度：第{continuous_count}轮 | 剩余尝试：{attempts_left}次"
            await UniMessage([
                Image(raw=img),
                progress_msg
            ]).send()
        else:
            await UniMessage(Image(raw=img)).send()
    else:
        await UniMessage(Image(raw=img)).send()
    