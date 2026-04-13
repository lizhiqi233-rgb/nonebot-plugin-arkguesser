from nonebot import on_message, require
from nonebot.plugin import PluginMetadata, inherit_supported_adapters
from nonebot.adapters import Bot, Event
from nonebot.matcher import Matcher
from nonebot.params import Depends
from nonebot.permission import SuperUser
from nonebot.rule import Rule
from nonebot.exception import FinishedException
require("nonebot_plugin_alconna")
require("nonebot_plugin_uninfo")
require("nonebot_plugin_htmlrender")
require("nonebot_plugin_localstore")

from nonebot_plugin_uninfo import Uninfo
from nonebot_plugin_alconna import UniMessage, Image, on_alconna, Args
from arclet.alconna import Alconna, Arparma, Option
from .config import ArkGuesserConfig

from .game import OperatorGuesser
from .render import render_guess_result, render_correct_answer
from .pool_manager import pool_manager
from .mode_manager import mode_manager
from .continuous_manager import ContinuousManager
import csv
import nonebot_plugin_localstore as store

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
/arkstart 模式 查看 - 查看当前模式设置
/arkstart 模式 重置 - 重置为默认模式

🔄 连战模式设置:
/arkstart 连战 开启 - 开启连战模式
/arkstart 连战 关闭 - 关闭连战模式
/arkstart 连战 查看 - 查看当前连战模式设置
/arkstart 连战 重置 - 重置为默认连战模式设置

🏷️ 干员别称:
/arkstart 别称 干员名 别称 - 为指定干员添加一个别称（存储在 aliases.csv）""",
    type="application",
    homepage="https://github.com/lizhiqi233-rgb/nonebot-plugin-arkguesser",
    config=ArkGuesserConfig,
    supported_adapters=inherit_supported_adapters(
        "nonebot_plugin_alconna",
        "nonebot_plugin_uninfo",
        "nonebot_plugin_htmlrender",
        "nonebot_plugin_localstore",
    ),
)

# 延迟创建游戏实例，避免在导入时调用配置
_game_instance = None

def get_game_instance():
    """获取游戏实例，延迟初始化"""
    global _game_instance
    if _game_instance is None:
        _game_instance = OperatorGuesser()
    return _game_instance


_su = SuperUser()


async def _superuser_dep(bot: Bot, event: Event) -> bool:
    """`nonebot.permission.SuperUser`：兼容纯 user_id 与配置中的 `adapter:user_id`。"""
    return await _su(bot, event)


def is_playing() -> Rule:
    async def _checker(uninfo: Uninfo) -> bool:
        return bool(get_game_instance().get_game(uninfo))
    return Rule(_checker)

# 简化 Alconna 结构，使用 Option 而不是 Subcommand
start_cmd = on_alconna(
    Alconna(
        "arkstart",
        Option("题库", Args["range_str;?", str]),
        Option("模式", Args["mode;?", str]),
        Option("连战", Args["action;?", str]),
        Option("更新", Args["update_type;?", str]),
        Option("别称", Args["operator_name", str]["alias", str]),
    ),
    aliases={"明日方舟开始"}
)

guess_matcher = on_message(rule=is_playing(), priority=15, block=False)


def _alc_arg_str(arp: Arparma, name: str) -> str | None:
    """从 Alconna 解析结果取出字符串参数（去空白，空串视为未提供）。"""
    raw = arp.all_matched_args.get(name)
    if raw is None:
        return None
    s = raw.strip() if isinstance(raw, str) else str(raw).strip()
    return s or None


@start_cmd.assign("题库")
async def _dispatch_pool(uninfo: Uninfo, matcher: Matcher, result: Arparma):
    await handle_pool_settings(uninfo, matcher, _alc_arg_str(result, "range_str"))


@start_cmd.assign("模式")
async def _dispatch_mode(uninfo: Uninfo, matcher: Matcher, result: Arparma):
    await handle_mode_settings(uninfo, matcher, _alc_arg_str(result, "mode"))


@start_cmd.assign("连战")
async def _dispatch_continuous(uninfo: Uninfo, matcher: Matcher, result: Arparma):
    await handle_continuous_settings(uninfo, matcher, _alc_arg_str(result, "action"))


@start_cmd.assign("更新")
async def _dispatch_update(
    uninfo: Uninfo,
    matcher: Matcher,
    is_superuser: bool = Depends(_superuser_dep),
):
    if not is_superuser:
        await matcher.send("⛔ 无权限：仅 NoneBot 管理员可使用「arkstart 更新」")
        return
    await handle_update_resources(uninfo, matcher)


@start_cmd.assign("别称")
async def _dispatch_alias(
    uninfo: Uninfo,
    matcher: Matcher,
    result: Arparma,
    is_superuser: bool = Depends(_superuser_dep),
):
    if not is_superuser:
        await matcher.send("⛔ 无权限：仅 NoneBot 管理员可使用「arkstart 别称」")
        return
    await handle_alias_settings(
        uninfo,
        matcher,
        _alc_arg_str(result, "operator_name"),
        _alc_arg_str(result, "alias"),
    )


@start_cmd.assign("$main")
async def handle_start_main(uninfo: Uninfo, matcher: Matcher):
    """仅匹配「arkstart / 明日方舟开始」主命令，无子选项时开局。"""
    try:
        if get_game_instance().get_game(uninfo):
            await matcher.send("🎮 游戏已在进行中！\n💬 请继续猜测或输入「结束」来结束游戏")
            return

        game_data = get_game_instance().start_new_game(uninfo)

        user_id = str(uninfo.user.id) if uninfo.user else None
        group_id = str(uninfo.group.id) if uninfo.group else None
        continuous_enabled = continuous_manager.get_continuous_mode(user_id, group_id)

        game_data["continuous_mode"] = continuous_enabled

        allowed_rarities = game_data.get("allowed_rarities", [6])
        current_mode = game_data.get("current_mode", "大头")

        if len(allowed_rarities) == 1:
            range_display = f"{allowed_rarities[0]}星"
        else:
            range_display = f"{min(allowed_rarities)}-{max(allowed_rarities)}星"

        start_msg = f"🎮 游戏开始！\n"
        start_msg += f"📚 {range_display} | 🎭 {current_mode}"

        if continuous_enabled:
            start_msg += f" | 🔄 连战"

        start_msg += f"\n🎯 {get_game_instance().max_attempts}次机会 | 💬 直接输入干员名"

        await matcher.send(start_msg)

    except FinishedException:
        return
    except Exception as e:
        await matcher.send(f"❌ 插件运行出错，请检查日志: {str(e)}")

async def handle_alias_settings(
    uninfo: Uninfo,
    matcher: Matcher,
    operator_name: str | None,
    alias: str | None,
):
    """
    添加干员别称

    用法：arkstart 别称 <干员名> <别称>
    - 需先存在 aliases.csv，且第一列 operator_name 中能找到该干员名
    - 同一干员多个别称存储在同一单元格内，使用 '|' 分隔
    """
    import re

    if not operator_name or not alias:
        await matcher.send("🏷️ 用法：arkstart 别称 干员名 别称\n示例：arkstart 别称 艾雅法拉 小羊")
        return

    if operator_name == alias:
        await matcher.send("❌ 别称不能与干员名相同")
        return

    aliases_file = store.get_plugin_data_file("aliases.csv")
    if not aliases_file.exists():
        await matcher.send("❌ 未找到 aliases.csv\n💡 请先执行：arkstart 更新\n（更新数据库时会自动生成 aliases.csv）")
        return

    # 读取整表
    rows: list[list[str]] = []
    header_ok = False
    operator_row_idx: int | None = None
    existing_aliases_cell = ""

    try:
        with aliases_file.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            for idx, row in enumerate(reader):
                rows.append(row)
                if idx == 0 and row and row[0].strip() == "operator_name":
                    header_ok = True
                    continue
                if idx == 0 and not header_ok:
                    # 无表头也允许，但仍按两列处理
                    pass
                if row and row[0].strip() == operator_name:
                    operator_row_idx = idx
                    existing_aliases_cell = row[1].strip() if len(row) >= 2 else ""
    except Exception as e:
        await matcher.send(f"❌ 读取 aliases.csv 失败：{str(e)}")
        return

    # 必须在 aliases.csv 里存在该干员名
    if operator_row_idx is None:
        await matcher.send(f"❌ aliases.csv 中未找到干员【{operator_name}】\n💡 请确认干员名与数据库一致，或先执行：arkstart 更新")
        return

    # 解析/合并别称（支持已有内容用多种分隔符；写回统一用 |）
    def _split_aliases(cell: str) -> list[str]:
        if not cell:
            return []
        parts = re.split(r"[|,，;；/\\\s]+", cell.strip())
        return [p for p in (x.strip() for x in parts) if p]

    existing_list = _split_aliases(existing_aliases_cell)
    if alias in existing_list:
        await matcher.send(f"✅ 干员【{operator_name}】已包含别称【{alias}】")
        return

    new_list = existing_list + [alias]
    new_cell = "|".join(new_list)

    # 写回该行（保证至少两列）
    while len(rows[operator_row_idx]) < 2:
        rows[operator_row_idx].append("")
    rows[operator_row_idx][1] = new_cell

    # 若没有表头，补上（尽量不破坏你已有文件结构：只有在完全空文件时补）
    if not rows:
        rows = [["operator_name", "aliases"], [operator_name, new_cell]]
    elif not header_ok and rows[0] and rows[0][0].strip() != "operator_name":
        # 保守：不强行插入表头，避免改变用户已有格式
        pass

    try:
        # 原子写入：避免写入过程中异常导致文件被截断
        tmp_file = aliases_file.with_suffix(aliases_file.suffix + ".tmp")
        with tmp_file.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        tmp_file.replace(aliases_file)
    except Exception as e:
        await matcher.send(f"❌ 写入 aliases.csv 失败：{str(e)}")
        return

    # 写入成功后立刻刷新内存别称映射，不影响正在进行的游戏
    try:
        get_game_instance().reload_aliases()
    except Exception:
        # 刷新失败不影响写入结果；用户仍可通过重载/重启生效
        pass

    await matcher.send(f"✅ 已为干员【{operator_name}】添加别称【{alias}】\n当前别称：{new_cell}")

async def handle_continuous_settings(
    uninfo: Uninfo, matcher: Matcher, action: str | None
):
    """统一的连战模式设置处理（参数来自 Alconna Option「连战」的 action）。"""
    try:
        user_id = str(uninfo.user.id) if uninfo.user else None
        group_id = str(uninfo.group.id) if uninfo.group else None
        arg = (action or "").strip()

        if arg == "查看":
            info = continuous_manager.get_continuous_info(user_id, group_id)
            msg = f"🔄 当前连战模式设置\n"
            msg += f"状态：{info['status']}\n"
            msg += f"描述：{info['description']}\n"
            msg += f"设置来源：{info['source']}"
            
            # 添加当前连战统计信息
            current_game = get_game_instance().get_game(uninfo)
            if current_game and current_game.get("continuous_mode", False):
                continuous_count = get_game_instance().get_continuous_count(uninfo)
                if continuous_count > 0:
                    msg += f"\n\n📊 当前连战统计\n"
                    msg += f"连战轮数：{continuous_count}轮\n"
                    msg += f"剩余尝试：{get_game_instance().max_attempts - len(current_game['guesses'])}次"
            
            await matcher.send(msg)
            return

        elif arg == "重置":
            reset_result = continuous_manager.reset_continuous_mode(user_id, group_id)
            if reset_result["success"]:
                msg = f"✅ 连战模式已重置\n"
                msg += f"当前状态：{reset_result['status']}\n"
                msg += f"作用范围：{reset_result['scope']}"
                await matcher.send(msg)
            else:
                await matcher.send(f"❌ 重置失败：{reset_result['message']}")
            return

        elif arg == "开启":
            set_result = continuous_manager.set_continuous_mode(True, user_id, group_id)
            if set_result["success"]:
                msg = f"✅ 连战模式已开启\n"
                msg += f"作用范围：{set_result['scope']}\n"
                msg += f"💡 猜对后会自动开始下一轮，无需重新输入开始指令"
                await matcher.send(msg)
            else:
                await matcher.send(f"❌ 开启失败：{set_result['message']}")
            return

        elif arg == "关闭":
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
            msg += f"[arkstart 连战 开启] - 开启连战模式\n"
            msg += f"[arkstart 连战 关闭] - 关闭连战模式\n"
            msg += f"[arkstart 连战 查看] - 查看当前设置\n"
            msg += f"[arkstart 连战 重置] - 重置为默认设置\n\n"
            msg += f"💡 提示：连战模式设置会影响游戏体验"
            await matcher.send(msg)
    
    except FinishedException:
        return
    except Exception as e:
        await matcher.send(f"❌ 连战模式设置出错，请检查日志: {str(e)}")

async def handle_pool_settings(
    uninfo: Uninfo, matcher: Matcher, range_str: str | None
):
    """统一的题库设置处理（参数来自 Alconna Option「题库」的 range_str）。"""
    try:
        user_id = str(uninfo.user.id) if uninfo.user else None
        group_id = str(uninfo.group.id) if uninfo.group else None
        arg = (range_str or "").strip()

        if arg == "查看":
            info = pool_manager.get_pool_info(user_id, group_id)
            msg = f"📚 当前题库设置\n"
            msg += f"星级范围：{info['range_display']}星\n"
            msg += f"可选干员：{info['operator_count']}个\n"
            msg += f"设置来源：{info['source']}"
            await matcher.send(msg)
            return

        elif arg == "重置":
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

        if not arg:
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
            msg += f"[arkstart 题库 6] - 设置为6星\n"
            msg += f"[arkstart 题库 4-6] - 设置为4-6星\n"
            msg += f"[arkstart 题库 查看] - 查看当前设置\n"
            msg += f"[arkstart 题库 重置] - 重置为默认设置"
            await matcher.send(msg)
            return

        set_result = pool_manager.set_pool_range(user_id, group_id, arg)
        if set_result["success"]:
            rarity_display = f"{min(set_result['rarity_list'])}-{max(set_result['rarity_list'])}" if len(set_result['rarity_list']) > 1 else str(set_result['rarity_list'][0])
            msg = f"✅ 题库设置成功\n"
            msg += f"星级范围：{rarity_display}星\n"
            msg += f"可选干员：{set_result['operator_count']}个\n"
            msg += f"作用范围：{set_result['scope']}"

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
        await matcher.send(f"❌ 题库设置出错，请检查日志: {str(e)}")

async def handle_mode_settings(uninfo: Uninfo, matcher: Matcher, mode: str | None):
    """统一的模式设置处理（参数来自 Alconna Option「模式」的 mode）。"""
    try:
        user_id = str(uninfo.user.id) if uninfo.user else None
        group_id = str(uninfo.group.id) if uninfo.group else None
        arg = (mode or "").strip()

        if arg == "查看":
            info = mode_manager.get_mode_info(user_id, group_id)
            msg = f"🎭 当前模式设置\n"
            msg += f"模式：{info['mode']}\n"
            msg += f"描述：{info['description']}\n"
            msg += f"设置来源：{info['source']}"
            await matcher.send(msg)
            return

        elif arg == "重置":
            reset_result = mode_manager.reset_mode(user_id, group_id)
            if reset_result["success"]:
                msg = f"✅ 模式已重置\n"
                msg += f"当前模式：{reset_result['mode']}\n"
                msg += f"作用范围：{reset_result['scope']}"
                await matcher.send(msg)
            else:
                await matcher.send(f"❌ 重置失败：{reset_result['message']}")
            return

        if not arg:
            info = mode_manager.get_mode_info(user_id, group_id)
            msg = f"🎭 当前模式设置\n"
            msg += f"模式：{info['mode']}\n"
            msg += f"描述：{info['description']}\n"
            msg += f"设置来源：{info['source']}\n\n"
            msg += f"💡 模式说明：\n"
            msg += f"👤 大头模式：适合正常游戏体验\n\n"
            msg += f"🔧 使用方法：\n"
            msg += f"[arkstart 模式 大头] - 设置为大头模式\n"
            msg += f"[arkstart 模式 查看] - 查看当前设置\n"
            msg += f"[arkstart 模式 重置] - 重置为默认模式\n\n"
            msg += f"💡 提示：模式设置会影响游戏体验"
            await matcher.send(msg)
            return

        set_result = mode_manager.set_mode(arg, user_id, group_id)
        if set_result["success"]:
            msg = f"✅ 模式设置成功\n"
            msg += f"模式：{set_result['mode']}\n"
            msg += f"作用范围：{set_result['scope']}\n"
            msg += f"描述：{mode_manager._get_mode_description(set_result['mode'])}\n\n"

            msg += f"👤 已切换到大头模式\n"
            msg += f"💡 下次开始游戏时将使用大头模式\n"
            msg += f"🎯 大头模式适合正常的游戏体验"
            await matcher.send(msg)
        else:
            await matcher.send(f"❌ 设置失败：{set_result['message']}")

    except FinishedException:
        return
    except Exception as e:
        await matcher.send(f"❌ 模式设置出错，请检查日志: {str(e)}")

async def handle_end(uninfo: Uninfo):
    game_data = get_game_instance().get_game(uninfo)
    operator = game_data["operator"]
    current_mode = game_data.get("current_mode", "大头")
    get_game_instance().end_game(uninfo)
    img = await render_correct_answer(operator, current_mode)
    await UniMessage(Image(raw=img)).send()

@guess_matcher.handle()
async def handle_guess(uninfo: Uninfo, event: Event):
    guess_name = event.get_plaintext().strip()
    if guess_name in ("", "结束", "arkstart"):
        if guess_name == "结束":
            # 检查是否在连战模式中
            game_data = get_game_instance().get_game(uninfo)
            if game_data and game_data.get("continuous_mode", False):
                continuous_count = get_game_instance().get_continuous_count(uninfo)
                if continuous_count > 0:
                    # 连战模式退出
                    operator = game_data["operator"]
                    current_mode = game_data.get("current_mode", "大头")
                    get_game_instance().end_game(uninfo)
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
    game_data = get_game_instance().get_game(uninfo)
    if not game_data:
        return
    # 先将别称解析成标准名，用于“重复猜测”与后续 guess
    resolved_guess_name = get_game_instance().resolve_operator_name(guess_name) or guess_name
    # 检查重复猜测
    if any(g["name"] == resolved_guess_name for g in game_data["guesses"]):
        await UniMessage.text(f"🤔 已经猜过【{resolved_guess_name}】了，请尝试其他干员！").send()
        return
        
    correct, guessed, comparison = get_game_instance().guess(uninfo, resolved_guess_name)
    
    if correct:
        # 检查连战模式
        continuous_mode = game_data.get("continuous_mode", False)
        
        if continuous_mode:
            # 连战模式：自动开始新游戏
            # 更新连战计数
            continuous_count = get_game_instance().update_continuous_count(uninfo)
            
            # 结束当前游戏
            get_game_instance().end_game(uninfo)
            
            # 开始新游戏
            new_game = get_game_instance().start_new_game(uninfo)
            
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
            get_game_instance().end_game(uninfo)
            current_mode = game_data.get("current_mode", "大头")
            img = await render_correct_answer(guessed, current_mode)
            await UniMessage([
                "🎉 恭喜你猜对了！\n🎯 正确答案：",
                Image(raw=img)
            ]).send()
        return
    
    if not guessed:
        similar = get_game_instance().find_similar_operators(guess_name)
        if not similar:
            return
        err_msg = f"❓ 未找到干员【{guess_name}】！\n💡 尝试以下相似结果：" + "、".join(similar)
        await UniMessage.text(err_msg).send()
        return

    attempts_left = get_game_instance().max_attempts - len(game_data["guesses"])
    # 检查尝试次数
    if attempts_left <= 0:
        operator = game_data["operator"]
        current_mode = game_data.get("current_mode", "大头")
        get_game_instance().end_game(uninfo)
        img = await render_correct_answer(operator, current_mode)
        await UniMessage([
            "😅 尝试次数已用尽！\n🎯 正确答案：",
            Image(raw=img)
        ]).send()
        return
    
    current_mode = game_data.get("current_mode", "大头")
    img = await render_guess_result(guessed, comparison, attempts_left, current_mode)
    
    # 添加连战模式进度显示
    if get_game_instance().is_continuous_mode(uninfo):
        continuous_count = get_game_instance().get_continuous_count(uninfo)
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

async def handle_update_resources(uninfo: Uninfo, matcher: Matcher):
    """统一的资源更新处理（鉴权在 _dispatch_update 中用 SUPERUSER 完成）。"""
    await matcher.send("🔄 开始更新资源...")
    try:
        await update_database(matcher)
    except Exception as e:
        await matcher.send(f"❌ 更新过程中出错：{str(e)}\n💡 请检查日志或稍后重试")

async def update_database(matcher: Matcher):
    """更新干员数据库"""
    try:
        # 进程内直接调用更新模块，确保 localstore 配置生效
        from ..resource_tools.data_update import update_data
        
        # 运行数据库更新
        success = await update_data()
        
        if success:
            # 数据库更新成功后，重新加载数据
            reload_success = get_game_instance().reload_data()
            
            if reload_success:
                await matcher.send("✅ 更新完成")
            else:
                await matcher.send("⚠️ 干员数据库更新完成，但数据重新加载失败\n💡 请尝试重新启动插件或联系管理员")
        else:
            await matcher.send("❌ 干员数据库更新失败，请检查日志")
        
    except Exception as e:
        await matcher.send(f"❌ 数据库更新失败：{str(e)}")
        raise