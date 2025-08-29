import base64
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
import zipfile
from typing import Optional, Dict
from nonebot_plugin_htmlrender import html_to_pic

# 设置Jinja2环境
env = Environment(
    loader=FileSystemLoader(Path(__file__).parent.parent / "resources" / "templates"),
    autoescape=True,
    enable_async=True
)

# 获取可写数据目录（与更新/下载/游戏读取一致）
def _safe_get_data_dir() -> Path:
    try:
        from nonebot_plugin_localstore import get_plugin_data_dir as _get_dir  # type: ignore
        return _get_dir()
    except Exception:
        return Path.home() / ".local" / "share" / "nonebot2" / "nonebot_plugin_arkguesser"

DATA_DIR = _safe_get_data_dir()
ILLUSTRATIONS_DIR = DATA_DIR / "illustrations"
IMAGES_DIR = DATA_DIR / "images"

def get_local_image_path(filename: str) -> str:
    """
    获取本地图片的完整路径
    
    Args:
        filename: 本地图片的文件路径或原文件名
    
    Returns:
        本地图片的文件路径或原文件名
    """
    if not filename:
        return ""
    
    # 检查是否是ZIP文件中的图片
    if "/" in filename and not filename.startswith(('http://', 'https://')):
        # 这是ZIP文件中的路径，格式如 "6/阿米娅.webp"
        return filename
    
    # 如果是其他格式的文件名，直接返回
    return filename

def get_zip_image_content(filename: str) -> Optional[bytes]:
    """
    从ZIP文件中获取图片内容
    
    Args:
        filename: ZIP文件中的图片路径，格式如 "6/阿米娅.webp"
    
    Returns:
        图片的二进制内容，如果失败则返回None
    """
    if not filename or "/" not in filename:
        return None
    
    try:
        # 构建ZIP文件路径 - 使用数据目录
        zip_path = IMAGES_DIR / "illustrations.zip"
        if not zip_path.exists():
            print(f"ZIP文件不存在: {zip_path}")
            return None
        
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            if filename in zip_file.namelist():
                return zip_file.read(filename)
            else:
                print(f"ZIP文件中未找到图片: {filename}")
                return None
                
    except Exception as e:
        print(f"读取ZIP文件失败 {filename}: {e}")
        return None

# 删除了random导入，不再需要随机背景图功能

def image_to_data_uri(image_path: str) -> str:
    """
    将本地图片转换为 data URI 格式
    
    Args:
        image_path: 图片路径或ZIP文件中的路径
    
    Returns:
        data URI 字符串、原路径或特殊标识符
    """
    if not image_path or image_path.startswith(('http://', 'https://')):
        # 如果是URL，直接返回
        return image_path
    
    # 检查是否是新的目录结构路径
    if "/" in image_path and not image_path.startswith(('http://', 'https://')) and not Path(image_path).exists():
        # 这是新的目录结构路径，格式如 "稀有度6/近卫/半身像/银灰_半身像_精英2.png"
        result = _new_illustration_to_data_uri(image_path)
        if not result:
            return "MISSING_IMAGE"  # 返回特殊标识符
        return result
    
    # 检查是否是ZIP文件中的图片路径
    if "/" in image_path and not Path(image_path).exists():
        # 这是ZIP文件中的路径，格式如 "6/阿米娅.webp"
        result = _zip_image_to_data_uri(image_path)
        if not result:
            return "MISSING_IMAGE"  # 返回特殊标识符
        return result
    
    # 处理本地文件
    try:
        image_file = Path(image_path)
        if not image_file.exists():
            print(f"图片文件不存在: {image_path}")
            return "MISSING_IMAGE"  # 返回特殊标识符
        
        # 检查文件大小
        file_size = image_file.stat().st_size
        MAX_SIZE = 3 * 1024 * 1024  # 3MB限制
        
        if file_size > MAX_SIZE:
            print(f"图片文件太大: {image_path} ({file_size / 1024:.1f} KB)")
            return "MISSING_IMAGE"  # 返回特殊标识符
        
        # 读取图片并转换为 base64
        with open(image_file, 'rb') as f:
            image_data = f.read()
        
        return _convert_image_data_to_uri(image_data, image_file.suffix)
        
    except Exception as e:
        print(f"转换图片失败 {image_path}: {e}")
        return "MISSING_IMAGE"  # 返回特殊标识符

def _new_illustration_to_data_uri(image_path: str) -> str:
    """
    将新目录结构中的立绘转换为 data URI 格式
    
    Args:
        image_path: 新目录结构中的图片路径，格式如 "稀有度6/近卫/半身像/银灰_半身像_精英2.png"
    
    Returns:
        data URI 字符串或空字符串
    """
    try:
        # 构建完整的文件路径（数据目录）
        full_path = ILLUSTRATIONS_DIR / image_path
        
        if not full_path.exists():
            print(f"新目录结构中的图片不存在: {full_path}")
            return "MISSING_IMAGE"
        
        # 检查文件大小
        file_size = full_path.stat().st_size
        MAX_SIZE = 3 * 1024 * 1024  # 3MB限制
        
        if file_size > MAX_SIZE:
            print(f"新目录结构中的图片太大: {image_path} ({file_size / 1024:.1f} KB)")
            return "MISSING_IMAGE"
        
        # 读取图片并转换为 base64
        with open(full_path, 'rb') as f:
            image_data = f.read()
        
        return _convert_image_data_to_uri(image_data, full_path.suffix)
        
    except Exception as e:
        print(f"转换新目录结构图片失败 {image_path}: {e}")
        return ""

def _zip_image_to_data_uri(zip_path: str) -> str:
    """
    将ZIP文件中的图片转换为 data URI 格式
    
    Args:
        zip_path: ZIP文件中的图片路径，格式如 "6/阿米娅.webp"
    
    Returns:
        data URI 字符串或空字符串
    """
    try:
        # 从ZIP文件中获取图片内容
        image_data = get_zip_image_content(zip_path)
        if not image_data:
            return "MISSING_IMAGE"
        
        # 检查数据大小
        if len(image_data) > 3 * 1024 * 1024:  # 3MB限制
            print(f"ZIP中的图片太大: {zip_path} ({len(image_data) / 1024:.1f} KB)")
            return "MISSING_IMAGE"
        
        # 从路径中提取文件扩展名
        suffix = Path(zip_path).suffix.lower()
        return _convert_image_data_to_uri(image_data, suffix)
        
    except Exception as e:
        print(f"转换ZIP图片失败 {zip_path}: {e}")
        return ""

def _convert_image_data_to_uri(image_data: bytes, suffix: str) -> str:
    """
    将图片数据转换为 data URI 格式
    
    Args:
        image_data: 图片的二进制数据
        suffix: 文件扩展名
    
    Returns:
        data URI 字符串
    """
    # 确定 MIME 类型
    if suffix == '.png':
        mime_type = 'image/png'
    elif suffix == '.webp':
        mime_type = 'image/webp'
    elif suffix in ['.jpg', '.jpeg']:
        mime_type = 'image/jpeg'
    elif suffix == '.gif':
        mime_type = 'image/gif'
    else:
        mime_type = 'image/png'  # 默认
    
    # 转换为 base64
    base64_data = base64.b64encode(image_data).decode('utf-8')
    
    # 检查转换后的大小
    if len(base64_data) > 4 * 1024 * 1024:  # 4MB限制
        print(f"Base64数据太大: {len(base64_data)} 字符")
        return ""
    
    # 构建 data URI
    return f"data:{mime_type};base64,{base64_data}"

async def render_guess_result(
    guessed_operator: Optional[Dict],
    comparison: Dict,
    attempts_left: int,
    mode: str = "大头"
) -> bytes:
    # 设置图片尺寸
    width = 450   # 调整为正方形比例 1:1
    height = 450   # 保持高度不变
    
    # 兔头模式和大头模式使用相同的立绘文件
    illustration_filename = guessed_operator.get("illustration", "")
    
    illustration_path = get_local_image_path(illustration_filename)
    illustration_uri = image_to_data_uri(illustration_path)
    
    # 检查图像是否缺失
    if illustration_uri == "MISSING_IMAGE":
        # 生成缺失图像的提示信息
        operator_name = guessed_operator.get("name", "未知干员")
        rarity = guessed_operator.get("rarity", "未知")
        career = guessed_operator.get("profession", "未知")
        
        missing_msg = f"⚠️ 立绘资源缺失\n"
        missing_msg += f"干员：{operator_name}\n"
        missing_msg += f"稀有度：{rarity}星\n"
        missing_msg += f"职业：{career}\n"
        missing_msg += f"💡 请使用 [arkstart 更新 立绘] 来下载立绘资源"
        
        # 将缺失图像信息传递给模板
        illustration_uri = f"data:text/plain;base64,{missing_msg}"
    
    # 移除了背景图功能
    
    # 根据模式选择模板
    template_name = "guess_rabbit.html" if mode == "兔头" else "guess.html"
    template = env.get_template(template_name)
    
    if mode == "兔头":
        # 兔头模式参数
        html = await template.render_async(
            operator_name=guessed_operator["name"],
            attempts_left=attempts_left,
            attack=guessed_operator.get("attack", "未知"),
            attack_comparison=comparison.get("attack", {}),
            defense=guessed_operator.get("defense", "未知"),
            defense_comparison=comparison.get("defense", {}),
            hp=guessed_operator.get("hp", "未知"),
            hp_comparison=comparison.get("hp", {}),
            res=guessed_operator.get("res", "未知"),
            res_comparison=comparison.get("res", {}),
            rarity=guessed_operator["rarity"],
            rarity_comparison=comparison["rarity"],
            gender=guessed_operator["gender"],
            gender_correct=comparison["gender"],
            interval=guessed_operator.get("interval", "未知"),
            interval_comparison=comparison.get("interval", {}),
            cost=guessed_operator.get("cost", "未知"),
            cost_comparison=comparison.get("cost", {}),
            tags=guessed_operator.get("tags", []),
            tags_comparison=comparison.get("tags", {}),
            all_correct=comparison.get("all_correct", False),
            illustration=illustration_uri,
            width=width,
            height=height
        )
    else:
        # 大头模式参数
        html = await template.render_async(
            operator_name=guessed_operator["name"],
            attempts_left=attempts_left,
            profession=guessed_operator.get("profession", "未知"),
            profession_correct=comparison.get("profession", False),
            subProfession=guessed_operator.get("subProfession", "未知"),
            subProfession_correct=comparison.get("subProfession", False),
            rarity=guessed_operator["rarity"],
            rarity_class=comparison["rarity"],
            origin=guessed_operator.get("origin", "未知"),
            origin_correct=comparison.get("origin", False),
            race=guessed_operator.get("race", "未知"),
            race_correct=comparison.get("race", False),
            gender=guessed_operator["gender"],
            gender_correct=comparison["gender"],
            position=guessed_operator.get("position", "未知"),
            position_correct=comparison.get("position", False),
            faction=guessed_operator.get("faction", "未知"),
            parent_faction=guessed_operator.get("parentFaction", ""),
            faction_comparison=comparison.get("faction", {}),
            tags=guessed_operator.get("tags", []),
            tags_comparison=comparison.get("tags", {}),
            illustration=illustration_uri,
            width=width,
            height=height
        )
    
    return await html_to_pic(html, viewport={"width": width, "height": height})

async def render_correct_answer(operator: Dict, mode: str = "大头") -> bytes:
    # 设置图片尺寸
    width = 450   # 调整为正方形比例 1:1
    height = 450   # 保持高度不变
    
    # 兔头模式和大头模式使用相同的立绘文件
    illustration_filename = operator.get("illustration", "")
    
    illustration_path = get_local_image_path(illustration_filename)
    illustration_uri = image_to_data_uri(illustration_path)
    
    # 检查图像是否缺失
    if illustration_uri == "MISSING_IMAGE":
        # 立绘缺失时抛出异常，不继续渲染
        operator_name = operator.get("name", "未知干员")
        rarity = operator.get("rarity", "未知")
        career = operator.get("profession", "未知")
        
        missing_msg = f"立绘资源缺失\n干员：{operator_name}\n稀有度：{rarity}星\n职业：{career}\n请使用 [arkstart 更新 立绘]来下载立绘资源"
        
        raise ValueError(missing_msg)
    
    # 移除了背景图功能
    
    # 根据模式选择模板
    template_name = "correct_rabbit.html" if mode == "兔头" else "correct.html"
    template = env.get_template(template_name)
    
    if mode == "兔头":
        # 兔头模式参数
        html = await template.render_async(
            name=operator.get("name", "未知干员"),
            attack=operator.get("attack", "未知"),
            defense=operator.get("defense", "未知"),
            hp=operator.get("hp", "未知"),
            res=operator.get("res", "未知"),
            rarity=operator.get("rarity", 1),
            gender=operator.get("gender", ""),
            interval=operator.get("interval", "未知"),
            cost=operator.get("cost", "未知"),
            tags=operator.get("tags", []),
            illustration=illustration_uri,
            width=width,
            height=height
        )
    else:
        # 大头模式参数
        html = await template.render_async(
            name=operator.get("name", "未知干员"),
            profession=operator.get("profession", "未知"),
            subProfession=operator.get("subProfession", "未知"),
            rarity=operator.get("rarity", 1),
            origin=operator.get("origin", "未知"),
            race=operator.get("race", "未知"),
            gender=operator.get("gender", ""),
            position=operator.get("position", "未知"),
            faction=operator.get("faction", "未知"),
            parent_faction=operator.get("parentFaction", ""),
            tags=operator.get("tags", []),
            illustration=illustration_uri,
            width=width,
            height=height
        )
    
    return await html_to_pic(html, viewport={"width": width, "height": height})