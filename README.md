<div align="center">
  <a href="https://v2.nonebot.dev/store"><img src="https://github.com/A-kirami/nonebot-plugin-template/blob/resources/nbp_logo.png" width="180" height="180" alt="NoneBotPluginLogo"></a>
  <br>
  <p><img src="https://github.com/A-kirami/nonebot-plugin-template/blob/resources/NoneBotPlugin.svg" width="240" alt="NoneBotPluginText"></p>
</div>

<div align="center">

# nonebot-plugin-arkguesser

_✨ 明日方舟猜干员游戏 - 支持多种游戏模式和题库设置 ✨_


<a href="./LICENSE">
    <img src="https://img.shields.io/github/license/lizhiqi233-rgb/nonebot-plugin-arkguesser.svg" alt="license">
</a>
<a href="https://pypi.python.org/pypi/nonebot-plugin-arkguesser">
    <img src="https://img.shields.io/pypi/v/nonebot-plugin-arkguesser.svg" alt="pypi">
</a>
<img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="python">
<img src="https://img.shields.io/badge/nonebot-2.4.2+-green.svg" alt="nonebot">

</div>

## 📖 介绍

这是一个基于 NoneBot2 的明日方舟猜干员游戏插件，支持多种游戏模式和题库设置，为群聊和私聊提供有趣的游戏体验。

### 🎮 游戏特色
- **多种星级范围题库**：支持1-6星干员的不同组合
- **大头模式**：适合正常游戏体验
- **兔头模式**：增加游戏趣味性
- **连战模式**：猜对后自动开始下一轮，享受连续游戏乐趣
- **智能题库管理**：支持群组和个人设置，优先级明确
- **丰富的干员信息**：包含职业、种族、势力、数值等详细属性
- **智能匹配系统**：支持干员名称和拼音模糊匹配

## 🏗️ 技术架构

### 核心组件
- **游戏引擎** (`game.py`): 核心游戏逻辑和状态管理
- **题库管理器** (`pool_manager.py`): 星级范围设置和干员抽取
- **模式管理器** (`mode_manager.py`): 大头/兔头模式切换
- **连战管理器** (`continuous_manager.py`): 连战模式控制
- **渲染引擎** (`render.py`): HTML模板渲染和图片处理
- **配置管理** (`config.py`): 插件配置和环境变量

### 技术特性
- 基于 NoneBot2 框架构建
- 使用 Alconna 命令解析器
- 支持多种适配器
- 模块化设计架构
- 本地数据存储管理
- HTML模板渲染系统

## 💿 安装

<details open>
<summary>使用 nb-cli 安装</summary>
在 nonebot2 项目的根目录下打开命令行, 输入以下指令即可安装

    nb plugin install nonebot-plugin-arkguesser

</details>

<details>
<summary>使用包管理器安装</summary>
在 nonebot2 项目的插件目录下, 打开命令行, 根据你使用的包管理器, 输入相应的安装命令

<details>
<summary>pip</summary>

    pip install nonebot-plugin-arkguesser
</details>
<details>
<summary>pdm</summary>

    pdm add nonebot-plugin-arkguesser
</details>
<details>
<summary>poetry</summary>

    poetry add nonebot-plugin-arkguesser
</details>
<details>
<summary>conda</summary>

    conda install nonebot-plugin-arkguesser
</details>

打开 nonebot2 项目根目录下的 `pyproject.toml` 文件, 在 `[tool.nonebot]` 部分追加写入

    plugins = ["nonebot_plugin_arkguesser"]

</details>

## 📦 依赖要求

### 必需依赖
- **nonebot2** >= 2.4.2 - 机器人框架
- **nonebot-plugin-alconna** >= 0.59.3 - 命令解析器
- **nonebot-plugin-uninfo** >= 0.9.0 - 用户信息管理
- **nonebot-plugin-htmlrender** >= 0.6.6 - HTML渲染
- **nonebot-plugin-localstore** >= 0.7.4 - 本地数据存储
- **pypinyin** >= 0.55.0 - 中文拼音支持
- **pydantic** >= 2.11.7 - 数据验证
- **httpx** >= 0.24.0 - HTTP客户端
- **Pillow** >= 9.0.0 - 图像处理
- **loguru** >= 0.7.0 - 日志记录
- **arclet-alconna** >= 0.59.3 - Alconna命令解析器核心

## ⚙️ 配置

在 nonebot2 项目的`.env`文件中添加下表中的配置项

| 配置项 | 必填 | 默认值 | 说明 |
|:-----:|:----:|:----:|:----:|
| `arkguesser_max_attempts` | 否 | 10 | 最大尝试次数 |
| `arkguesser_default_rarity_range` | 否 | "6" | 默认星级范围 |
| `arkguesser_default_mode` | 否 | "大头" | 默认游戏模式 |

## 🎉 使用

### 指令表
| 指令 | 权限 | 需要@ | 范围 | 说明 |
|:-----:|:----:|:----:|:----:|:----:|
| `arkstart` | 群员 | 否 | 群聊/私聊 | 开始游戏 |
| `结束` | 群员 | 否 | 群聊/私聊 | 结束游戏 |
| 直接输入干员名 | 群员 | 否 | 群聊/私聊 | 开始猜测 |

### 🎮 游戏指令详解
默认情况下本游戏指令无需前置斜杠/
#### 基础游戏
- `arkstart` - 开始游戏
- `结束` - 结束游戏
- 直接输入干员名即可开始猜测

#### 📚 题库设置
- `/arkstart 题库` - 查看题库设置和使用方法
- `/arkstart 题库 6` - 设置题库为6星干员
- `/arkstart 题库 4-6` - 设置题库为4-6星干员
- `/arkstart 题库 查看` - 查看当前题库设置
- `/arkstart 题库 重置` - 重置为默认设置

#### 🎭 模式设置
- `/arkstart 模式` - 查看模式设置和使用方法
- `/arkstart 模式 大头` - 设置为大头模式
- `/arkstart 模式 兔头` - 设置为兔头模式
- `/arkstart 模式 查看` - 查看当前模式设置
- `/arkstart 模式 重置` - 重置为默认模式

#### 🔄 连战模式设置
- `/arkstart 连战` - 查看连战模式设置和使用方法
- `/arkstart 连战 开启` - 开启连战模式
- `/arkstart 连战 关闭` - 关闭连战模式
- `/arkstart 连战 查看` - 查看当前连战模式设置
- `/arkstart 连战 重置` - 重置为默认连战模式设置

#### 🔄 资源更新设置
- `/arkstart 更新` - 查看资源更新系统说明和使用方法
- `/arkstart 更新 数据库` - 更新干员数据库（从PRTS Wiki获取最新数据）
- `/arkstart 更新 立绘` - 更新所有干员立绘资源
- `/arkstart 更新 6星` - 只更新6星干员立绘
- `/arkstart 更新 4-6星` - 更新4-6星干员立绘
- `/arkstart 更新 全量` - 全量更新（数据库+所有立绘）

### ⚙️ 群组配置说明
- **群聊设置**：对所有群成员生效
- **个人设置**：只在私聊中生效  
- **优先级**：群聊设置 > 个人设置 > 默认设置

### 💡 使用技巧
1. **题库选择**：根据群组水平选择合适的星级范围
2. **模式切换**：大头模式适合正常游戏，兔头模式增加趣味性
3. **连战模式**：适合活跃的群聊，保持游戏连续性
4. **个人设置**：私聊中可以设置个人偏好，不影响群聊

## 🎯 游戏特性详解

### 干员信息展示
- **基础属性**：名称、星级、职业、子职业
- **背景信息**：种族、性别、出身地、势力
- **战斗属性**：攻击、防御、生命值、法抗、攻击间隔、费用
- **部署信息**：部署位置、标签

### 智能匹配系统
- **精确匹配**：完全匹配干员名称
- **模糊匹配**：支持部分名称匹配
- **拼音匹配**：支持拼音输入和匹配
- **容错处理**：自动处理常见输入错误

### 题库管理
- **星级范围**：支持1-6星的任意组合
- **动态调整**：实时修改题库范围
- **数据统计**：显示当前题库干员数量
- **持久化存储**：设置自动保存和恢复

## 🚀 特性

- ✅ 支持多种星级范围题库（1-6星）
- ✅ 大头模式和兔头模式切换
- ✅ 连战模式自动下一轮
- ✅ 群组和个人设置分离
- ✅ 智能优先级管理
- ✅ 完整的指令系统
- ✅ 美观的游戏界面
- ✅ 丰富的干员信息展示
- ✅ 智能名称匹配系统
- ✅ 本地数据持久化
- ✅ 模块化架构设计

## 📁 项目结构

```
nonebot-plugin-arkguesser/                    # 项目根目录
├── nonebot_plugin_arkguesser/               # 插件包目录
│   ├── __init__.py                          # 插件入口和主要逻辑
│   ├── game_tools/                          # 游戏工具模块
│   │   ├── __init__.py                      # 游戏工具模块入口
│   │   ├── game.py                          # 游戏核心引擎
│   │   ├── pool_manager.py                  # 题库管理器（动态统计星级数量）
│   │   ├── mode_manager.py                  # 模式管理器
│   │   ├── continuous_manager.py            # 连战模式管理器
│   │   ├── render.py                        # 渲染引擎
│   │   └── config.py                        # 配置管理
│   ├── resource_tools/                      # 资源工具模块
│   │   ├── __init__.py                      # 资源工具模块入口
│   │   ├── illustration_config.py           # 立绘配置管理
│   │   ├── illustration_downloader_v2.py    # 立绘下载器V2
│   │   ├── update_simple.py                 # 简化资源更新工具
│   │   └── run_illustration_download.py     # 立绘下载执行脚本
│   └── resources/                           # 资源文件目录
│       ├── images/                          # 图片资源目录
│       │   └── xlpj/                       # 特殊图片资源
│       └── templates/                       # HTML模板目录
│           ├── guess.html                   # 猜测结果模板
│           ├── correct.html                 # 正确答案模板
│           ├── guess_rabbit.html            # 兔头模式猜测模板
│           └── correct_rabbit.html          # 兔头模式答案模板
├── pyproject.toml                           # 项目配置文件
├── requirements.txt                          # 依赖列表
├── README.md                                # 项目说明文档
├── .gitignore                               # Git忽略文件配置
└── MANIFEST.in                              # 分发包文件清单
```

## 📝 更新日志

### v0.3.0 🆕
- 🔄 题库星级统计改为动态读取 `characters.csv`，移除固定估算
- 🔁 数据更新后自动刷新星级统计缓存
- 🧭 兼容 `nonebot_plugin_localstore` 数据目录路径读取
- 🐞 修复了少量bug，新增了大量ai代码

### v0.2.0
- 🏗️ **架构重构**：将游戏逻辑和资源工具分离到独立模块
- 📁 **模块化设计**：新增 `game_tools/` 和 `resource_tools/` 模块
- 🔧 **代码优化**：重构代码结构，提升可维护性和扩展性
- 📦 **资源管理**：新增立绘资源自动更新和管理工具
- 🚀 **性能提升**：优化模块导入和资源加载机制
- 🎯 **功能增强**：改进游戏逻辑和用户体验
- 📚 **文档完善**：更新项目结构和说明文档

### v0.1.4
- 🎉 移除部分答辩代码增加大量答辩代码
- 🎮 完整的明日方舟猜干员游戏功能
- 📚 支持多种星级范围题库设置
- 🎭 大头和兔头模式切换
- 🔄 连战模式支持
- ⚙️ 群组和个人配置管理
- 🏗️ 模块化架构设计
- 🎨 HTML模板渲染系统
- 💾 本地数据持久化存储

### v0.1.1
- 🔧 修复缺失的pypinyin依赖问题
- 📦 完善项目依赖配置
- 🚀 提升插件加载稳定性

### v0.1.0
- 🎉 初始版本发布
- 🎮 基础猜干员游戏功能
- 🎭 大头和兔头模式
- 🔄 连战模式支持
- ⚙️ 题库和模式设置
- 👥 群组和个人配置

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

### 开发环境设置
1. 克隆项目仓库
2. 安装依赖：`pip install -r requirements.txt`
3. 运行测试：`python -m pytest`

### 贡献指南
- 遵循 PEP 8 代码规范
- 添加适当的注释和文档
- 确保所有测试通过
- 提交前运行代码格式化

## 📄 许可证

本项目采用 [MIT](./LICENSE) 许可证。

## 🙏 致谢
- 因作者编程水平较差，许多代码使用了Cursor，所以你可能会看到大量ai代码
- [FrostN0v0](https://github.com/FrostN0v0) - 感谢提供技术指导和建议
- [NoneBot2](https://github.com/nonebot/nonebot2) - 优秀的机器人框架
- [nonebot-plugin-alconna](https://github.com/ArcletProject/nonebot-plugin-alconna) - 强大的指令解析器
- [nonebot-plugin-htmlrender](https://github.com/kexue-z/nonebot-plugin-htmlrender) - 美观的渲染器
- [nonebot-plugin-mhguesser](https://github.com/Proito666/nonebot-plugin-mhguesser) - 原项目灵感来源
- [ArknightsGameData](https://github.com/Kengxxiao/ArknightsGameData) - 明日方舟游戏数据
- [Arknights_guessr](https://github.com/Dioxane123/Arknights_guessr) - 部分数据来源
- [arknights-toolkit](https://github.com/RF-Tar-Railt/arknights-toolkit) - 明日方舟相关功能整合库
- [PRTS Wiki](https://prts.wiki/w/%E9%A6%96%E9%A1%B5) - 明日方舟游戏资料百科

## 📞 联系方式

- 项目主页：[GitHub](https://github.com/lizhiqi233-rgb/nonebot-plugin-arkguesser)
- 问题反馈：[Issues](https://github.com/lizhiqi233-rgb/nonebot-plugin-arkguesser/issues)
- 讨论交流：[Discussions](https://github.com/lizhiqi233-rgb/nonebot-plugin-arkguesser/discussions)
