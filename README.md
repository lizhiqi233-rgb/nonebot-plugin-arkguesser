# nonebot-plugin-arkguesser

基于 [NoneBot2](https://nonebot.dev/) 的《明日方舟》猜干员小游戏插件，支持题库星级范围、大头模式、连战模式与资源更新等。

**当前版本：v1.0.1**

- 仓库：<https://github.com/lizhiqi233-rgb/nonebot-plugin-arkguesser>
- PyPI 包名：`nonebot-plugin-arkguesser`

## 安装

```bash
pip install nonebot-plugin-arkguesser
```

或在项目 `pyproject.toml` 的 `[tool.nonebot]` 中启用：

```toml
plugins = ["nonebot_plugin_arkguesser"]
```

## 依赖

见 `pyproject.toml` / `requirements.txt`（NoneBot2、`nonebot-plugin-alconna`、`nonebot-plugin-uninfo`、`nonebot-plugin-htmlrender`、`nonebot-plugin-localstore`、`jinja2` 等）。

**可选依赖（大头对齐）**：`resource_tools` 在更新流程中会生成 `char_e2_head_align.csv`，需 OpenCV 与 NumPy。若未安装，`update_simple.py` 会在对应步骤报错并提示安装。安装方式：

```bash
pip install "nonebot-plugin-arkguesser[head-align]"
```

## 配置（`.env`）

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `arkguesser_max_attempts` | `10` | 最大尝试次数 |
| `arkguesser_default_rarity_range` | `"6"` | 默认星级范围 |
| `arkguesser_default_mode` | `"大头"` | 默认游戏模式 |
| `arkguesser_recent_exclude_count` | `40` | 最近已出干员不重复抽取数量 |
| `arkguesser_render_scale` | `1.0` | 结果图渲染缩放（相对 540×540） |

## 使用摘要

- `arkstart`：开始游戏（别名：`明日方舟开始`）
- 游戏中直接输入干员名猜测；`结束` 结束本局
- `arkstart 题库 …` / `arkstart 模式 …` / `arkstart 连战 …`：题库、模式、连战相关设置
- 管理员：`arkstart 更新`（更新数据库）、`arkstart 别称 …`（维护别称表）

完整说明以仓库文档为准。

## 从源码构建

```bash
pip install build
python -m build
```

产物在 `dist/`（wheel 与 sdist）。

## 许可证

MIT，见 `LICENSE`。

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
