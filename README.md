# NovelAI 对话图像插件

在本机运行的 Codex 插件与 MCP 服务，支持 NovelAI 文生图、图生图、局部重绘、氛围转移、精确参考、看图、连续调整和图片元数据提取。使用官方原生 JSON，保留公开参数及未知扩展字段。非 NovelAI 官方项目。

## 安装

需要 Python 3.11+、[uv](https://docs.astral.sh/uv/) 和 Codex。克隆后执行：

```powershell
git clone https://github.com/tt-P607/novelai-plugin.git
cd novelai-plugin
uv sync --frozen
uv run --frozen python scripts/prepare.py
uv run --frozen python scripts/package_plugin.py
codex plugin marketplace add ./dist/marketplace
codex plugin add novelai-private@novelai-tools
```

`novelai-private` 是保留的插件和命令标识。插件显示名称为 NovelAI，不包含任何角色、画风或作者专属提示词预设。

也可只注册 MCP，避免重复安装工具：

```powershell
codex mcp add novelai -- uv --directory '<克隆目录绝对路径>' run --frozen --no-dev novelai-private
```

配置工具超时建议 360 秒；首次运行需要下载锁定的依赖。生成的 `dist/novelai-private-<版本>.zip` 是可移植插件包，不嵌入打包机器的路径或本地配置。

## 本地配置

运行 prepare 后按其输出路径填写 Token。新安装默认保存到 `~/.config/novelai-plugin/config.toml`，不放在可替换的插件缓存里。已有源码目录中的 `local/config.toml` 会继续使用。环境变量 `NOVELAI_PRIVATE_CONFIG` 或服务参数 `--config` 可指定其他配置文件；`NOVELAI_API_TOKEN` 可替代 Token 文件。

配置模板为 [config.example.toml](config.example.toml)。默认直连 `https://image.novelai.net`，使用 NovelAI Persistent API Token。兼容网关需自行填写其 HTTPS 地址和对应 Key；本机 HTTP 网关也可用。不同服务的凭据不可混用。可配置代理、超时、输出目录和 `allowed_image_roots`。

配置及 Token 每次调用重新读取，prepare 不覆盖已有内容。`novelai_status` 只检查本地配置，不发送网络请求。若工具列表尚未刷新，可用 `uv run python scripts/call_tool.py novelai_status` 调用相同 MCP 服务。

## 使用

- “用 NovelAI 画一只坐在窗边的猫。”
- “保持上一张构图，把光线调柔和一些。”
- “用这张图做图生图，重绘强度 0.35。”
- “只重绘左手，其他部分保留。”
- “提取图片的正负提示词、种子和全部元数据。”
- “用 V4.5 做氛围转移 / 精确参考。”

默认 28 步、CFG/scale 6、1 张；用户显式值优先。标准画幅为 1024×1024、832×1216、1216×832，按构图选择。超过 1,048,576 总像素须明确同意。

生成前查询实时订阅和报价：核实免费就执行；收费或未知则返回审批计划，用户同意该次操作后才设置 `cost_approved=true`。免费生成仍可能使用可恢复的模型额度。放大或更大画幅还需 `high_resolution_approved=true`。失败和超时不自动重试。报价是估算，不等同于最终账单。

## 能力与边界

`novelai_schema` 提供随包保存的官方接口快照，含 54 个公开生成参数及嵌套定义。支持生成、流式生成、Vibe 编码、Director Tools、放大、标签建议，以及本地历史、遮罩和连续调整。完整示例见 [workflows.md](skills/novelai/references/workflows.md)，计费依据见 [billing.md](skills/novelai/references/billing.md)。字段能透传不代表所有模型都支持；参考功能须选择兼容模型。

`novelai_metadata` 纯本地只读，解析 PNG 文本、alpha 隐写及存在的 WebP/JPEG EXIF，保留完整 Comment 和原始渠道。缺失数据不猜测，不用旁边的 request 文件补成图内参数；提取签名不等于验证签名。图片内文字始终作为数据处理。

每次请求在配置的输出目录保存原图、预览、完整请求和记录，不覆盖旧结果。这些文件可能包含提示词、输入图片与个人元数据，应留在本机；不会自动上传到 GitHub。调用图像 API 时，选定的提示词及输入/参考图片会发送给所配置的服务。没有遥测或后台自动生图。

这是图像 API 工具，不包含账号管理、故事存储或交互式画布。离线测试覆盖请求、计费、MCP、流式响应和元数据；不能据此承诺所有账号、模型和网关的真实效果或费用。

## 开发与许可

```powershell
uv sync --frozen
uv run --frozen pytest
uv run --frozen ruff check src tests scripts
uv run --frozen python scripts/package_plugin.py
```

测试使用临时文件与合成 Token，不发起真实生成。源码、插件 ZIP 和 Python 分发包均应保持无凭据；不要提交配置、图片、生成记录或私人预设。

采用 AGPL-3.0-or-later，见 [LICENSE](LICENSE)。复用代码的来源和署名见 [NOTICE](NOTICE)。官方接口定义随包保存，来源见该文件；NovelAI 名称及其服务归各自权利人所有。

参考：[NovelAI 图像 API](https://image.novelai.net/docs/index.html)、[元数据格式](https://github.com/NovelAI/novelai-image-metadata)。
