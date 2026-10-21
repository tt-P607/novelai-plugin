# 官方图像请求用法

接口格式来源：https://image.novelai.net/docs/doc.json 。功能说明：https://docs.novelai.net/en/image/ 。完整 schema 已随插件保存；`novelai_schema` 提供实时可查的本地快照。

以下是请求构造示例，不是已经验证出图的效果预设。不要把参考项目的 UI 上限或默认值当作 NovelAI 硬限制。原生字段完整保留，包括 OpenAPI 尚未列出的字段；不自动修改模型、种子、质量词或限制张数。

## 所有付费操作的前置确认

以下 JSON 示例是生成/工具的真实请求；省略批准字段仍会先核对费用，确认免费可执行，收费/未知会被拦截。只查价格时用 `novelai_quote(operation, request)`。收费/未知需向用户申请本次同意，获准后在工具参数顶层添加 `cost_approved: true`，不得塞进官方 request。继续调整也重新核价；没有确认时禁止自行添加true。免费规则与完整公式见[billing.md](billing.md)。

标准画幅按构图选1024×1024、832×1216、1216×832，不固定默认方图。大于1,048,576总像素须额外获准才可添加 `high_resolution_approved: true`。upscale的最终尺寸不能预先确定时也要说明此风险并获准。所有批准只针对已说明的操作、尺寸和张数。

## 文生图

传给 `novelai_generate` 的完整参数：

```json
{
  "request": {
    "input": "1girl, reading a book by a rainy window, soft light",
    "model": "nai-diffusion-4-5-full",
    "action": "generate",
    "parameters": {
      "params_version": 4,
      "width": 832,
      "height": 1216,
      "n_samples": 1,
      "steps": 28,
      "scale": 6,
      "sampler": "k_euler_ancestral",
      "seed": 123456789,
      "negative_prompt": "lowres, blurry",
      "v4_prompt": {
        "caption": {"base_caption": "1girl, reading a book by a rainy window, soft light", "char_captions": []},
        "use_coords": false,
        "use_order": true
      },
      "v4_negative_prompt": {
        "caption": {"base_caption": "lowres, blurry", "char_captions": []},
        "legacy_uc": false
      }
    }
  }
}
```

V5 使用其 API 模型名，例如 `nai-diffusion-5-full`，保留 V4 形式的结构化 caption；不因为 schema 没有 model enum 就猜测不存在的模型。种子为用户未指定时可自行选取的 uint32 整数。V3 可使用 input / parameters.prompt / negative_prompt，省略 V4 caption。

`stream: true` 选择 `/ai/generate-image-stream`。常规对话优先使用 `parameters.stream: "sse"`，也支持原生 `parameters.stream: "msgpack"`，按官方前端的 4 字节大端长度帧解码二进制图片。仅最终帧视为结果；中间预览不能证明生成成功。两种协议已用本地替身验证，仍需真实账号验证。

## 图生图

沿用上面的 request，改 `action: "img2img"`，增加 `parameters.strength`、`noise`、`extra_noise_seed`、`color_correct` 等所需字段。绑定：

```json
{"files":[{"pointer":"/parameters/image","path":"D:/素材/原图.png"}]}
```

如继续上一次结果：`novelai_revise(run_id="...", prompt="...", patch={"parameters":{"strength":0.35,"noise":0}}, use_output_as_input=true)`。

## 局部重绘

先查看原图。用用户提供的 mask，或通过 `novelai_create_mask` 按原图像素坐标画 rectangle/ellipse/polygon；白色重绘，黑色保留。预览的显示尺寸可能与原图不同，坐标应换算到返回的原图 width/height。先检查遮罩覆盖的位置，再发起生成。

- `action: "infill"`。
- 使用对应 inpainting 模型：V4.5 Full 为 `nai-diffusion-4-5-full-inpainting`；V5 Full/Curated 按参考项目映射至 `nai-diffusion-5-full-inpainting`；V4 Curated Preview 为 `nai-diffusion-4-curated-inpainting`。
- 绑定 `/parameters/image` 和 `/parameters/mask`。
- mask 应与源图等尺寸；用户已有官方低分辨率 mask 可保留为各边 1/8 尺寸，不擅自放大。
- V4+ 的局部 strength 可使用 `parameters.img2img: {"strength":0.55,"color_correct":true}`。该字段与顶层 img2img action 不同。
- 不把原图宽高悄悄改成另一个生成尺寸。

## 氛围转移

V3 可按其能力发送参考图；V4/V4.5 通常先编码：

```json
{
  "request": {"model":"nai-diffusion-4-5-full","information_extracted":1},
  "files":[{"pointer":"/image","path":"D:/素材/氛围参考.png"}]
}
```

这调用 `novelai_encode_vibe`，返回 `vibe.bin`。可提供 mask、crop_to_mask、focus_seed、info_extract_seed 等全部原生参数。编码也可能消耗额度；已有编码时直接复用。

然后在生成参数中设置：

```json
{
  "reference_information_extracted_multiple":[1],
  "reference_strength_multiple":[0.6],
  "normalize_reference_strength_multiple":true
}
```

绑定 `{"pointer":"/parameters/reference_image_multiple/0","path":"D:/.../vibe.bin","mode":"vibe"}`。多个参考的各个数组顺序一致。

支持 `.naiv4vibe` 和 JSON bundle 的编码读取：`mode:"vibe"`，可用 `information_extracted` 选择变体、`vibe_index` 选择 bundle 条目。模型 key 和 information_extracted 必须匹配；没有编码的 image-only 文件需先读取原图并 encode，不能把 PNG 当作 opaque V4 编码使用。该容器依据用户参考项目实现，官方 OpenAPI 只描述编码接口。

## 精确参考

选择支持此功能的 V4.5。加入：

```json
{
  "director_reference_descriptions":[{"caption":{"base_caption":"character&style","char_captions":[]},"legacy_uc":false}],
  "director_reference_information_extracted":[1],
  "director_reference_strength_values":[1],
  "director_reference_secondary_strength_values":[0]
}
```

绑定 `{"pointer":"/parameters/director_reference_images/0","path":"D:/素材/角色参考.png","prepare":"precise_reference"}`。预处理保持比例，拟合到官方三个画布之一并补黑边；选择画布的规则是本插件按最少补边选择，不是官方公开的阈值。用户已有处理好的原生输入可用 prepare:none 或直接传 base64。

类型 caption 使用 `character`、`style` 或 `character&style`；数组按参考顺序对应。`secondary_strength_values` 保留 API 原值，参考项目的 UI Fidelity 映射为 `1 - fidelity`。不要再对用户已经提供的原生值做反转。多个角色参考可能被混合，不保证自动一一绑定到不同角色。

## 多角色、透明背景、其他参数

`v4_prompt.caption.char_captions` 逐项放 `{ "char_caption":"...", "centers":[{"x":0.3,"y":0.5}] }`，负面 caption 对应设置。`use_coords`、`use_order` 等按用户意图选择，不设本地角色数白名单。

透明背景可在支持模型中使用相关提示词和 `tag_hint_transparent_background`、`straight_alpha`、`image_format`。其他采样、SMEA、噪声、CFG、ControlNet、legacy、Variety+、upscaled_enhance 等参数直接传入官方 request；模型支持范围以官方为准。

V5 Effort 的官网说明和公开 OpenAPI 字段可能不同步；不要凭一个网页开关名猜测 API 字段。用户有已工作的原生 payload 时可原样透传。

## Director Tools、放大、标签

- `novelai_augment` 对应 `/ai/augment-image`，request 可带 `req_type`、`width`、`height`、`prompt`、`defry`，绑定 `/image`。req_type 例如 `bg-removal`、`lineart`、`sketch`、`colorize`、`emotion`、`declutter`。
- `novelai_upscale` 对应 `/ai/upscale`，使用官方字段（当前包含 model、image、declared_blur_sigma）；不把旧接口的 scale 参数当成新版必填。未知扩展字段仍可传递。
- `novelai_suggest_tags` 对应 GET 标签接口，request 为 model、prompt、可选 lang。

## 来源与验证边界

官方 schema 快照：2026-10-20；官方 Precise/Vibe/模型文档同日检查。Huishu 相关实现的来源与许可见仓库 NOTICE。

MCP 工具能调用、替身接口通过、图片能解码，仅证明本地链路。真实账号的字段兼容、扣费和生成质量，必须在用户配置 Token 后实际调用才能确认。

MessagePack framing 依据官方公开前端 bundle：https://novelai.net/_next/static/chunks/0wi7fejx317_y.js?dpl=b50892a-38995131626-1 。
