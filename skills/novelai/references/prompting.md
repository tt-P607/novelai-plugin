# 通用提示词与参数

不内置人物、画风、服装或专属提示词模板。根据当前用户描述组织主体、动作、环境、构图和光线；仅写词或讨论时不生成。

## 参数映射

| 需求 | 原生字段 | 含义 |
|---|---|---|
| Guidance / CFG / scale | parameters.scale | 提示词引导强度，不是图片缩放 |
| Guidance rescale / PGR | parameters.cfg_rescale | 独立于 scale 的引导重缩放 |
| 图生图重绘强度 | parameters.strength | 与输入图的变化程度 |
| 局部重绘图生图强度 | parameters.img2img.strength | 按模型及对应工作流使用 |
| 放大 | novelai_upscale | 使用独立放大接口 |

未指定时 steps=28、scale=6、n_samples=1；保留用户原有参数。连续调 CFG 时只 patch scale，保持其他变量方便比较：

```json
{"run_id":"上次记录ID","patch":{"parameters":{"scale":5.5}}}
```

V4/V4.5/V5 的正负提示词按相应模型使用结构化 caption。多角色身份与动作放入 char_captions，公共场景放 base_caption；centers 使用 0–1 数值坐标。提示词权重、文字渲染和支持长度以所选模型的官方文档为准，不从其他服务推断。

读取原图进行复现时，保留提示词、种子、尺寸、采样器、步数、CFG 和有效输入参数。图片元数据还包含输出记录和内部字段；不要把签名、model_hash、request_type 等输出字段整份回传为生成参数。原始元数据完整保留用于分析，重建请求按官方输入 schema 和所选服务支持情况处理。

仅在用户要求实际生成时调用；核实免费可执行，收费或价格未知先申请。看图后展示原图绝对路径和关键参数。

来源：[Guidance](https://docs.novelai.net/en/image/stepsguidance/)、[权重](https://docs.novelai.net/en/image/strengthening-weakening/)、[模型](https://docs.novelai.net/en/image/models/)、[文字渲染](https://docs.novelai.net/en/image/textrendering/)。
