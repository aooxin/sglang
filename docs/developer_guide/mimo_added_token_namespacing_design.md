# MiMo Added Token 命名空间隔离：整体设计方案

## 1. 方案概览

本方案在 tokenizer 初始化时，为 MiMo 的 added tokens 统一设置内部拼写，并同步调整聊天模板、推理解析器、工具调用解析器和多模态标记。

**核心原则：改变控制 token 的字符串表示，保持它们的 token ID、属性及模型侧语义不变；请求正文保留原文，原始公开拼写按普通文本编码。**

例如，原来的 `<tool_call>` 对应 token ID `T`，改造后：

- 模板输出内部拼写 `<tool_call{suffix}>`，仍编码为 `[T]`。
- 请求正文中的字面量 `<tool_call>` 按普通文本编码，不再命中 `T`。
- 模型生成 `T` 时，detokenizer 输出内部拼写，由 MiMo 解析器识别为真正的工具调用边界。

由此，普通文本与协议控制标记在编码、解码和解析三个环节形成一致的区分。

### 1.1 分析范围

本文以当前分支 `dev/tangzhen/temp-fix-added_token_decode` 的实现为依据，描述 MiMo added-token 命名空间隔离的整体设计。范围包括 tokenizer 初始化、聊天模板、运行时多进程接入、推理与工具调用解析、多模态占位处理及测试覆盖。

## 2. 问题背景与设计目标

### 2.1 原有字符串表示存在歧义

MiMo 的 added tokens 涵盖聊天边界、推理、工具调用及多模态占位等标记，例如：

- 聊天边界：`<|im_start|>`、`<|im_end|>`。
- 推理边界：`<think>`、`</think>`。
- 工具调用：`<tool_call>`、`</tool_call>`、`<tool_response>`。
- 多模态：`<|image_pad|>`、`<|video_pad|>`、`<|mimo_audio_start|>`。

原有表示下，模板生成的协议标记和请求正文中同样的字面量，可能命中同一个 added-token ID。模型输出经过解码后，文本解析器也容易混淆控制标记与普通内容。

这会影响三个层面：

1. **输入语义**：正文中的 EOS、角色或推理标记可能作为结构 token 进入模型，干扰预期语义。这里不意味着“输入中出现 EOS ID 就一定直接触发生成停止”。
2. **输出解析**：用户要求讨论或复述 `<tool_call>`、`<think>` 时，字面文本与协议边界容易混淆。
3. **多模态对齐**：正文中的 `<|image_pad|>` 如果被当作真实图片占位，会影响占位数量与真实媒体数据的对应关系。

### 2.2 目标与约束

| 目标 | 实现约束 |
| --- | --- |
| 区分正文和控制标记 | 模板使用内部拼写，正文不做统一字符串替换 |
| 兼容已有模型权重 | 保持原 token ID 和词表大小，不新增 embedding 行 |
| 保持 token 行为 | 保留 `special`、`normalized`、空白处理等 AddedToken 属性 |
| 覆盖编码与解码 | 同时更新 tokenizer 后端和 Python 侧元数据 |
| 多进程一致 | 各运行时组件加载 tokenizer 时使用同一套规则 |
| 兼容多模态与流式解析 | 同步更新媒体标记、正则和流式边界识别 |
| 支持重复初始化 | 命名转换幂等，允许模板覆盖后的再次规范化 |

这里的 **added token 不等于 `special=True` 的 token**。例如工具调用标记可以是 `special=False`，仍然具有独立 added-token ID，也需要纳入隔离。

## 3. 总体架构与数据流

```text
初始化阶段
  get_tokenizer() / get_processor()
    └─ 识别 MiMo
       └─ patch_mimo_tokenizer()
          ├─ 建立 public spelling → internal spelling 映射
          ├─ 重建 tokenizer 的编码/解码状态，保留 ID
          ├─ 同步 special-token 元数据与初始化配置
          └─ 替换 tokenizer / processor 的聊天模板

请求阶段
  请求正文（保留原文） + 已规范化的 Jinja 模板
    └─ 渲染 prompt
       ├─ 模板控制标记 → 内部拼写 → 原控制 token ID
       └─ 正文字面标记 → 公开拼写 → 普通文本 token 序列
          └─ 多模态处理器按相同命名空间识别真实媒体占位
             └─ 模型推理

输出阶段
  模型输出原控制 token ID
    └─ 已规范化的 detokenizer 解码为内部拼写
       ├─ MiMoReasoningDetector → reasoning / 正文
       └─ MiMoDetector → tool_calls / 正文
          └─ OpenAI 兼容响应
```

控制 token 的内部字符串虽然变长，但仍是原来的单个 token ID。对于模板中的控制标记，这不会因后缀长度而增加模型输入 token 数；正文中的公开标记改为普通文本后，其 token 数可能变化。

## 4. 核心实现：统一命名与 tokenizer 原位改造

核心代码位于 `python/sglang/srt/utils/patch_tokenizer.py`。

### 4.1 唯一命名规则

当前实现使用的后缀常量为：

```python
MIMO_TOKEN_SUFFIX = "_36a8b94076c558de0938f7e4f3cd4f18c5490ca0"
```

下文以 `{suffix}` 表示该完整常量。`mimo_token()` 将后缀插入结束定界符之前：

| 原始拼写 | 内部拼写示意 |
| --- | --- |
| `<\|im_end\|>` | `<\|im_end{suffix}\|>` |
| `<tool_call>` | `<tool_call{suffix}>` |
| `</tool_call>` | `</tool_call{suffix}>` |
| `<think>` | `<think{suffix}>` |
| 无 `>` 结尾的 added token | 直接在末尾追加 `{suffix}` |

函数识别 `|>`、`>` 两种结尾，并检查名称是否已包含当前后缀，避免重复追加。保留尖括号结构也使工具调用标记继续适用于现有 XML/SAX 解析流程。

后缀是代码中的固定常量，不会在每次请求时随机生成。tokenizer、模板和解析器共享该常量，保证内部控制标记的拼写一致。

### 4.2 MiMo 识别与映射生成

`patch_mimo_tokenizer(tokenizer, is_mimo=False)` 在满足任一条件时生效：

1. 调用方显式传入 `is_mimo=True`。
2. `added_tokens_decoder` 中存在内容以 `<|mimo_` 开头的 token。

第二种方式用于识别 `/preset-models/` 等无法从路径看出模型身份的加载场景。没有显式指定、也没有 MiMo 标记时，函数直接返回。

识别成功后，函数遍历 **全部 `added_tokens_decoder` 条目** 建立映射，而不是维护工具调用或 EOS 的局部白名单。因此，实际处理范围也包含模型导出的其他 added tokens。

映射保存在 tokenizer 的 `_sglang_mimo_token_replacements` 属性中，作为模板和多模态模块共享的映射来源。若加载到的 token 已带当前后缀，则反推出公开拼写，使重新加载后的旧格式模板仍可被转换。

### 4.3 重命名流程与不变量

处理顺序如下：

1. 记录原始词表大小与 added-token ID。
2. 根据原 token 的公共字段创建新的 `transformers.AddedToken`，只替换 `content`。
3. 检查目标拼写是否已经属于另一个 ID；有冲突则在修改后端前报错。
4. 更新 fast 或 slow tokenizer 的内部状态。
5. 同步 special-token 属性、`init_kwargs` 和聊天模板。
6. 校验词表大小及每个重命名 token 的 ID，失败则抛出 `ValueError`。

需要保持的不变量为：

```text
len(tokenizer_after) == len(tokenizer_before)
id(internal_spelling) == original_added_token_id
eos_token_id / pad_token_id / special-token ID 集合保持一致
AddedToken 的行为属性保持一致
```

这里采用已有 ID 的重命名，不通过追加新 token 的方式分配新 ID。对权重而言，模板控制 token 和模型生成控制 token 仍指向原有 embedding / 输出词表位置。

### 4.4 Fast、slow 与版本兼容

| 场景 | 实现方式 | 解决的问题 |
| --- | --- | --- |
| Fast tokenizer | 将 `backend.to_str()` 转为 JSON，递归替换精确匹配的 token 拼写，再用 `type(backend).from_str()` 重建 `_tokenizer` | 同步真正参与编码、解码和后处理的后端状态，避免只修改 Python decoder |
| Slow tokenizer | 重建 `_added_tokens_decoder`、`_added_tokens_encoder`，同步 `encoder` / `decoder`，重建 Trie、更新词表大小并清空 cache | 兼容 Qwen 词表中 added token 同时存在于基础词表的情况 |
| fastokens | 从 `_AddedTokenInfo` 与 AddedToken 共有字段重建 token，并沿用原后端类型反序列化 | 保留后端类型与 token 属性 |
| Transformers v4/v5 | 优先读取 `special_tokens_map_extended`；缺失时读取 `_special_tokens_map` 与 `_extra_special_tokens` | 适配 v5 的 special-token 存储变化，避免只读取字符串而丢失属性 |

`_replace_mimo_token_text()` 支持 AddedToken、字符串、字典和列表。处理 tokenizer 状态时，字符串和字典键采用精确匹配，避免把基础词表中的 `prefix<tool_call>suffix` 或路径 `models/<tool_call>/tokenizer` 一并改写。

### 4.5 幂等性与持久化

- 同一个 tokenizer 再次调用 patch 时，复用已有映射，只重新处理聊天模板。
- token 已带当前后缀时不会再次追加，支持 `save_pretrained()` 后重新加载再 patch。
- 正常服务启动仅修改内存对象；保存到磁盘是调用方显式执行 `save_pretrained()` 时的行为。

幂等判断针对**当前后缀**。持久化 tokenizer 与运行时应使用一致的命名规则，当前实现没有跨后缀版本的专门迁移逻辑。

## 5. 聊天模板：只规范化模板源文本

`replace_mimo_chat_template(tokenizer, template)` 使用 tokenizer 上的共享映射，在模板字符串中执行子串替换，也支持列表或字典形式的模板集合。

它与 tokenizer 状态替换的区别是：模板需要替换嵌在 Jinja 源文本中的标记，所以显式启用 `replace_substrings=True`。

例如：

```jinja
{{ '<|im_start|>user\n' + messages[0]['content'] + '<|im_end|>' }}
```

规范化后，模板中的两处控制标记使用内部拼写，`messages[0]['content']` 在渲染时仍原样插入。即使内容包含 `<|im_end|>`，也不会跟随模板一起改写为内部 EOS 标记。

接入点包括：

1. tokenizer 首次 patch 时处理 `tokenizer.chat_template`，以及 `init_kwargs` 中的模板。
2. `get_processor()` 同步处理独立的 `processor.chat_template`。
3. `TemplateManager.load_chat_template()` 在显式模板或默认模板加载完成后再执行一次规范化，覆盖外部 `.jinja` 文件替换原模板的情况。
4. `_detect_reasoning_pattern()` 同时识别原始与内部命名空间下的强制推理前缀。

这个机制是模板源文本替换，不是 Jinja AST 分析：模板源文本中所有匹配片段都会被替换；通过变量在运行时插入的请求正文不会被该函数扫描。

## 6. 运行时接入与多进程一致性

### 6.1 加载入口

| 入口 | MiMo 判定来源 | 初始化顺序 |
| --- | --- | --- |
| `get_tokenizer()` | `is_mimo` 参数，另有 tokenizer 自识别兜底 | HF tokenizer 修复 → MiMo patch → 附加 stop ID → 通用 patch |
| `get_processor()` | `config.model_type.lower()` 是否包含 `mimo`，另有自识别兜底 | 提取 tokenizer → 原有编码修复 → MiMo patch → processor 模板同步 → 附加 stop ID |

以下四个组件在直接调用 `get_tokenizer()` 的分支中统一传入：

```python
is_mimo = (
    server_args.tool_call_parser == "mimo"
    or server_args.reasoning_parser == "mimo"
)
```

- `TokenizerManager`
- `DetokenizerManager`
- `Scheduler`
- `TpModelWorker`

这使输入编码端、输出解码端以及其他持有 tokenizer 的运行时进程能够采用一致的内部拼写，避免只有一侧改名。

### 6.2 开关关系

命名空间隔离没有独立开关。满足 MiMo 识别条件时，加载路径直接执行 patch。

虽然代码放在 `patch_tokenizer.py` 中，但它与原有 `patch_tokenizer()` 是两个独立调用，**不受 `SGLANG_PATCH_TOKENIZER` 控制**。

## 7. 输出协议：工具调用与推理解析同步

### 7.1 MiMo 工具调用

`mimo_detector.py` 统一通过 `mimo_token()` 定义开始、结束标记与 SAX 元素名，用于：

- 非流式工具调用正则，使用 `re.escape()` 构造边界。
- 流式前缀识别与未完整标签缓冲。
- SAX 开始、结束事件中的元素名判断。

工具调用采用内部命名空间下的外层边界，内部结构使用 `<function=...>`、`<parameter=...>` 等语法。解析器同时负责参数类型转换和 freeform 参数处理。

`OpenAIServingChat` 在 `tool_call_parser == "mimo"` 时，将实例上的 `_TOOLCALL_TAGS` 规范化为内部拼写。工具调用 ID 通过以下方式查找：

```python
tokenizer.get_added_vocab().get(self._TOOLCALL_TAGS[0])
```

这样只有实际存在于 added vocab 的内部开始标记才会成为 gate 的依据，避免把未知 token 的兜底 ID 当成有效工具调用 ID。

当前分支还提供两项可选机制，默认均为 `False`：

- `SGLANG_DISCRETIZE_USER_TOOLCALL_TAGS`：在相应文本路径上离散化 user/tool 正文中的目标标签；MiMo 的目标标签随 `_TOOLCALL_TAGS` 切换为内部拼写。
- `SGLANG_REQUIRE_SPECIAL_TOOLCALL_TOKEN`：通过输出 ID 中是否包含工具调用开始 ID 辅助判断。当前 gate 是整段输出级判断，ID 不可用时不阻断。

这两项机制与命名空间隔离配合使用；公开标签变成普通文本的基础能力来自 tokenizer patch。

### 7.2 MiMo 推理解析

`ReasoningParser` 将 `"mimo"` 映射到专用的 `MiMoReasoningDetector`，使用内部 `<think>` / `</think>` 拼写识别推理边界。

流式推理、强制推理以及续写相关参数交由公共基类处理。Qwen3 等其他解析器使用各自的边界定义。

在通常的非强制推理场景下，公开拼写 `<think>literal</think>` 和 `<tool_call>literal</tool_call>` 会保留为正文；模型原控制 ID 解码后的内部标记则进入相应协议解析。

## 8. 多模态：共享映射，保持占位与 ID 一致

`MiMoV2Processor` 通过局部函数 `media_marker(start, pad, end)`，按 tokenizer 的实际映射生成媒体占位字符串及正则：

```text
image: vision_start + image_pad + vision_end
video: vision_start + video_pad + vision_end
audio: mimo_audio_start + audio_pad + mimo_audio_end
```

每组正则匹配 `start + pad（一或多个）+ end`，各标记使用 `re.escape()` 转义。

这里使用 `replace_mimo_chat_template()` 复用**实际存在的 token 映射**，而非直接对所有媒体名称调用 `mimo_token()`。原因是某些 vision-only 导出在 config 中预留 audio ID，但 tokenizer 并没有对应 audio token；没有映射的标记应保持原拼写。

`IM_TOKEN_ID`、`VIDEO_TOKEN_ID`、`AUDIO_TOKEN_ID` 等仍来自原模型配置。后续媒体占位展开、embedding 对齐及相关计算继续使用这些既有 ID。

典型场景是两张真实图片加一段含有字面量 `<|image_pad|>` 的说明文字：模板生成两个内部图片占位，说明文字保持公开拼写，编码结果中真实 `image_id` 的数量仍为 2。system 内容中的字面量 `<|im_end|>` 同理，不增加模板 EOS 控制 ID 的数量。

## 9. 兼容性与适用边界

1. **模型 ID 层兼容，原始文本使用内部表示。** 模型权重及既有控制 ID 继续使用；直接观察原始 decode 输出、或自行解析原始文本的调用方会看到内部拼写，需要采用同一命名规则。标准 reasoning/tool-call 解析链路使用该内部表示。
2. **保留 `skip_special_tokens` 的原语义。** `special=True` 的标记仍可被跳过；`special=False` 的工具标记不会仅因命名空间改造而被跳过。测试对此有明确断言。
3. **隔离的是公开拼写与内部控制表示。** 后缀是公开固定常量，显式输入完整内部拼写仍可能命中控制 ID；直接提交 token ID 也不经过字符串隔离。方案本身不是不可伪造的控制通道。
4. **规范化范围以实际接入路径为准。** 当前入口覆盖 HF tokenizer/processor 加载及聊天模板处理。内置 conversation 模板、自定义绕过加载函数的 tokenizer，以及调用方自行拼接的 raw prompt，需要按各自路径确认控制标记采用正确的内部拼写。
5. **正确性依赖各端命名一致。** tokenizer、detokenizer、MiMo parser 和媒体标记应使用同一版本的后缀规则，该常量构成内部文本协议的一部分。
6. **启动成本与请求成本分开。** 后端重建、词表检查和映射生成发生在初始化阶段；请求正文无需额外做命名空间替换。本文未进行性能测量，启动开销与请求性能需通过基准测试量化。

## 10. 测试设计与证据

`test/registered/unit/utils/test_mimo_tokenizer.py` 使用临时构造的 ByteLevel/Qwen tokenizer 做离线测试，并注册到 `stage-a-test-cpu`。

| 验证维度 | 测试覆盖内容 |
| --- | --- |
| 核心不变量 | 词表大小、EOS/PAD ID、special ID 集合、AddedToken 属性保持一致 |
| 编码与解码 | 内部拼写编码为原 ID；原 ID 解码为内部拼写；公开拼写不再命中对应 ID |
| 后端兼容 | Qwen2Tokenizer、Qwen2TokenizerFast；可选 fastokens；v5 extra special tokens |
| 模板隔离 | 模板控制标记替换，正文保持原文，模板集合及外部模板转换 |
| 幂等与保存 | 重复 patch、模板覆盖后重新 patch、保存再加载 |
| 模型输出链路 | patch 前构造的控制 ID 序列，patch 后解码仍能被 MiMo 推理及工具解析器正确消费 |
| 后处理器 | Fast tokenizer 的 EOS post-processor 与新拼写同步 |
| 多模态字面量 | 两个真实图片占位与正文伪占位区分，system 字面 EOS 不增加控制 ID 数量 |
| 冲突与精确替换 | 命名冲突在修改前报错；基础词表子串及配置路径保持原样 |
| 非 MiMo | 默认不改动；显式 `is_mimo=True` 时启用 |

工具调用测试覆盖以下链路：

- `python/sglang/test/mimo_utils.py` 提供 `mimo_tool_call_text()`，保持 fixture 可读，同时转换为运行时拼写。
- `test_mimo_detector.py` 覆盖工具解析、freeform、流式输出和公开标签不触发解析。
- `test_mimo_streaming_alignment.py` 先转换完整 fixture，再按不同边界切块，包含后缀内部逐字符切分，验证流式与非流式结果一致。
- `test_function_call_parser.py` 使用内部拼写的 MiMo fixture 验证解析行为。
- `test_serving_chat.py` 的 token-level gate 测试通过 tokenizer mock，从 added vocab 查找内部标记。

以上为当前分支测试代码的静态覆盖情况。本文整理过程中未执行测试；真实模型、完整多模态服务和后端版本组合的验证结果需以实际测试为准。

具备项目测试依赖时，可从仓库根目录运行核心离线验证：

```bash
python3 test/registered/unit/utils/test_mimo_tokenizer.py
python3 test/registered/unit/function_call/test_mimo_detector.py
python3 test/registered/unit/function_call/test_mimo_streaming_alignment.py
```

## 11. 设计归纳

整个 feature 将 MiMo added token 的字符串约定收敛到三个公共函数：

| 公共函数 | 职责 |
| --- | --- |
| `mimo_token()` | 定义单个控制标记的内部拼写 |
| `patch_mimo_tokenizer()` | 在初始化时统一重建 token 映射及编码、解码状态 |
| `replace_mimo_chat_template()` | 将共享映射应用到模板与媒体标记 |

各层围绕同一规则协同：**模板负责产生内部控制拼写，tokenizer 负责保持原 ID，detokenizer 输出内部控制拼写，parser 与多模态处理器负责识别它，正文保留公开文本语义。**

当前分支通过共享命名规则和集中式初始化处理，将输入编码、输出解析与多模态占位串联为一致的控制 token 表示体系。在保持模型侧 token ID 的前提下，协议控制标记与正文中的公开字面量具有不同的编码和解析行为。
