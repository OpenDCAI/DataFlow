
# PR #572 新增算子审查清单

审查基线：PR head 49f2233；以 45a9a62 对 get_desc 的要求为准。

45a9a62 要求 get_desc 准确描述所有输入参数、默认值、run 实际写入的每个列、输出值语义、是否过滤行以及返回值；不能包含不存在的参数、过期默认值或错误返回值。

## 总体结论

- 新增算子共 10 个：3 个格式转换/生成算子、5 个结构评估算子、2 个过滤算子。
- input_/output_ 参数命名总体符合框架规则。
- generate、eval、filter 目录归类总体合理。
- 10 个新增算子的 get_desc 都没有完全达到 45a9a62 的完整度。
- ValidationFlagFilter 已与实际的通用布尔验证列过滤职责一致。
- validate_preference 是高度领域化的 preference/DPO 校验函数，不适合继续放在通用 training_data 模块中；但 preference 算子归入 text_sft 是得体的。
- check_keys 有通用价值；evaluate_rows 有复用价值，但属于训练数据领域抽象，不是完全通用的 DataFlow 原语。

## 新增算子清单

### 1. QAMessagesGenerator

文件：[dataflow/operators/text_sft/generate/qa_messages_generator.py](dataflow/operators/text_sft/generate/qa_messages_generator.py)

签名：

~~~python
__init__(system_prompt=None)
run(storage, input_key="prompt", input_answer_key="answer",
    output_key="messages")
~~~

功能和输入输出：

- 读取 prompt、answer 两个非空字符串字段。
- 写入 output_key，默认 messages。
- 输出是 user/assistant message 列表，可在最前面插入 system message。
- 保留源字段、行数和顺序。
- 任一记录非法时整批失败，不过滤记录。
- 返回 [output_key]。

命名：

- 基本符合 generate/*Generator 习惯。
- 这是确定性的格式转换，不调用模型；Generator 可能让人误以为会生成 QA 内容。
- 若项目坚持 generate 下统一使用 Generator，可以保留；若强调语义，QAMessagesFormatter 或 QAConversationGenerator 更直白。

重复性：

- 不等价于 core_text 的 Text2QAGenerator，后者调用 LLM 生成新的 QA。
- 与 QAStructureSampleEvaluator 重复 prompt/answer 校验，应共享 QA 构造/校验函数。

get_desc 缺失：

- messages 的完整结构。
- system_prompt 的构造函数语义、插入位置和冲突规则。
- 不过滤、整批失败和返回值。

### 2. AlpacaMessagesGenerator

文件：[dataflow/operators/text_sft/generate/alpaca_messages_generator.py](dataflow/operators/text_sft/generate/alpaca_messages_generator.py)

签名：

~~~python
__init__(system_prompt=None)
run(storage, input_key="instruction", input_answer_key="output",
    output_key="messages", input_context_key=None,
    input_history_key=None, input_system_key=None)
~~~

功能和输入输出：

- instruction、output 是必需字段。
- context、history、system 默认关闭，不自动探测。
- context 启用后与 instruction 用两个换行连接。
- history 启用后读取 [question, answer] pair 列表并放在当前 instruction 前。
- system 可来自显式列或 constructor system_prompt。
- 写入 messages，保留源字段、行数和顺序。
- 非法内容或输出列已存在时失败，不过滤记录。
- 返回 [output_key]。

命名：

- 符合 Alpaca 格式转换和 generate 分类。
- MessagesGenerator 略有“生成内容”的歧义，但可接受。

重复性：

- 没有发现功能等价的现有注册算子。
- 与 AlpacaStructureSampleEvaluator 共享 optional/history/system 规则，应复用同一纯函数。

get_desc 缺失：

- optional 字段类型和启用条件。
- history 结构。
- context 拼接规则。
- source system 与 system_prompt 冲突行为。
- messages schema、行行为和返回值。

### 3. ShareGPTMessagesGenerator

文件：[dataflow/operators/conversations/generate/sharegpt_messages_generator.py](dataflow/operators/conversations/generate/sharegpt_messages_generator.py)

签名：

~~~python
__init__(system_prompt=None)
run(storage, input_key="conversations", output_key="messages",
    input_system_key=None)
~~~

功能和输入输出：

- 读取 ShareGPT from/value 列表。
- human/gpt/system 映射为 user/assistant/system。
- 可选读取顶层 system，也可使用 system_prompt。
- 拒绝未知嵌套字段、权重字段和多模态字段。
- 写入 messages，保留源字段、行数和顺序。
- 不过滤，非法内容整批失败。
- 返回 [output_key]。

命名：

- 名称直白，能够看出输入方言和输出目标。
- 不建议改成泛化的 ConversationGenerator，否则会隐藏输入方言。

重复性：

- 没有发现等价的 ShareGPT 转换算子。
- 与 ShareGPTStructureSampleEvaluator 共享几乎全部解析逻辑，应使用同一转换函数。

get_desc 缺失：

- messages role/content schema。
- input_system_key=None 的含义。
- 已有 system、顶层 system、system_prompt 的冲突规则。
- 未知字段、权重、多模态字段的处理结果。
- 不过滤和返回值。

### 4. ConversationStructureSampleEvaluator

文件：[dataflow/operators/conversations/eval/conversation_structure_sample_evaluator.py](dataflow/operators/conversations/eval/conversation_structure_sample_evaluator.py)

签名：

~~~python
__init__(profile="conversation")
run(storage, input_key="messages",
    output_valid_key="conversation_valid",
    output_error_key="conversation_error")
~~~

功能和输入输出：

- profile=conversation：校验 system/user/assistant/tool 角色顺序、tool call ID、tool response 配对、arguments JSON 和完整 assistant 结尾。
- profile=single_turn：校验文本型单轮 user/assistant，可选首个 system。
- 写入 conversation_valid: bool。
- 写入 conversation_error: None 或包含 field/code/message/row 的错误对象；有 source location 时还包含 source_location。
- 保留全部行，不过滤。
- 返回两个输出列名。

命名：

- 名称清楚，是新增算子中最符合功能的命名之一。

重复性：

- 不等价于 FuncCallConversationSampleEvaluator；后者是 LLM 质量评估。
- validate_messages/validate_conversation 可以保留为底层规则，但错误序列化应集中处理。

get_desc 缺失：

- 两个 profile 的差异。
- tool call 校验边界，以及不校验工具实际执行结果的边界。
- 两个输出列的完整 schema。
- 不过滤和返回值。

### 5. ValidationFlagFilter

文件：[dataflow/operators/conversations/filter/validation_flag_filter.py](dataflow/operators/conversations/filter/validation_flag_filter.py)

签名：

~~~python
run(storage, input_key="source_valid")
~~~

功能和输入输出：

- 读取任意布尔验证列。
- 删除列值为 False 的记录。
- 非 bool 值直接报错。
- 不新增列，保留剩余内容和顺序。
- 返回 []。

命名审查：

- 已改为通用名称。实现不检查 conversation，可处理任意布尔验证列。
- 示例还用它过滤 source_valid，说明真实职责是通用布尔验证过滤。
- 默认输入列为 `source_valid`；conversation 结果列需要显式传入 `input_key="conversation_valid"`。

重复性：

- 它不是新的过滤算法，只是按验证标记过滤。
- 与 PreferenceStructureFilter 实现完全相同，仅默认列名不同，应共享底层实现。

### 6. QAStructureSampleEvaluator

文件：[dataflow/operators/text_sft/eval/qa_structure_sample_evaluator.py](dataflow/operators/text_sft/eval/qa_structure_sample_evaluator.py)

签名：

~~~python
__init__(system_prompt=None)
run(storage, input_key="prompt", input_answer_key="answer",
    output_valid_key="source_valid", output_error_key="source_error")
~~~

功能和输入输出：

- 检查 prompt、answer 是否存在且为非空字符串。
- 写入 source_valid: bool。
- 写入 source_error: None 或错误对象。
- 保留全部行，不过滤。
- 返回两个输出列名。

命名审查：

- QAStructureSampleEvaluator 可以理解，但实际只做 QA 字段和文本合法性检查。
- QASourceSampleEvaluator 或 QATextStructureSampleEvaluator 更精确。
- 若项目统一使用 StructureSampleEvaluator，可保留但必须在描述中缩小语义范围。

重复性：

- 与 QAMessagesGenerator 重复 prompt/answer 校验。
- 应共享 validate_qa_record 或 build_qa_messages。

get_desc 必须修正：

- 删除不存在的 optional 字段说明。
- system_prompt 当前不会改变正常校验结果，建议删除或实现构造后校验。
- 明确 source_valid=True 与 source_error=None 的关系，以及错误对象 schema。

### 7. AlpacaStructureSampleEvaluator

文件：[dataflow/operators/text_sft/eval/alpaca_structure_sample_evaluator.py](dataflow/operators/text_sft/eval/alpaca_structure_sample_evaluator.py)

签名：

~~~python
__init__(system_prompt=None)
run(storage, input_key="instruction", input_answer_key="output",
    input_context_key=None, input_history_key=None,
    input_system_key=None, output_valid_key="source_valid",
    output_error_key="source_error")
~~~

功能和输入输出：

- 校验 Alpaca required 字段和显式启用的 optional 字段。
- 复用 instruction/output/context/history 的转换规则。
- 写入 source_valid 和 source_error。
- 不删行，返回两个输出列名。

命名审查：

- 基本符合格式 + 结构评估的命名逻辑。
- 如果 QA evaluator 改名为 QASourceSampleEvaluator，本类也应改为 AlpacaSourceSampleEvaluator，保持成组一致。

重复性：

- 与 AlpacaMessagesGenerator 共享几乎全部源字段解析逻辑。
- 两者保留为两个 pipeline 步骤合理，但底层规则不应实现两遍。

get_desc 必须补充：

- optional 字段类型。
- history 结构。
- system 冲突规则。
- 输出错误 schema。
- 不删行和返回值。

### 8. ShareGPTStructureSampleEvaluator

文件：[dataflow/operators/conversations/eval/sharegpt_structure_sample_evaluator.py](dataflow/operators/conversations/eval/sharegpt_structure_sample_evaluator.py)

签名：

~~~python
__init__(system_prompt=None)
run(storage, input_key="conversations", input_system_key=None,
    output_valid_key="source_valid", output_error_key="source_error")
~~~

功能和输入输出：

- 校验 ShareGPT from/value 列表。
- 校验角色顺序和完整对话结构。
- 写入 source_valid 和 source_error。
- 不删行，返回两个输出列名。

命名审查：

- 名称直白，能看出输入方言和 per-row evaluation。
- 应与 ShareGPTMessagesGenerator 使用同一套术语：from/value、role/content、source system。

重复性：

- 与 ShareGPTMessagesGenerator 共享格式解析。
- 两者是同一能力的两个 pipeline 阶段，不应出现两套规则。

get_desc 必须补充：

- 输入方言。
- 转换与校验的边界。
- 错误 schema。
- system 冲突和行行为。

### 9. PreferenceStructureSampleEvaluator

文件：[dataflow/operators/text_sft/eval/preference_structure_sample_evaluator.py](dataflow/operators/text_sft/eval/preference_structure_sample_evaluator.py)

签名：

~~~python
run(storage, input_prompt_key="prompt",
    input_chosen_key="chosen", input_rejected_key="rejected",
    output_valid_key="preference_valid",
    output_error_key="preference_error")
~~~

功能和输入输出：

- 检查三个字段名非空且互不相同。
- 检查每条记录的 prompt/chosen/rejected 都是非空字符串。
- 写入 preference_valid 和 preference_error。
- 不过滤行，保留两种答案原文。
- 返回两个输出列名。

命名审查：

- 名称清晰，符合 preference 结构检查的直觉。
- SampleEvaluator 符合 per-row 写入评估字段的现有命名。

重复性：

- 没有发现功能等价的结构校验算子。
- 与模型型 preference evaluator 不同：本算子只检查数据是否可用，不判断答案优劣。

get_desc 必须补充：

- 明确不做答案优劣判断。
- 明确不做多轮/多模态转换。
- 精确描述错误对象和不删行行为。

### 10. PreferenceStructureFilter

文件：[dataflow/operators/text_sft/filter/preference_structure_filter.py](dataflow/operators/text_sft/filter/preference_structure_filter.py)

签名：

~~~python
run(storage, input_key="preference_valid")
~~~

功能和输入输出：

- 根据 preference 验证布尔列过滤 False。
- 不新增列，保留剩余行内容和顺序。
- 返回 []。

命名审查：

- 如果只处理 preference_valid，名字是直白的。
- 但实现与 ValidationFlagFilter 完全相同，说明过滤算法本身是通用的；建议抽出通用实现，领域类只作为可选包装。

## 公共实现审查

### check_keys

位置：[dataflow/utils/training_data.py](dataflow/utils/training_data.py)

职责：检查字段名是否为非空字符串。

结论：保留。

- 有必要：所有新增算子都需要相同的配置校验。
- 足够通用：不依赖具体格式，也不读取 Storage。
- 名称略泛，validate_field_names 或 validate_key_names 更直白。
- 必须在 docstring 中说明它只检查字段名配置，不检查 DataFrame 列是否存在。
- 列是否存在继续由 require_columns 负责。

### validate_preference

位置：[dataflow/utils/training_data.py](dataflow/utils/training_data.py)

职责：固定检查 prompt/chosen/rejected 三个字段的非空字符串。

结论：实现可以保留，但当前位置不合理。

- 功能非常聚焦，只适用于字符串型 DPO/preference 三元组。
- 当前只有一个注册算子使用它，不能证明它是通用框架抽象。
- 不支持多轮 preference、多模态 preference 或 message 列表。
- 建议移动到 text_sft 私有 helper 或 training_data_formats.py。
- 如果需要通用层，只提供 validate_required_text_fields(row, keys)，再由 text_sft preference 薄封装 validate_preference。

### evaluate_rows

位置：[dataflow/utils/training_operator_batch.py](dataflow/utils/training_operator_batch.py)

职责：

- 一次读取 DataFrame。
- 检查输入列和输出列冲突。
- 对每行运行 validator。
- 捕获 TrainingDataError，输出 valid/error 两列。
- 保留所有行并一次性写回。
- 使用 source locations 和 batch metadata 填充错误位置。

结论：保留，但明确为训练数据领域抽象。

- 有真实复用价值：五类 evaluator 的外围流程高度一致。
- 不完全通用：固定依赖 TrainingDataError 的 field/code/message、valid/error 双列、DataFrame Storage、source locations、batch_step/batch_size。
- 更像训练数据结构评估器内部框架，而不是通用 DataFlow evaluator 原语。
- 当前直接读取、逐行评估并一次写回，避免临时 Batch adapter；仍保留一次读取和 source location/batch 行号语义。
- 错误序列化仍固定为训练数据评估所需的 field/code/message/row 结构，暂不扩展为通用可注入策略。

### 相关 helper

- transform_rows：相对通用，适合行数不变、增加新列、整批成功后写入。
- filter_validation_rows：高度聚焦于布尔验证列过滤，不是一般 filter framework。
- require_text：通用性较好，可作为非空字符串校验原语。
- convert_alpaca、convert_sharegpt、map_preference_fields：格式/领域 helper，不应伪装成通用框架抽象。

## 针对公共 import 的进一步结论

这里需要区分“减少重复代码”和“建立通用框架抽象”。当前两个 import 的价值并不相同：

| import | 当前使用范围 | 实际职责 | 结论 |
| --- | --- | --- | --- |
| `validate_preference` | 仅 `PreferenceStructureSampleEvaluator` | 固定检查 `prompt/chosen/rejected` 三列为非空字符串 | 不应放在通用 `training_data.py`；应删除，直接复用 `require_text`，或放在 preference 算子同目录的私有 helper 中 |
| `check_keys` | 多个新增算子，以及 `record_execution`、`training_data_formats` | 检查配置传入的字段名是非空字符串 | 有一定通用价值，但它不是训练数据校验；可保留一个小 helper，建议改名为 `validate_field_names`，并放入更中性的字段/算子校验模块 |
| `evaluate_rows` | 5 个结构 evaluator | 读取 DataFrame、逐行校验、捕获 `TrainingDataError`、写入 valid/error 两列 | 有真实复用价值，但边界是“训练数据结构评估”；不是 DataFlow 通用 evaluator 原语 |

### `validate_preference` 的建议

这个函数目前只有一层很薄的包装：

~~~python
for key in (prompt, chosen, rejected):
    require_text(row, key)
~~~

它没有 preference 特有的规则、错误码或结构处理。保留它反而让公共模块看起来支持一个尚未定义的 preference 框架。建议在 evaluator 内直接调用 `require_text`；如果未来出现多种 preference schema，再按 schema 在 `text_sft` 下建立明确的校验器。

### `check_keys` 的建议

`check_keys` 本身足够小，也确实被非 preference 代码复用，因此不必为了“抽象”再包一层。需要调整的是语义位置和名字：它只校验参数字符串，不检查 DataFrame 是否有这些列，也不校验列内容。建议：

1. 保留单一实现，避免每个算子重复 `isinstance`/空字符串判断；
2. 改名为 `validate_field_names`（或在短期兼容期保留 `check_keys` 别名）；
3. 将列存在性继续交给 `require_columns`，将记录内容继续交给 `require_text`/具体格式校验器；
4. 不要把它和 `validate_preference` 放在同一“训练数据规则”层级。

另外，当前新增算子普遍在 `run` 开头显式调用 `check_keys`，随后又调用 `transform_rows`、`evaluate_rows` 或 `filter_validation_rows`；这些批处理 helper 内部已经通过 `require_columns`/输出列检查再次调用了同一校验。因此大多数 import 和调用是重复的。更简洁的做法是让批处理 helper 统一负责字段名和列存在性检查，算子本身只保留领域约束，例如输入输出字段必须不同、profile 值必须受限。这样可以删除新增算子中的大部分 `check_keys` import，而不降低校验强度。

### `evaluate_rows` 的建议

它的外围模式是合理的：多个 evaluator 都是“保留所有行，新增布尔结果列和错误列”。当前实现直接完成读取、评估和写回，阅读路径是：

~~~text
evaluator -> evaluate_rows -> storage
~~~

`evaluate_rows` 现在是一个直接的训练数据 helper：

- 一次读取并检查输入列；
- 逐行调用 `validator`；
- 直接构造两个输出列；
- 保留 `source_location` 和 batch 行号计算；
- 一次写回并返回 `[output_valid_key, output_error_key]`。

参数使用关键字参数，例如 `validator=...`、`output_valid_key=...`、`output_error_key=...`、`input_columns=...`，避免调用点难以辨认。

### 推荐的最小收敛方案

当前 PR 不需要再增加一套大而全的公共框架。建议只做以下三步：

1. 删除 `validate_preference` import，在 preference evaluator 内直接循环调用 `require_text`；
2. 保留 `check_keys` 一个实现，并用 docstring 明确它是字段名配置校验；
3. 保留直接实现的训练数据 `evaluate_rows`，不再增加额外 adapter 或通用 evaluator 层。

这样既消除了 preference 专用抽象，也避免把五个 evaluator 复制成五份，同时和现有算子常见的“run 负责声明列、helper 负责一类稳定批处理语义”的规约保持一致。

## `require_text` 与既有算子的统一性

不建议为了“和全仓库所有旧算子一致”而删除或改写 `require_text`。仓库中既有算子对文本字段的处理并不统一：有的严格拒绝非字符串和空白值，有的只检查类型，有的将数值转成字符串，有的会先做 `strip()`。这些行为对应不同的产品语义，不能机械合并。

`require_text` 在新增训练数据算子内部是相对一致的严格规则：

- 字段不存在：`missing_field`；
- 值不是字符串：`invalid_type`；
- 字符串全为空白：`empty_content`；
- 返回原始字符串，不隐式 `strip()` 或修改内容；
- 通过 `TrainingDataError` 进入统一的 `field/code/message/row` 错误结构。

这套规则适合 QA、Alpaca、ShareGPT 和字符串型 preference 这类“格式转换/结构评估”算子，因为这些算子需要保证输出训练样本的结构有效。它不适合直接替换所有历史算子的宽松行为，否则会把原本允许的数值、空字符串或自动字符串化行为改成失败。

建议保持三层职责清晰：

1. `validate_field_names`/`check_keys`：检查算子参数中的字段名配置；
2. `require_columns`：检查 DataFrame 中是否存在声明的列；
3. `require_text` 或具体格式校验器：检查单条记录的值和结构。

其中 `optional_text` 与 `require_text` 的差异是有意的：可选字段允许缺失、`None`、`NaN` 或空白文本并视为未提供；必需字段则必须是非空字符串。应在公共 helper 的 docstring 和 `get_desc` 中明确这一点。

因此，统一目标应限定为“新增训练数据算子内部统一”，不应扩大为“全仓库所有算子强制使用 `require_text`”。

## 建议的重构方向

保留：

- check_keys，或改名后的字段名校验函数。
- require_text。
- transform_rows。
- 训练数据领域内的 evaluate_rows。

收敛：

- QA：统一 validate_qa_record/build_qa_messages。
- Alpaca：统一 required/optional/history/system 规则。
- ShareGPT：统一 from/value 解析和 conversation 结构验证。
- system prompt：统一非空校验和冲突处理。
- 两个结构过滤器：统一底层布尔过滤实现。

避免：

- 不要把 validate_preference 扩展成通用 preference 框架，除非同时定义多轮、多模态和不同 preference schema。
- ValidationFlagFilter 明确承担通用布尔验证列过滤职责；PreferenceStructureFilter 保留 preference 领域入口。
- 不要通过大量重复 prose 维护 get_desc；应使用统一模板或元数据生成描述，再保留领域特有说明。

## 最终 Review Checklist

### get_desc

- [x] 10 个新增算子全部采用统一的 Input/Output/Row behavior/Return/Errors 结构。
- [x] 所有 run 输入参数、构造函数参数和默认值准确列出。
- [x] 所有实际写入列都列出名称、默认值、类型和内容语义。
- [x] 明确 evaluator 不删行、generator 不删行、filter 删除哪些行。
- [x] 明确每个算子的真实返回值：字段名列表或空列表。
- [x] 删除不存在的 optional 参数说明。
- [x] 删除或修正 evaluator 中无效的 system_prompt 说明。
- [x] 中英文描述语义一致。

### 命名和分类

- [ ] 决定 QAMessagesGenerator/AlpacaMessagesGenerator/ShareGPTMessagesGenerator 是否接受“Generator 表示确定性格式转换”的约定。
- [ ] 决定 *StructureSampleEvaluator 是否改为更精确的 *SourceSampleEvaluator。
- [x] 将 ConversationStructureFilter 改为通用的 ValidationFlagFilter，并同步默认字段和调用方。
- [x] preference 算子归入 text_sft，并同步更新导出、文档和测试引用。

### 重复和公共实现

- [x] QA evaluator/generator 共享同一套 QA 规则。
- [x] Alpaca evaluator/generator 共享同一套 optional/history/system 规则。
- [x] ShareGPT evaluator/generator 共享同一套 from/value 解析规则。
- [x] 两个结构过滤器共享底层过滤实现。
- [x] 删除 `validate_preference`，由 preference evaluator 直接复用 `require_text`。
- [x] evaluate_rows 的训练数据专用边界写入 docstring 和文档。

### 合并前验证

- [x] 合并最新 main，确认包含 45a9a62。
- [ ] 重新运行 CPU/API/批处理/stream 测试。
- [x] 人工核对 get_desc 与实际写入列、默认值和 `None` 语义；按要求不新增 get_desc 回归测试。
- [x] 增加默认 optional 参数不会产生 compile warning 的测试。
- [ ] 评估是否将训练数据算子与 resilient JSONL/checkpoint 功能拆成两个 PR。

## 审查结论

新增算子中真正具有独立产品价值的是三类能力：

1. QA/Alpaca/ShareGPT 格式转换；
2. messages/conversation/preference 结构评估；
3. 显式验证结果过滤。

这些能力与已有 LLM 生成、LLM 质量评估和模型过滤算子并不等价，保留是合理的。当前 `get_desc` 已补齐默认值、`None` 语义、输出列、行行为、错误和返回值；公共 helper 也已收敛为清晰的配置检查、列检查、行校验和批处理层次。剩余工作主要是完整 CPU/API/批处理/stream 测试。

完成上述清单前，建议继续 Request changes，不建议直接 Approve。
