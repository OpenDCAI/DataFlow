class KnowledgeCleanerPrompt:
   '''
   知识清洗提示词生成器，支持中英文多语言适配
   Specialized in refining raw content with multilingual support.
   '''
   def __init__(self, lang: str = "en", strict_mode: bool = True):
      self.lang = lang
      self.strict_mode = strict_mode
      # self._init_prompt_header()

#    def _init_prompt_header(self):
#       """根据语言初始化提示词头部模板"""
#       if self.lang == "en":
#          self.prompt_header = f"""
# You are a meticulous Knowledge Refinement Engineer. Apply these rules STRICTLY:

# 1. Remove redundant tags but retain:
# - Semantic tags like <table>, <code>
# - Meaningful attributes

# 2. Normalize special characters:
# - Standardize quotes and dashes
# - Convert ellipsis (...)

# 3. URL handling:
# - Preserve footnote URLs
# - Extract display texts

# 4. Text structure:
# - Maintain paragraph/list breaks
# - Keep code indentation
# - Limit empty lines (max=2)

# 5. Reference processing (NEW):
# - Images → "[Image: alt_text]"
# - Signatures → "[Signature]"

# 6. Code blocks: {"(strict)" if self.strict_mode else ""}
# - {"Force closure" if self.strict_mode else "Preserve raw"}
# - Mark fragments as /*...*/

# 7. Absolute fidelity:
# - NO fact/number modifications
# - NO term paraphrasing
# - NO table structure changes

# 8. Security Processing (NEW):
# - PII: Phone/ID/Email must be masked, e.g. 
#    Original: phone 13800138000 → Processed: phone 138****8000
# - Classified: Mark 【Confidential】as 〖SEC∶classified〗
# - Illegal: Replace sensitive content with 〖ILLEGAL∶removed〗
# - Encryption tags: Use 〖〗for encrypted sections

# Example:
# Input:
# <div class="article">
# <h1>Knowledge Cleaning™</h1>
# <figure>
#    <img src="process.png" alt="Cleaning Flowchart" title="Three Phases">
#    <figcaption>Fig.1: Core Process</figcaption>
# </figure>
# <p>Contact: <span class="phone">+8613800138000</span></p>
# <p>Text with "curly quotes" and – dash – here…</p>
# <table><tr><td>Table data</td></tr></table>
# <pre><code>function test() {{</code></pre>
# <blockquote>Signature: John <img src="sign.png" alt="e-signature"></blockquote>
# <p>Confidential: Project budget is 【Secret】</p>
# <p>Diagram: <img src="demo.jpg" class="diagram"></p>
# </div>

# Output:
# <cleaned_start>
# Knowledge Cleaning™

# [Image: Cleaning Flowchart (Three Phases) Fig.1: Core Process]

# Contact: +86*****8000

# Text with "straight quotes" and - dash - here...

# <table><tr><td>Table data</td></tr></table>

# <code>function test() {{ /*...*/ }}</code>

# [Signature]Signature: John [Image: e-signature]

# 〖SEC∶classified content〗

# Diagram: [Image: Diagram demo.jpg]
# <cleaned_end>
# """
#       else:
#          self.prompt_header =f"""
# 你是一名严谨的知识清洗工程师。请严格按照以下规则处理原始内容：

# 1. 移除冗余HTML/XML标签，但保留：
# - 语义化标签如 <table>、<code>、<formula>
# - 所有携带意义的属性值

# 2. 规范化特殊字符：
# - 将花引号（“ ” ‘ ’）转为标准引号（" "）
# - 将长破折号（– —）替换为短横线（-）
# - 中文省略号（…）转为英文省略号（...）
# - 保留数学符号和技术记号（如<<、>>等操作符）

# 3. 链接处理：
# - 脚注/参考文献中的URL保持原样
# - 移除超链接包装但保留显示文本
# 示例：<a href="https://example.com">示例</a> → 示例

# 4. 文本结构：
# - 保持原始段落/列表的换行
# - 保留代码/引用的缩进层级
# - 压缩连续空行为最多2行

# 5. 引用内容处理（新增）：
# - 图片引用转换为【引用图片：描述文本】
# - 签名区块标记为【签名引用】

# 6. 代码块处理：{"（严格模式）" if self.strict_mode else ""}
# - {"确保代码块闭合（如补全缺失的括号）" if self.strict_mode else "保持代码原样"}
# - 标记不完整代码为/*...*/

# 7. 绝对保真：
# - 禁止增删任何事实、数字或命名实体
# - 禁止改写专业术语或专有名词
# - 禁止修改表格数据结构

# 8. 安全处理（新增）：
# - 个人隐私：身份证号/手机号/邮箱等需脱敏，示例：
#    原文本：电话 13800138000 → 处理后：电话 138****8000
# - 涉密内容：检测到【机密】【秘密】等关键词时，整句替换为【涉密内容已加密】
# - 违规信息：政治敏感、暴恐等内容替换为【违规内容已屏蔽】
# - 加密标记：使用〖〗包裹加密区域，示例：
#    〖PII∶身份证号〗〖SEC∶机密字段〗

# 示例：
# 输入：
# <div class="article">
# <h1>知识清洗®</h1>
# <figure>
#    <img src="process.png" alt="清洗流程图" title="三阶段处理">
#    <figcaption>图1：核心流程</figcaption>
# </figure>
# <p>联系电话：<span class="phone">13800138000</span></p>
# <p>这是包含"花引号"和—破折号—的文本…</p>
# <table><tr><td>表格数据</td></tr></table>
# <pre><code>function test() {{</code></pre>
# <blockquote>签名：张三 <img src="sign.png" alt="电子签名"></blockquote>
# <p>机密信息：本项目预算为【机密】</p>
# <p>示意图：<img src="demo.jpg" class="diagram"></p>
# </div>

# 输出：
# <cleaned_start>
# 知识清洗®

# [引用图片：清洗流程图（三阶段处理）图1：核心流程]

# 联系电话：138****8000

# 这是包含"花引号"和-破折号-的文本...

# <table><tr><td>表格数据</td></tr></table>

# <code>function test() {{ /*...*/ }}</code>

# [签名引用]签名：张三 [引用图片：电子签名]

# 涉密内容已加密

# 示意图：[引用图片：示意图demo.jpg]
# <cleaned_end>
# """

   def build_prompt(self, raw_content: str) -> str:
      """生成知识清洗的思维链提示词（保持原有格式）"""
      if self.lang == "en":
         self.prompt_header = f"""
You are a meticulous Knowledge Refinement Engineer. Apply these rules STRICTLY:

1. Remove redundant tags but retain:
- Semantic tags like <table>, <code>
- Meaningful attributes

2. Normalize special characters:
- Standardize quotes and dashes
- Convert ellipsis (...)

3. URL handling:
- Preserve footnote URLs
- Extract display texts

4. Text structure:
- Maintain paragraph/list breaks
- Keep code indentation
- Limit empty lines (max=2)

5. Reference processing (NEW):
- Images → "[Image: alt_text]"
- Signatures → "[Signature]"

6. Code blocks: {"(strict)" if self.strict_mode else ""}
- {"Force closure" if self.strict_mode else "Preserve raw"}
- Mark fragments as /*...*/

7. Absolute fidelity:
- NO fact/number modifications
- NO term paraphrasing
- NO table structure changes

8. Security Processing (NEW):
- PII: Phone/ID/Email must be masked, e.g. 
   Original: phone 13800138000 → Processed: phone 138****8000
- Classified: Mark 【Confidential】as 〖SEC∶classified〗
- Illegal: Replace sensitive content with 〖ILLEGAL∶removed〗
- Encryption tags: Use 〖〗for encrypted sections

Example:
Input:
<div class="article">
<h1>Knowledge Cleaning™</h1>
<figure>
   <img src="process.png" alt="Cleaning Flowchart" title="Three Phases">
   <figcaption>Fig.1: Core Process</figcaption>
</figure>
<p>Contact: <span class="phone">+8613800138000</span></p>
<p>Text with "curly quotes" and – dash – here…</p>
<table><tr><td>Table data</td></tr></table>
<pre><code>function test() {{</code></pre>
<blockquote>Signature: John <img src="sign.png" alt="e-signature"></blockquote>
<p>Confidential: Project budget is 【Secret】</p>
<p>Diagram: <img src="demo.jpg" class="diagram"></p>
</div>

Output:
<cleaned_start>
Knowledge Cleaning™

[Image: Cleaning Flowchart (Three Phases) Fig.1: Core Process]

Contact: +86*****8000

Text with "straight quotes" and - dash - here...

<table><tr><td>Table data</td></tr></table>

<code>function test() {{ /*...*/ }}</code>

[Signature]Signature: John [Image: e-signature]

〖SEC∶classified content〗

Diagram: [Image: Diagram demo.jpg]
<cleaned_end>
"""
      else:
         self.prompt_header =f"""
你是一名严谨的知识清洗工程师。请严格按照以下规则处理原始内容：

1. 移除冗余HTML/XML标签，但保留：
- 语义化标签如 <table>、<code>、<formula>
- 所有携带意义的属性值

2. 规范化特殊字符：
- 将花引号（“ ” ‘ ’）转为标准引号（" "）
- 将长破折号（– —）替换为短横线（-）
- 中文省略号（…）转为英文省略号（...）
- 保留数学符号和技术记号（如<<、>>等操作符）

3. 链接处理：
- 脚注/参考文献中的URL保持原样
- 移除超链接包装但保留显示文本
示例：<a href="https://example.com">示例</a> → 示例

4. 文本结构：
- 保持原始段落/列表的换行
- 保留代码/引用的缩进层级
- 压缩连续空行为最多2行

5. 引用内容处理（新增）：
- 图片引用转换为【引用图片：描述文本】
- 签名区块标记为【签名引用】

6. 代码块处理：{"（严格模式）" if self.strict_mode else ""}
- {"确保代码块闭合（如补全缺失的括号）" if self.strict_mode else "保持代码原样"}
- 标记不完整代码为/*...*/

7. 绝对保真：
- 禁止增删任何事实、数字或命名实体
- 禁止改写专业术语或专有名词
- 禁止修改表格数据结构

8. 安全处理（新增）：
- 个人隐私：身份证号/手机号/邮箱等需脱敏，示例：
   原文本：电话 13800138000 → 处理后：电话 138****8000
- 涉密内容：检测到【机密】【秘密】等关键词时，整句替换为【涉密内容已加密】
- 违规信息：政治敏感、暴恐等内容替换为【违规内容已屏蔽】
- 加密标记：使用〖〗包裹加密区域，示例：
   〖PII∶身份证号〗〖SEC∶机密字段〗

示例：
输入：
<div class="article">
<h1>知识清洗®</h1>
<figure>
   <img src="process.png" alt="清洗流程图" title="三阶段处理">
   <figcaption>图1：核心流程</figcaption>
</figure>
<p>联系电话：<span class="phone">13800138000</span></p>
<p>这是包含"花引号"和—破折号—的文本…</p>
<table><tr><td>表格数据</td></tr></table>
<pre><code>function test() {{</code></pre>
<blockquote>签名：张三 <img src="sign.png" alt="电子签名"></blockquote>
<p>机密信息：本项目预算为【机密】</p>
<p>示意图：<img src="demo.jpg" class="diagram"></p>
</div>

输出：
<cleaned_start>
知识清洗®

[引用图片：清洗流程图（三阶段处理）图1：核心流程]

联系电话：138****8000

这是包含"花引号"和-破折号-的文本...

<table><tr><td>表格数据</td></tr></table>

<code>function test() {{ /*...*/ }}</code>

[签名引用]签名：张三 [引用图片：电子签名]

涉密内容已加密

示意图：[引用图片：示意图demo.jpg]
<cleaned_end>
"""

      if self.lang == "en":
         processing_steps = """
Processing Steps:
1. [Tag Analysis] Classify markup tags
2. [Reference Extraction] Isolate images/tables
3. [Character Audit] Log special chars
4. [Structure Check] Validate hierarchy
5. [Final Output] Generate cleaned text
""".strip()
         output_requirement = 'Response must contain ONLY cleaned text between <cleaned_start> and <cleaned_end>.'
      else:
         processing_steps = """
处理步骤：
1. [标签分析] 识别并分类所有标记标签
2. [引用提取] 分离图片/表格/签名等引用内容
3. [字符审核] 记录特殊字符变更
4. [结构检查] 验证文本层级
5. [最终输出] 生成清洗后文本
""".strip()
         output_requirement = '响应必须只包含清洗后文本，以<cleaned_start>开头，<cleaned_end>结尾，无其他内容。'

      return f"""
{self.prompt_header}

待清洗内容：
{raw_content}

{processing_steps}

{output_requirement}
""".strip()

   def post_process(self, cleaned_text: str) -> str:
      """后处理逻辑（新增引用校验）"""
      if self.strict_mode:
         # 校验引用标记完整性
         cleaned_text = re.sub(r'(!$$.*?$$)$.+?$', 
                              lambda m: f"【引用图片：{m.group(1)[2:-1]}" if "图片" in m.group(1) else m.group(0), 
                              cleaned_text)
      return cleaned_text
   
class MathbookQuestionExtractPrompt:
   def __init__(self):
      pass

   def build_prompt(self):
      PROMPT = """You are given a collection of images:

• page_n.jpg – the n-th page of a math textbook  
• page_n+1.jpg – the (n+1)-th page of the same book  
• index.jpg files (e.g. 1.jpg, 2.jpg, …) – all figures, diagrams or illustrations appearing on those two pages  

Your task:

1. Extract every exercise (math problem) that has at least one line or element on page_n.jpg. You should extract the problem in its original language, do not translate it.
2. If a problem is split across page_n.jpg and page_n+1.jpg, include it in full (using page_n+1.jpg only to complete it).  
3. Do not extract any problem that appears exclusively on page_n+1.jpg.  
4. For each extracted problem, locate any referenced figures among the index.jpg files and insert the exact filename in <image>...</image> (for example <image>3.jpg</image>) at the correct place in the problem text.  
5. Return all extracted problems concatenated into one string, using the literal token <SPACE> to separate them. For example:  
   PROBLEM_TEXT_1<SPACE>PROBLEM_TEXT_2<SPACE>PROBLEM_TEXT_3  
6. If no qualifying problems are found on page_n.jpg, return two consecutive spaces: "<SPACE><SPACE>".  

Ensure that figure tags exactly match the provided image filenames and that no extra separators or punctuation are added.
      """
      return PROMPT