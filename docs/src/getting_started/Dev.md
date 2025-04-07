# 开发流程

## 运行 Pipeline 示例

运行以下命令以执行 pipeline 示例：

```bash
python process.py --config configs/process/text_process_example.yaml
```

### 注意事项

- 在 `config` 文件夹下编写自己的测试 YAML 文件。
- 提交代码时，**只需在示范 YAML 中写调用方式**，不要提交自己测试的 YAML 文件。

---

## 算子分类

目前算子分为以下四大类：

1. **deduplicator**
2. **filter**
3. **refiner**
4. **reasoner**

---

## 新算子注册流程

### 1. 注册新的类

如果需要注册一个新的类，请在 `core/process` 中编写父类函数（后续补充具体写法规范）。

### 2. 注册已有类的新算子

如果是在已有类中注册新的算子，只需在 `dataflow/process/xxx` 文件夹中找到对应的类并编写代码。  
- 引用父类以传递参数。
- 可参考同文件夹下的其他文件作为示例。

- 注意，必须在算子目录init文件中加入新的算子项，支持lazyloader调用
---

## 测试流程

1. **编写测试代码后必须进行测试！**
2. 在 `demos` 文件夹中对应位置放置测试输入文件。
3. 指定测试输出文件保存到 `demos/demos_result` 文件夹，便于后续 review。

---

## 记录开发过程

如果遇到新的开发内容，请将开发过程记录到以下文档中：

```
/data/scy/DataFlow-Eval-Process/docs/src/getting_started/Dev.md
```

---

## 提交 PR

提交 PR 时需包含以下内容：

1. **运行截图**：展示新增算子的运行结果。
2. **算子名称**：明确说明添加的算子名称。
3. **必要信息**：包括但不限于功能描述、使用方法等。
4. **测试文件和结果文件**：按上文要求保存在 `demos` 和 `demos/demos_result` 文件夹中。