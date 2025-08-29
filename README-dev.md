# DataFlow-Preview-开发文档

你可以先创建一个纯净的`python==3.10`的运行环境。

然后克隆本仓库后本地安装：
```shell
pip install -e .
```

安装后可以使用如下指令检验是否正确安装：
```shell
dataflow -v
dataflow env
```

## 测试reasoning Pipeline的方式
目前测试用入口文件在/test/test_reasoning.py中
默认使用/dataflow/example/ReasoningPipeline/pipeline_math_short.json作为样例输入。


向系统export全局的key环境变量。
```shell
export API_KEY=<your key>
```

随后**切换工作路径到`/test`下**，直接执行即可体验一个超短的pipeline
```shell
python test_reasoning.py
```

<!-- 选择性安装:
```shell
pip install -e .[all]
```

安装text组件
```shell
pip install -e .[text]
```

## 基于命令行的调用方式
从pypi查看是否是最新版本
```
dataflow -v 
```

在本地某一路径生成算子运行所需的shell脚本和yaml脚本 (目前todo，可以讨论潜在的指令)
```shell
dataflow init all
dataflow init reasoning
``` -->


