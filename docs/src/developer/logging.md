## Logger

目前logger的初始化在pipeline_step.py中 
```python
import logging
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(filename)-20s- %(module)-20s- %(funcName)-20s- %(lineno)5d - %(name)-10s | %(levelname)8s | Processno %(process)5d - Threadno %(thread)-15d : %(message)s", 
    datefmt="%Y-%m-%d %H:%M:%S"
    )
```
使用方法如下所示，其中debug, info, warning, error代表不同的日志等级，默认情况下DEBUG等级的日志不会显示。
```python
def main():
    
    logging.debug("This is DEBUG message")
    logging.info("This is INFO message")
    logging.warning("This is WARNING message")
    logging.error("This is ERROR message")
    
    return

main()
```
关于等级的分配原则：
1. DEBUG：一些没什么用需要屏蔽的输出 / 不想展示的技术细节，如：
```python
                for x in ['Text', 'image', 'video']:
                    module_path = "dataflow.Eval." + x
                    try:
                        module_lib = importlib.import_module(module_path)
                        clss = getattr(module_lib, name)
                        self._obj_map[name] = clss
                        return clss
                    except AttributeError as e:
                        logging.debug(f"{str(e)}")
                        continue
                    except Exception as e:
                        raise e
```
2. INFO: 让用户得知目前的运行情况，如：
```python
def pipeline_step(yaml_path, step_name, step_type):
    import logging
    import yaml
    logging.info(f"Loading yaml {yaml_path} ......")
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    config = merge_yaml(config)
    logging.info(f"Load yaml success, config: {config}")
    if step_type == "process":
        algorithm = get_processor(step_name, config)
    elif step_type == "generator":
        algorithm = get_generator(step_name, config)
    logging.info("Start running ...")
    algorithm.run()
```
3. WARNING：可能出现问题的错误信息（暂时没有例子）
4. ERROR：运行出现错误，打印错误信息

算子内部的logging可以参考`DataFlow/dataflow/generator/algorithms/TreeSitterParser.py`