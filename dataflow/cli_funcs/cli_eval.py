# dataflow/cli_funcs/cli_eval.py
"""DataFlow 评估工具"""

import os
import json
import shutil
import importlib.util
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from dataflow import get_logger
from dataflow.serving import LocalModelLLMServing_vllm
from dataflow.operators.reasoning import ReasoningAnswerGenerator
from dataflow.prompts.reasoning.diy import DiyAnswerGeneratorPrompt
from dataflow.utils.storage import FileStorage
import torch
import gc

logger = get_logger()

DEFAULT_ANSWER_PROMPT = """Please answer the following question based on the provided academic literature. Your response should:
1. Provide accurate information from the source material
2. Include relevant scientific reasoning and methodology
3. Reference specific findings, data, or conclusions when applicable
4. Maintain academic rigor and precision in your explanation

Question: {question}

Answer:"""


class EvaluationPipeline:
    """评估管道"""

    def __init__(self, config: Dict[str, Any], cli_args=None):
        self.config = config
        self.cli_args = cli_args
        self.prepared_models = []
        self.generated_files = []

    def run(self) -> bool:
        try:
            # 1. 获取目标模型
            self.target_models = self._get_target_models()
            if not self.target_models:
                logger.error("No TARGET_MODELS found in config")
                return False

            self.prepared_models = self._prepare_models()
            if not self.prepared_models:
                return False

            # 2. 生成答案
            self.generated_files = self._generate_answers()
            if not self.generated_files:
                return False

            # 3. 执行评估
            results = self._run_evaluation()

            # 4. 生成报告
            self._generate_report(results)

            return True

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _get_target_models(self) -> List:
        """获取目标模型列表"""
        target_config = self.config.get("TARGET_MODELS", [])
        
        if not isinstance(target_config, list):
            logger.error(f"TARGET_MODELS must be a list, got {type(target_config)}")
            return []
        
        if not target_config:
            logger.error("TARGET_MODELS is empty")
            return []
            
        return target_config

    def _prepare_models(self) -> List[Dict]:
        """准备模型信息"""
        prepared = []
        default_config = self.config.get("DEFAULT_MODEL_CONFIG", {})
        
        for idx, item in enumerate(self.target_models, 1):
            if isinstance(item, str):
                model_info = {
                    "name": Path(item).name,
                    "path": item,
                    "type": "local",
                    **default_config
                }
            elif isinstance(item, dict):
                if "path" not in item:
                    logger.error(f"Model at index {idx} missing 'path'")
                    continue
                
                model_info = {
                    "name": item.get("name", Path(item["path"]).name),
                    "path": item["path"],
                    "type": "local",
                    **default_config,
                    **{k: v for k, v in item.items() if k not in ["name", "path"]}
                }
            else:
                logger.error(f"Invalid model format at index {idx}")
                continue
            
            prepared.append(model_info)
        
        return prepared

    def _clear_vllm_cache(self):
        """清理 vLLM 缓存"""
        cache_paths = [
            Path.home() / ".cache" / "vllm" / "torch_compile_cache",
            Path.home() / ".cache" / "vllm"
        ]
        
        for cache_path in cache_paths:
            if cache_path.exists():
                try:
                    shutil.rmtree(cache_path)
                except Exception as e:
                    logger.warning(f"Failed to clear cache: {e}")

    def _generate_answers(self) -> List[Dict]:
        """生成模型答案"""
        generated_files = []
        data_config = self.config.get("DATA_CONFIG", {})
        input_file = data_config.get("input_file", "./.cache/data/qa.json")

        if not Path(input_file).exists():
            logger.error(f"Input file not found: {input_file}")
            return []
        
        self._clear_vllm_cache()

        for idx, model_info in enumerate(self.prepared_models, 1):
            llm_serving = None
            answer_generator = None
            storage = None
            
            try:
                logger.info(f"[{idx}/{len(self.prepared_models)}] Processing: {model_info['name']}")
                
                cache_dir = model_info.get('cache_dir', './.cache/eval')
                Path(cache_dir).mkdir(parents=True, exist_ok=True)
                output_file = f"{cache_dir}/answers_{model_info['name']}.json"
                
                # 加载模型
                llm_serving = LocalModelLLMServing_vllm(
                    hf_model_name_or_path=model_info['path'],
                    vllm_tensor_parallel_size=model_info.get('tensor_parallel_size', 2),
                    vllm_max_tokens=model_info.get('max_tokens', 1024),
                    vllm_gpu_memory_utilization=model_info.get('gpu_memory_utilization', 0.8)
                )
                
                # 答案生成器
                custom_prompt = model_info.get('answer_prompt', DEFAULT_ANSWER_PROMPT)
                answer_generator = ReasoningAnswerGenerator(
                    llm_serving=llm_serving,
                    prompt_template=DiyAnswerGeneratorPrompt(custom_prompt)
                )
                
                # 存储
                cache_path = f"{cache_dir}/{model_info['name']}_generation"
                storage = FileStorage(
                    first_entry_file_name=input_file,
                    cache_path=cache_path,
                    file_name_prefix=model_info.get('file_prefix', 'answer_gen'),
                    cache_type=model_info.get('cache_type', 'json')
                )
                
                # 运行生成
                answer_generator.run(
                    storage=storage.step(),
                    input_key=data_config.get("question_key", "input"),
                    output_key=model_info.get('output_key', 'model_generated_answer')
                )
                
                # 保存结果
                gen_file = f"{cache_path}/{model_info.get('file_prefix', 'answer_gen')}_step1.json"
                if Path(gen_file).exists():
                    shutil.copy2(gen_file, output_file)
                    generated_files.append({
                        "model_name": model_info['name'],
                        "model_path": model_info['path'],
                        "file_path": output_file
                    })
                
            except Exception as e:
                logger.error(f"Failed to process {model_info['name']}: {e}")
                continue
                
            finally:
                if answer_generator: del answer_generator
                if storage: del storage
                if llm_serving: del llm_serving
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
        
        return generated_files

    def _run_evaluation(self) -> List[Dict]:
        """运行评估"""
        try:
            judge_serving = self.config["create_judge_serving"]()
        except Exception as e:
            logger.error(f"Failed to create judge: {e}")
            return []

        results = []
        eval_config = self.config.get("EVALUATOR_RUN_CONFIG", {})

        for file_info in self.generated_files:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                result_file = f"./eval_results/{timestamp}_{file_info['model_name']}/result.json"
                Path(result_file).parent.mkdir(parents=True, exist_ok=True)

                storage = self.config["create_storage"](
                    file_info["file_path"],
                    f"./.cache/eval/{file_info['model_name']}"
                )
                evaluator = self.config["create_evaluator"](judge_serving, result_file)

                evaluator.run(
                    storage=storage.step(),
                    input_test_answer_key=eval_config.get("input_test_answer_key", "model_generated_answer"),
                    input_gt_answer_key=eval_config.get("input_gt_answer_key", "output"),
                    input_question_key=eval_config.get("input_question_key", "input")
                )

                if Path(result_file).exists():
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                        if data:
                            data[0]["model_name"] = file_info['model_name']
                            results.append(data[0])

            except Exception as e:
                logger.error(f"Eval failed for {file_info['model_name']}: {e}")
                continue

        return results

    def _generate_report(self, results: List[Dict]):
        """生成报告"""
        if not results:
            logger.warning("No results")
            return

        sorted_results = sorted(results, key=lambda x: x.get("accuracy", 0), reverse=True)

        print("\n" + "=" * 60)
        print("Model Evaluation Results")
        print("=" * 60)
        for i, r in enumerate(sorted_results, 1):
            print(f"{i}. {r['model_name']}")
            print(f"   Accuracy: {r.get('accuracy', 0):.3f}")
            print(f"   Total: {r.get('total_samples', 0)}")
            print(f"   Matched: {r.get('matched_samples', 0)}")
            print()
        print("=" * 60)
        
        # 保存详细报告
        report_file = "./eval_results/report.json"
        Path(report_file).parent.mkdir(parents=True, exist_ok=True)
        with open(report_file, 'w') as f:
            json.dump({"results": sorted_results}, f, indent=2)
        print(f"Detailed report: {report_file}")

class DataFlowEvalCLI:
    """CLI工具"""

    def __init__(self):
        self.current_dir = Path.cwd()

    def _get_template_path(self, eval_type: str) -> Path:
        current_file = Path(__file__)
        dataflow_dir = current_file.parent.parent
        return dataflow_dir / "cli_funcs" / "eval_pipeline" / f"eval_{eval_type}.py"

    def init_eval_files(self):
        """初始化配置文件"""
        files = [("eval_api.py", "api"), ("eval_local.py", "local")]
        
        existing = [f for f, _ in files if (self.current_dir / f).exists()]
        if existing:
            if input(f"{', '.join(existing)} exists. Overwrite? (y/n): ").lower() != 'y':
                return False
        
        for filename, eval_type in files:
            try:
                template = self._get_template_path(eval_type)
                if not template.exists():
                    logger.error(f"Template not found: {template}")
                    continue
                shutil.copy2(template, self.current_dir / filename)
                logger.info(f"Created: {filename}")
            except Exception as e:
                logger.error(f"Failed: {e}")
        logger.info("You must modified the eval_api.py or eval_local.py before you run dataflow eval api/local")
        return True

    def run_eval_file(self, eval_type: str, eval_file: str, cli_args):
        """运行评估"""
        config_path = self.current_dir / eval_file
        
        if not config_path.exists():
            logger.error(f"Config not found: {eval_file}")
            return False

        try:
            spec = importlib.util.spec_from_file_location("config", config_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            config = module.get_evaluator_config()
            return run_evaluation(config, cli_args)

        except Exception as e:
            logger.error(f"Failed: {e}")
            return False


def run_evaluation(config, cli_args=None):
    """运行评估"""
    pipeline = EvaluationPipeline(config, cli_args)
    return pipeline.run()