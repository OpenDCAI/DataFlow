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

    def __init__(self, config: Dict[str, Any]):
        self.config = config
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
                    **default_config,
                    **item,
                    "name": item.get("name", Path(item["path"]).name),
                    "type": "local"
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
        """生成模型答案 - 每个模型只加载一次"""
        generated_files = []
        bench_config_list = self.config.get("BENCH_CONFIG", [])
        
        if not bench_config_list:
            logger.error("No BENCH_CONFIG found")
            return []
        
        # 外层循环：遍历模型
        for idx, model_info in enumerate(self.prepared_models, 1):
            llm_serving = None
            
            try:
                logger.info(f"[{idx}/{len(self.prepared_models)}] Loading model: {model_info['name']}")
                
                # 清理缓存（每个模型加载前清理一次）
                self._clear_vllm_cache()
                
                # 加载模型
                llm_serving = LocalModelLLMServing_vllm(
                    hf_model_name_or_path=model_info['path'],
                    vllm_tensor_parallel_size=model_info.get('vllm_tensor_parallel_size', 2),
                    vllm_temperature=model_info.get('vllm_temperature', 0.7),
                    vllm_top_p=model_info.get('vllm_top_p', 0.9),
                    vllm_max_tokens=model_info.get('vllm_max_tokens', 1024),
                    vllm_repetition_penalty=model_info.get('vllm_repetition_penalty', 1.0),
                    vllm_seed=model_info.get('vllm_seed', None),
                    vllm_gpu_memory_utilization=model_info.get('vllm_gpu_memory_utilization', 0.8)
                )
                
                # 内层循环：遍历bench（复用模型）
                for bench_idx, bench_config in enumerate(bench_config_list, 1):
                    answer_generator = None
                    storage = None
                    
                    try:
                        bench_name = bench_config.get("name", "default")
                        logger.info(f"  [{bench_idx}/{len(bench_config_list)}] Processing bench: {bench_name}")
                        
                        input_file = bench_config["input_file"]
                        if not Path(input_file).exists():
                            logger.error(f"Input file not found: {input_file}")
                            continue
                        
                        question_key = bench_config.get("question_key", "input")
                        bench_output_dir = bench_config.get("output_dir", "./eval_results")
                        
                        # 设置缓存和输出目录
                        cache_dir = model_info.get('cache_dir', './.cache/eval')
                        Path(cache_dir).mkdir(parents=True, exist_ok=True)
                        Path(bench_output_dir).mkdir(parents=True, exist_ok=True)
                        
                        output_file = f"{bench_output_dir}/{bench_name}_answers_{model_info['name']}.json"
                        
                        # 答案生成器（复用llm_serving）
                        custom_prompt = model_info.get('answer_prompt', DEFAULT_ANSWER_PROMPT)
                        answer_generator = ReasoningAnswerGenerator(
                            llm_serving=llm_serving,
                            prompt_template=DiyAnswerGeneratorPrompt(custom_prompt)
                        )
                        
                        # 存储
                        cache_path = f"{cache_dir}/{bench_name}_{model_info['name']}_generation"
                        storage = FileStorage(
                            first_entry_file_name=input_file,
                            cache_path=cache_path,
                            file_name_prefix=model_info.get('file_prefix', 'answer_gen'),
                            cache_type=model_info.get('cache_type', 'json')
                        )
                        
                        # 运行生成
                        answer_generator.run(
                            storage=storage.step(),
                            input_key=question_key,
                            output_key=model_info.get('output_key', 'model_generated_answer')
                        )
                        
                        # 保存结果
                        file_prefix = model_info.get('file_prefix', 'answer_gen')
                        cache_type = model_info.get('cache_type', 'json')
                        pattern = f"{file_prefix}_step*.{cache_type}"
                        matching_files = sorted(Path(cache_path).glob(pattern))
                        
                        if matching_files:
                            gen_file = matching_files[-1]
                            shutil.copy2(gen_file, output_file)
                            generated_files.append({
                                "model_name": model_info['name'],
                                "model_path": model_info['path'],
                                "file_path": output_file,
                                "bench_name": bench_name
                            })
                            logger.success(f"  ✓ Generated answers for {bench_name}")
                        else:
                            logger.error(f"No generated file found in {cache_path}")
                            continue
                            
                    except Exception as e:
                        logger.error(f"Failed to process bench {bench_name}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                        
                    finally:
                        # 清理bench级别的资源
                        if answer_generator is not None:
                            del answer_generator
                        if storage is not None:
                            del storage
                        gc.collect()
                
            except Exception as e:
                logger.error(f"Failed to load model {model_info['name']}: {e}")
                import traceback
                traceback.print_exc()
                continue
                
            finally:
                # 清理模型级别的资源
                if llm_serving is not None:
                    logger.info(f"Unloading model: {model_info['name']}")
                    del llm_serving
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
        
        return generated_files

    def _run_evaluation(self) -> List[Dict]:
        """运行评估"""
        try:
            logger.info("Loading judge model...")
            judge_serving = self.config["create_judge_serving"]()
            logger.info("✓ Judge model loaded")
        except Exception as e:
            logger.error(f"Failed to create judge: {e}")
            return []

        results = []
        eval_config = self.config.get("EVALUATOR_RUN_CONFIG", {})
        
        total_evals = len(self.generated_files)
        
        for eval_idx, file_info in enumerate(self.generated_files, 1):
            try:
                bench_name = file_info.get('bench_name', 'unknown')
                model_name = file_info['model_name']
                
                logger.info(f"\n[Eval {eval_idx}/{total_evals}] {model_name} × {bench_name}")
                
                # 找到对应的bench配置
                bench_config = None
                for bc in self.config.get("BENCH_CONFIG", []):
                    if bc.get("name") == bench_name:
                        bench_config = bc
                        break
                
                if not bench_config:
                    logger.warning(f"  ⚠️  No bench config found")
                    continue
                
                eval_output_dir = bench_config.get("eval_output_dir", "./eval_results")
                
                # 执行评估
                logger.info(f"  📊 Evaluating...")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                result_file = f"{eval_output_dir}/{timestamp}_{model_name}/result.json"
                Path(result_file).parent.mkdir(parents=True, exist_ok=True)

                storage = self.config["create_storage"](
                    file_info["file_path"],
                    f"./.cache/eval/{model_name}",
                    bench_name
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
                            result_data = data[0].copy()
                            result_data["model_name"] = model_name
                            result_data["bench_name"] = bench_name
                            results.append(result_data)
                            logger.info(f"  ✓ Accuracy: {result_data.get('accuracy', 0):.3f}")

            except Exception as e:
                logger.error(f"  ✗ Evaluation failed: {e}")
                import traceback
                traceback.print_exc()
                continue

        logger.info(f"\n✓ Evaluation complete: {len(results)} results")
        return results

    def _generate_report(self, results: List[Dict]):
        """生成报告 - 支持批量bench独立输出"""
        if not results:
            logger.warning("No results")
            return

        # 打印报告
        print("\n" + "="*80)
        print("EVALUATION RESULTS - ALL BENCHES & MODELS")
        print("="*80)

        # 按准确率排序
        sorted_results = sorted(results, key=lambda x: x.get("accuracy", 0), reverse=True)
        
        for i, r in enumerate(sorted_results, 1):
            print(f"{i}. [{r.get('bench_name', 'unknown')}] {r['model_name']}")
            print(f"   Accuracy: {r.get('accuracy', 0):.3f}")
            print(f"   Total: {r.get('total_samples', 0)}")
            print(f"   Matched: {r.get('matched_samples', 0)}")
            print()
        
        # 按bench分组保存结果
        bench_config_list = self.config.get("BENCH_CONFIG", [])
        
        # 为每个bench单独保存结果
        bench_groups = {}
        for result in sorted_results:
            bench_name = result.get('bench_name', 'unknown')
            if bench_name not in bench_groups:
                bench_groups[bench_name] = []
            bench_groups[bench_name].append(result)
        
        # 保存每个bench的结果到各自的output_dir
        for bench_name, bench_results in bench_groups.items():
            # 找到对应bench的配置
            bench_output_dir = "./eval_results"  # 默认值
            for bench_config in bench_config_list:
                if bench_config.get("name") == bench_name:
                    bench_output_dir = bench_config.get("output_dir", "./eval_results")
                    break
            
            # 保存该bench的结果
            report_file = f"{bench_output_dir}/results.json"
            Path(report_file).parent.mkdir(parents=True, exist_ok=True)
            
            report_data = {
                "bench_name": bench_name,
                "timestamp": datetime.now().isoformat(),
                "total_evaluations": len(bench_results),
                "results": bench_results
            }
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            
            print(f"Bench '{bench_name}' results saved to: {report_file}")
        
        # 另外保存一个汇总文件（包含所有bench）
        all_results_file = "./eval_results/all_results.json"
        Path(all_results_file).parent.mkdir(parents=True, exist_ok=True)
        
        all_report_data = {
            "timestamp": datetime.now().isoformat(),
            "total_evaluations": len(sorted_results),
            "total_benches": len(bench_groups),
            "results": sorted_results
        }
        
        with open(all_results_file, 'w', encoding='utf-8') as f:
            json.dump(all_report_data, f, ensure_ascii=False, indent=2)
        
        print("="*80)
        print(f"All results summary saved to: {all_results_file}")
        print("="*80)


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

    def run_eval_file(self, eval_file: str):
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
            return run_evaluation(config)

        except Exception as e:
            logger.error(f"Failed: {e}")
            return False


def run_evaluation(config):
    """运行评估"""
    pipeline = EvaluationPipeline(config)
    return pipeline.run()