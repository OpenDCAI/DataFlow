# dataflow/cli_funcs/cli_eval.py
"""DataFlow 评估工具 - 修复版本"""

import os
import json
import shutil
import subprocess
import argparse
import importlib.util
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from dataflow import get_logger
from dataflow.serving import LocalModelLLMServing_vllm
from dataflow.operators.reasoning import ReasoningAnswerGenerator
from dataflow.prompts.reasoning.diy import DiyAnswerGeneratorPrompt
from dataflow.utils.storage import FileStorage
import torch
import gc

logger = get_logger()

# 默认的答案生成提示词模板
DEFAULT_ANSWER_PROMPT = """Please answer the following question based on the provided academic literature. Your response should:
1. Provide accurate information from the source material
2. Include relevant scientific reasoning and methodology
3. Reference specific findings, data, or conclusions when applicable
4. Maintain academic rigor and precision in your explanation

Question: {question}

Answer:"""


# =============================================================================
# 评估管道核心逻辑
# =============================================================================

class EvaluationPipeline:
    """评估管道：从模型检测到评估报告的完整流程"""

    def __init__(self, config: Dict[str, Any], cli_args=None):
        self.config = config
        self.cli_args = cli_args or argparse.Namespace()
        self.detected_models = []
        self.prepared_models = []
        self.generated_files = []

    # 运行步骤
    def run(self) -> bool:
        try:
            # 1. 获取目标模型
            logger.info("Loading target models...")
            self.target_models = self._get_target_models()  # 新方法
            if not self.target_models:
                self._show_no_models_help()
                return False

            self.prepared_models = self._prepare_models()

            # 2. 生成答案
            logger.info("Starting answer generation...")
            self.generated_files = self._generate_answers()

            # 3. 执行评估
            logger.info("Starting evaluation...")
            results = self._run_evaluation()

            # 4. 生成报告
            self._generate_report(results)

            return True

        except Exception as e:
            logger.error(f"Evaluation pipeline failed: {e}")
            return False

    def _get_target_models(self) -> List[str]:
        """从配置文件获取目标模型列表"""
        target_config = self.config.get("TARGET_MODELS", {})
        
        # 支持格式：列表格式 这里可以添加其他格式
        if isinstance(target_config, list):
            models = target_config
        else:
            logger.error(f"Invalid TARGET_MODELS format: {type(target_config)}")
            return []
        
        if not models:
            logger.error("No models specified in TARGET_MODELS configuration")
            return []
            
        logger.info(f"Target models from config: {models}")     
        return models


    def _show_no_models_help(self):
        """显示无模型时的帮助信息"""
        logger.error("No valid models found!")
        print()
        print("=" * 60)
        print("No models available for evaluation")
        print("=" * 60)
        print()
        print("Solutions:")
        print()
        print("1. Edit your evaluation config file:")
        print("   TARGET_MODELS = [")
        print("       'Qwen/Qwen2.5-7B-Instruct',")
        print("       'meta-llama/Llama-3-8B-Instruct',")
        print("       '/path/to/your/local/model'")
        print("   ]")
        print()
        print("2. Or use dictionary format:")
        print("   TARGET_MODELS = {")
        print("       'models': [")
        print("           'Qwen/Qwen2.5-7B-Instruct',")
        print("           '/path/to/local/model'")
        print("       ]")
        print("   }")
        print()
        print("3. Specify models via command line:")
        print("   dataflow eval local --models 'Qwen/Qwen2.5-7B,/path/to/model'")
        print()
        print("=" * 60)


    def _prepare_models(self) -> List[Dict[str, str]]:
        """准备模型信息"""
        prepared = []
        for model_path in self.target_models:
            model_name = Path(model_path).name
            prepared.append({
                "name": model_name,
                "path": model_path,
                "type": "local"
            })
        return prepared
    def _clear_vllm_cache(self):
        """清理 vLLM 编译缓存"""
        cache_paths = [
            Path.home() / ".cache" / "vllm" / "torch_compile_cache",
            Path.home() / ".cache" / "vllm"
        ]
        
        for cache_path in cache_paths:
            if cache_path.exists():
                try:
                    shutil.rmtree(cache_path)
                    logger.info(f"Cleared vLLM cache: {cache_path}")
                except Exception as e:
                    logger.warning(f"Failed to clear cache {cache_path}: {e}")
    def _generate_answers(self) -> List[Dict[str, Any]]:
        """生成模型答案"""
        import time
        
        generated_files = []
        data_config = self.config.get("DATA_CONFIG", {})
        input_file = data_config.get("input_file", "./.cache/data/qa.json")

        if not Path(input_file).exists():
            logger.error(f"Evaluation data file does not exist: {input_file}")
            return []
        
        self._clear_vllm_cache()  # 清理磁盘缓存

        for idx, model_info in enumerate(self.prepared_models, 1):
            llm_serving = None
            answer_generator = None
            storage = None
            
            try:
                logger.info(f"[{idx}/{len(self.prepared_models)}] Loading: {model_info['name']}")
                
                # ========== 加载模型 ==========
                Path("./.cache/eval").mkdir(parents=True, exist_ok=True)
                output_file = f"./.cache/eval/answers_{model_info['name']}.json"
                
                llm_serving = LocalModelLLMServing_vllm(
                    hf_model_name_or_path=model_info['path'],
                    vllm_tensor_parallel_size=2,
                    vllm_max_tokens=1024,
                    vllm_gpu_memory_utilization=0.8
                )
                logger.info(f"Model loaded: {model_info['name']}")
                
                # ========== 创建答案生成器 ==========
                answer_generator = ReasoningAnswerGenerator(
                    llm_serving=llm_serving,
                    prompt_template=DiyAnswerGeneratorPrompt(DEFAULT_ANSWER_PROMPT)
                )
                
                # ========== 创建存储 ==========
                cache_path = f"./.cache/eval/{model_info['name']}_generation"
                storage = FileStorage(
                    first_entry_file_name=input_file,
                    cache_path=cache_path,
                    file_name_prefix="answer_gen",
                    cache_type="json"
                )
                
                # ========== 生成答案 ==========
                logger.info(f"Generating answers for {model_info['name']}...")
                question_key = data_config.get("question_key", "input")
                
                answer_generator.run(
                    storage=storage.step(),
                    input_key=question_key,
                    output_key="model_generated_answer"
                )
                
                # ========== 保存结果 ==========
                generated_data_path = f"{cache_path}/answer_gen_step1.json"
                if Path(generated_data_path).exists():
                    shutil.copy2(generated_data_path, output_file)
                    logger.info(f"Answers saved to: {output_file}")
                    
                    generated_files.append({
                        "model_name": model_info['name'],
                        "model_path": model_info['path'],
                        "file_path": output_file
                    })
                else:
                    logger.error(f"Generated file not found: {generated_data_path}")
                    continue
                
            except Exception as e:
                logger.error(f"Failed: {e}")
                import traceback
                traceback.print_exc()  # 打印详细错误堆栈
                continue
                
            finally:
                # ========== 清理GPU显存 ==========
                logger.info(f"Releasing GPU memory for {model_info['name']}...")
                
                if answer_generator is not None:
                    del answer_generator
                if storage is not None:
                    del storage
                if llm_serving is not None:
                    del llm_serving
                
                gc.collect()
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                logger.info(f"✓ Memory released\n")
        
        return generated_files

    def _create_eval_result_path(self, model_name: str) -> str:
        """创建基于时间的评估结果路径"""
        timestamp = datetime.now()
        date_str = timestamp.strftime("%Y-%m-%d")
        time_str = timestamp.strftime("%H-%M-%S")

        # 清理模型名称
        clean_model_name = "".join(c for c in model_name if c.isalnum() or c in ('-', '_'))

        # 创建目录结构
        eval_dir = Path("./eval_results") / date_str / f"{time_str}_{clean_model_name}"
        eval_dir.mkdir(parents=True, exist_ok=True)

        return str(eval_dir / "eval_result.json")

    def _save_model_info(self, model_name: str, eval_dir: Path, model_path: str = ""):
        """保存模型信息和配置"""
        model_info = {
            "model_name": model_name,
            "model_path": model_path,
            "evaluation_time": datetime.now().isoformat(),
            "eval_config": self.config.get("EVAL_CONFIG", {}),
            "judge_model": self.config.get("JUDGE_MODEL_CONFIG", {}).get("model_name", "unknown")
        }

        with open(eval_dir / "model_info.json", 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)

    def _run_evaluation(self) -> List[Dict[str, Any]]:
        """运行评估"""
        data_config = self.config.get("DATA_CONFIG", {})

        # 创建judge serving
        try:
            judge_serving = self.config["create_judge_serving"]()
        except Exception as e:
            logger.error(f"Failed to create evaluation service: {e}")
            return []

        results = []

        for file_info in self.generated_files:
            logger.info(f"Evaluating model: {file_info['model_name']}")

            try:
                # 创建基于时间的结果路径
                result_file = self._create_eval_result_path(file_info['model_name'])
                eval_dir = Path(result_file).parent

                # 保存模型信息
                self._save_model_info(file_info['model_name'], eval_dir, file_info.get('model_path', ''))

                # 创建存储（使用.cache/eval目录）
                cache_name = "".join(c for c in file_info['model_name'] if c.isalnum() or c in ('-', '_'))
                storage = self.config["create_storage"](
                    file_info["file_path"],
                    f"./.cache/eval/{cache_name}"
                )

                # 创建评估器
                evaluator = self.config["create_evaluator"](judge_serving, result_file)

                # 运行评估
                evaluator_config = self.config.get("EVALUATOR_RUN_CONFIG", {})
                evaluator.run(
                    storage=storage.step(),
                    input_test_answer_key=evaluator_config.get("input_test_answer_key", "model_generated_answer"),
                    input_gt_answer_key=evaluator_config.get("input_gt_answer_key", "output"),
                    input_question_key=evaluator_config.get("input_question_key", "input")
                )

                # 读取结果
                if Path(result_file).exists():
                    with open(result_file, 'r', encoding='utf-8') as f:
                        result_data = json.load(f)
                        if isinstance(result_data, list) and result_data:
                            result_data[0]["model_name"] = file_info['model_name']
                            result_data[0]["result_file"] = result_file
                            results.append(result_data[0])

            except Exception as e:
                logger.error(f"Failed to evaluate model {file_info['model_name']}: {e}")
                continue

        return results

    def _generate_report(self, results: List[Dict[str, Any]]):
        """生成对比报告"""
        if not results:
            logger.warning("No valid evaluation results")
            return

        print("\n" + "=" * 60)
        print("Model Evaluation Results Comparison")
        print("=" * 60)

        # 按准确率排序
        sorted_results = sorted(results, key=lambda x: x.get("accuracy", 0), reverse=True)

        for i, result in enumerate(sorted_results, 1):
            print(f"{i}. {result.get('model_name', 'Unknown')}")
            print(f"   Accuracy: {result.get('accuracy', 0):.3f}")
            print(f"   Total samples: {result.get('total_samples', 0)}")
            print(f"   Matched samples: {result.get('matched_samples', 0)}")
            print(f"   Result file: {result.get('result_file', 'N/A')}")
            print()

        # 保存详细报告
        report = {
            "evaluation_summary": {
                "total_models": len(results),
                "evaluation_time": datetime.now().isoformat(),
                "results": sorted_results
            }
        }

        report_file = "./eval_results/model_comparison_report.json"
        Path(report_file).parent.mkdir(parents=True, exist_ok=True)
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"Detailed report saved: {report_file}")
        print("=" * 60)

# =============================================================================
# CLI工具类
# =============================================================================

class DataFlowEvalCLI:
    """DataFlow 评估命令行工具"""

    def __init__(self):
        self.current_dir = Path.cwd()

    def _get_template_path(self, eval_type: str) -> Path:
        """获取模板文件路径"""
        # 直接硬编码路径
        current_file = Path(__file__)  # cli_eval.py的路径
        dataflow_dir = current_file.parent.parent  # dataflow目录
        template_path = dataflow_dir / "cli_funcs" / "eval_pipeline" / f"eval_{eval_type}.py"
        return template_path

    def init_eval_files(self):
        """生成评估配置文件 - 一次性复制两个模板"""
        files_to_create = [
            ("eval_api.py", "api"),
            ("eval_local.py", "local")
        ]

        created_files = []
        existing_files = []

        # 检查已存在的文件
        for filename, eval_type in files_to_create:
            output_path = self.current_dir / filename
            if output_path.exists():
                existing_files.append(filename)

        # 如果有文件已存在，询问是否覆盖
        if existing_files:
            logger.warning(f"Following files already exist: {', '.join(existing_files)}")
            user_input = input("Overwrite? (y/n): ").strip().lower()
            if user_input != 'y':
                logger.info("Operation cancelled")
                return False

        # 复制文件
        for filename, eval_type in files_to_create:
            try:
                output_path = self.current_dir / filename

                # 获取内置模板文件路径
                template_path = self._get_template_path(eval_type)

                if not template_path.exists():
                    logger.error(f"Built-in template file does not exist: {template_path}")
                    continue

                # 复制文件
                shutil.copy2(template_path, output_path)
                created_files.append(filename)
                logger.info(f"Created: {filename}")

            except Exception as e:
                logger.error(f"Failed to create {filename}: {e}")
                continue

        if created_files:
            logger.info(f"\nEvaluation config files initialization complete!")
            logger.info("Created files:")
            for filename in created_files:
                logger.info(f"  - {filename}")
            logger.info("\nUsage:")
            logger.info("  dataflow eval api    # API model evaluation")
            logger.info("  dataflow eval local  # Local model evaluation")
            logger.info("\nYou must edit these files to customize evaluation configuration")
            return True
        else:
            logger.error("No config files created successfully")
            return False

    def _validate_config(self, config: Dict[str, Any], eval_type: str = None) -> bool:
        """验证配置文件的必要参数 - 根据评估类型区分验证"""
        required_keys = [
            "TARGET_MODELS",
            "JUDGE_MODEL_CONFIG",
            "DATA_CONFIG",
            "create_judge_serving",
            "create_evaluator",
            "create_storage"
        ]

        for key in required_keys:
            if key not in config:
                logger.error(f"Missing required config key: {key}")
                return False

        
        # 验证 TARGET_MODELS 结构 - 支持两种格式
        target_models = config.get("TARGET_MODELS", {})
        if isinstance(target_models, list):
            # 新格式：直接列表
            if not target_models:
                logger.warning("TARGET_MODELS list is empty")
        elif isinstance(target_models, dict):
            # 原格式：字典包含models
            logger.debug("TARGET_MODELS is dict format")
            if "models" not in target_models:
                logger.error("TARGET_MODELS dict missing 'models' key")
                return False
        else:
            logger.error(f"TARGET_MODELS must be a list or dictionary, got {type(target_models)}")
            return False

        # 验证 JUDGE_MODEL_CONFIG - 根据评估类型区分
        judge_config = config.get("JUDGE_MODEL_CONFIG", {})
        if not isinstance(judge_config, dict):
            logger.error("JUDGE_MODEL_CONFIG must be a dictionary")
            return False
        logger.debug("✅ Configuration validation passed")
        return True

    def run_eval_file(self, eval_type: str, eval_file: str, cli_args):
        """运行评估"""
        eval_path = self.current_dir / eval_file

        if not eval_path.exists():
            logger.error(f"Evaluation config file does not exist: {eval_path}")
            logger.info(f"Please run first: dataflow eval init")
            return False

        try:
            # 动态导入用户的配置文件
            config = self._import_user_config(eval_path)
            if not config:
                logger.error("❌ Configuration file parameters are incorrect")
                logger.info("💡 Please check your config file contains get_evaluator_config() function")
                return False

            # 验证配置文件必要参数 - 修复：确保传入正确的评估类型
            logger.debug(f"Validating config for eval_type: {eval_type}")
            if not self._validate_config(config, eval_type):
                logger.error("❌ Configuration file parameters are incorrect")
                logger.info("💡 Please run 'dataflow eval init' to regenerate config files")
                return False

            # 应用CLI参数覆盖
            if hasattr(cli_args, 'models') and cli_args.models:
                model_list = [m.strip() for m in cli_args.models.split(',')]
                config["TARGET_MODELS"]["models"] = model_list
                logger.info(f"Using CLI specified models: {model_list}")

            # 运行评估
            success = run_evaluation(config, cli_args)
            return success

        except Exception as e:
            logger.error(f"❌ Configuration file parameters are incorrect: {e}")
            logger.info("💡 Troubleshooting steps:")
            logger.info(f"   1. Check config file syntax: python -m py_compile {eval_file}")
            logger.info("   2. Regenerate config: dataflow eval init")
            if eval_type == "api":
                logger.info("   3. Verify API keys: echo $DF_API_KEY")
            else:
                logger.info("   3. Check local model paths in config")
            return False

    def _import_user_config(self, config_path: Path):
        """动态导入用户配置文件"""
        try:
            # 创建模块规范
            spec = importlib.util.spec_from_file_location("user_eval_config", config_path)
            if spec is None:
                logger.error(f"Cannot create module spec: {config_path}")
                return None

            # 创建模块
            user_config_module = importlib.util.module_from_spec(spec)

            # 执行模块
            spec.loader.exec_module(user_config_module)

            # 获取配置
            if hasattr(user_config_module, 'get_evaluator_config'):
                config = user_config_module.get_evaluator_config()
                logger.info(f"Successfully loaded config: {config_path}")
                return config
            else:
                logger.error(f"Config file missing get_evaluator_config() function: {config_path}")
                return None

        except Exception as e:
            logger.error(f"Failed to import config file: {e}")
            logger.error(f"Please check config file syntax: {config_path}")
            return None
# =============================================================================
# 简化的执行接口
# =============================================================================

def run_evaluation(config, cli_args=None):
    """运行评估"""
    logger.info("DataFlow model evaluation started")

    try:
        pipeline = EvaluationPipeline(config, cli_args)
        success = pipeline.run()

        if success:
            logger.info("Model evaluation completed")
        else:
            logger.error("Model evaluation failed")

        return success

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise