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

# LoRA合并依赖
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    LORA_SUPPORT = True
except ImportError:
    LORA_SUPPORT = False
    logger.warning("LoRA support not available. Install transformers and peft to enable LoRA adapter merging.")

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

    def run(self) -> bool:
        """执行完整的评估流程"""
        try:
            # 1. 模型检测和准备
            logger.info("Starting model detection...")
            self.detected_models = self._detect_models()
            if not self.detected_models:
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

    def _detect_models(self) -> List[str]:
        """智能检测模型 - 自动找到微调模型对应的base model"""
        target_config = self.config.get("TARGET_MODELS", {})

        if not target_config.get("auto_detect", True):
            models = target_config.get("models", [])
            logger.info(f"Using specified models: {models}")
            return models

        detected = []
        base_models = set()  # 避免重复添加相同的base model

        # 扫描微调模型
        cache_dirs = ["./.cache/saves", "./saves"]
        for cache_dir in cache_dirs:
            cache_path = Path(cache_dir)
            if cache_path.exists():
                logger.info(f"Scanning directory: {cache_path}")
                for model_dir in cache_path.iterdir():
                    if model_dir.is_dir() and self._is_lora_adapter(model_dir):
                        # 检测到LoRA适配器，尝试合并
                        try:
                            merged_model_path = self._merge_lora_adapter(model_dir)
                            detected.append(merged_model_path)
                            logger.info(f"Found LoRA adapter, merged to: {Path(merged_model_path).name}")

                            # 也尝试找到对应的base model
                            base_model_path = self._get_base_model_path(model_dir)
                            if base_model_path and base_model_path not in base_models:
                                if self._validate_base_model(base_model_path):
                                    detected.append(base_model_path)
                                    base_models.add(base_model_path)
                                    logger.info(f"Auto-discovered base model: {base_model_path}")

                        except Exception as e:
                            logger.error(f"Failed to merge LoRA adapter {model_dir.name}: {e}")
                            continue

        if detected:
            logger.info(
                f"Total detected {len(detected)} models (including fine-tuned models and corresponding base models)")
            return detected

        # 如果没找到微调模型，退回到原来的检测逻辑
        logger.info("No fine-tuned models found, trying fallback detection...")
        return self._fallback_detection()

    def _is_lora_adapter(self, model_path: Path) -> bool:
        """检查是否是LoRA适配器"""
        adapter_files = ["adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"]
        has_adapter_config = (model_path / "adapter_config.json").exists()
        has_adapter_model = any((model_path / f).exists() for f in adapter_files[1:])
        return has_adapter_config and has_adapter_model

    def _get_base_model_path(self, adapter_path: Path) -> str:
        """从adapter配置中获取base model路径"""
        adapter_config_file = adapter_path / "adapter_config.json"

        try:
            with open(adapter_config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)

            base_model_path = config.get("base_model_name_or_path", "")
            if base_model_path:
                logger.info(f"Read base model path from {adapter_path.name}: {base_model_path}")
                return base_model_path
            else:
                logger.warning(f"base_model_name_or_path field not found in adapter_config.json")
                return None

        except Exception as e:
            logger.error(f"Failed to read adapter config: {e}")
            return None

    def _validate_base_model(self, model_path: str) -> bool:
        """验证base model是否存在且有效"""
        # 如果是HuggingFace模型ID，假设有效
        if not model_path.startswith(('.', '/')):
            logger.info(f"Detected HuggingFace model ID: {model_path}")
            return True

        # 如果是本地路径，检查文件是否存在
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            logger.warning(f"Local model path does not exist: {model_path}")
            return False

        # 检查是否有config.json
        has_config = (model_path_obj / "config.json").exists()
        if not has_config:
            logger.warning(f"Missing config.json in {model_path}")
            return False

        # 检查模型权重文件（支持多种格式）
        weight_files = ["pytorch_model.bin", "model.safetensors"]
        has_single_weights = any((model_path_obj / f).exists() for f in weight_files)

        # 检查分片模型（pytorch格式）
        has_sharded_pytorch = bool(list(model_path_obj.glob("pytorch_model-*.bin")))

        # 检查分片模型（safetensors格式）
        has_sharded_safetensors = bool(list(model_path_obj.glob("model-*-of-*.safetensors")))

        if has_single_weights or has_sharded_pytorch or has_sharded_safetensors:
            logger.info(f"Found valid model at: {model_path}")
            return True

        logger.warning(f"No valid model weights found in {model_path}")
        return False

    def _fallback_detection(self) -> List[str]:
        """回退检测逻辑 - 原来的检测方法"""
        detected = []
        cache_dirs = ["./.cache/saves", "./saves"]

        for cache_dir in cache_dirs:
            cache_path = Path(cache_dir)
            if cache_path.exists():
                for model_dir in cache_path.iterdir():
                    if model_dir.is_dir() and self._is_valid_model_dir_fallback(model_dir):
                        detected.append(str(model_dir))
                        logger.info(f"Found model: {model_dir.name}")

        return detected

    def _is_valid_model_dir_fallback(self, model_path: Path) -> bool:
        """回退的模型检测方法"""
        # 检查LoRA适配器
        if self._is_lora_adapter(model_path):
            return True

        # 检查完整模型
        has_config = (model_path / "config.json").exists()
        weight_files = ["pytorch_model.bin", "model.safetensors"]
        has_weights = any((model_path / f).exists() for f in weight_files)
        has_sharded = bool(list(model_path.glob("pytorch_model-*.bin")) or
                           list(model_path.glob("model-*-of-*.safetensors")))

        return has_config and (has_weights or has_sharded)

    def _merge_lora_adapter(self, adapter_path: Path) -> str:
        """合并LoRA适配器与基础模型"""
        if not LORA_SUPPORT:
            raise RuntimeError("LoRA merging requires transformers and peft libraries. Please install them.")

        # 读取适配器配置
        adapter_config_file = adapter_path / "adapter_config.json"
        with open(adapter_config_file, 'r') as f:
            adapter_config = json.load(f)

        base_model_path = adapter_config["base_model_name_or_path"]

        # 生成合并后模型的保存路径
        merged_model_name = f"{adapter_path.name}_merged"
        merged_model_path = adapter_path.parent / merged_model_name

        # 如果已经存在合并后的模型，直接返回路径
        if merged_model_path.exists() and (merged_model_path / "config.json").exists():
            logger.info(f"Found existing merged model: {merged_model_path}")
            return str(merged_model_path)

        logger.info(f"Merging LoRA adapter {adapter_path.name} with base model {base_model_path}")

        try:
            # 加载基础模型和分词器
            logger.info("Loading base model...")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype="auto",
                device_map="cpu"  # 先加载到CPU，避免GPU内存不足
            )

            tokenizer = AutoTokenizer.from_pretrained(base_model_path)

            # 加载LoRA适配器
            logger.info("Loading LoRA adapter...")
            model = PeftModel.from_pretrained(base_model, str(adapter_path))

            # 合并适配器到基础模型
            logger.info("Merging adapter with base model...")
            merged_model = model.merge_and_unload()

            # 保存合并后的模型
            logger.info(f"Saving merged model to {merged_model_path}...")
            merged_model_path.mkdir(exist_ok=True)

            merged_model.save_pretrained(str(merged_model_path))
            tokenizer.save_pretrained(str(merged_model_path))

            logger.info(f"Successfully merged LoRA adapter. Merged model saved to: {merged_model_path}")

            # 清理内存
            del base_model, model, merged_model
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return str(merged_model_path)

        except Exception as e:
            logger.error(f"Failed to merge LoRA adapter {adapter_path}: {e}")
            raise

    def _is_valid_model_dir(self, model_path: Path) -> bool:
        """检查是否是有效的模型目录"""
        return self._is_lora_adapter(model_path) or self._is_valid_model_dir_fallback(model_path)

    def _prepare_models(self) -> List[Dict[str, str]]:
        """准备模型信息"""
        prepared = []
        for model_path in self.detected_models:
            model_name = Path(model_path).name
            prepared.append({
                "name": model_name,
                "path": model_path,
                "type": "local"
            })
        return prepared

    def _generate_answers(self) -> List[Dict[str, Any]]:
        """生成模型答案 - 修复版本：真正调用模型推理"""
        generated_files = []
        data_config = self.config.get("DATA_CONFIG", {})
        input_file = data_config.get("input_file", "./.cache/data/qa.json")

        if not Path(input_file).exists():
            logger.error(f"Evaluation data file does not exist: {input_file}")
            return []

        for model_info in self.prepared_models:
            try:
                logger.info(f"Loading model for inference: {model_info['name']}")

                # 确保评估缓存目录存在
                Path("./.cache/eval").mkdir(parents=True, exist_ok=True)
                output_file = f"./.cache/eval/answers_{model_info['name']}.json"

                # 检查是否已存在答案文件，如果是占位符则重新生成
                skip_generation = False
                if Path(output_file).exists():
                    # 检查是否是占位符答案，如果是就重新生成
                    try:
                        with open(output_file, 'r', encoding='utf-8') as f:
                            existing_data = json.load(f)

                        # 检查是否包含占位符格式的答案
                        is_placeholder = False
                        if isinstance(existing_data, list) and existing_data:
                            sample_answer = existing_data[0].get("model_generated_answer", "")
                            if sample_answer.startswith("Generated answer for:"):
                                is_placeholder = True
                                logger.info(f"Found placeholder answers for {model_info['name']}, will regenerate")

                        if not is_placeholder:
                            logger.info(f"Found existing real answers for {model_info['name']}, skipping inference")
                            skip_generation = True
                    except Exception as e:
                        logger.warning(
                            f"Could not check existing answers for {model_info['name']}: {e}, will regenerate")

                if skip_generation:
                    generated_files.append({
                        "model_name": model_info['name'],
                        "model_path": model_info['path'],
                        "file_path": output_file
                    })
                    continue

                # 创建LLM服务
                try:
                    llm_serving = LocalModelLLMServing_vllm(
                        hf_model_name_or_path=model_info['path'],
                        vllm_tensor_parallel_size=2,  # 根据GPU数量调整
                        vllm_max_tokens=1024,
                        vllm_gpu_memory_utilization=0.8
                    )
                    logger.info(f"Model loaded: {model_info['name']}")

                except Exception as e:
                    logger.error(f"Failed to load model {model_info['name']}: {e}")
                    continue

                # 创建答案生成器
                answer_generator = ReasoningAnswerGenerator(
                    llm_serving=llm_serving,
                    prompt_template=DiyAnswerGeneratorPrompt(DEFAULT_ANSWER_PROMPT)
                )

                # 创建存储
                cache_path = f"./.cache/eval/{model_info['name']}_generation"
                storage = FileStorage(
                    first_entry_file_name=input_file,
                    cache_path=cache_path,
                    file_name_prefix="answer_gen",
                    cache_type="json"
                )

                # 生成答案
                logger.info(f"Generating answers for {model_info['name']}...")
                data_config = self.config.get("DATA_CONFIG", {})
                question_key = data_config.get("question_key", "input")

                answer_generator.run(
                    storage=storage.step(),
                    input_key=question_key,
                    output_key="model_generated_answer"
                )

                # 读取生成的结果并保存到最终位置
                try:
                    # storage.read() 返回最后写入的文件路径
                    generated_data_path = f"./.cache/eval/{model_info['name']}_generation/answer_gen_step1.json"
                    if Path(generated_data_path).exists():
                        # 复制到标准输出位置
                        shutil.copy2(generated_data_path, output_file)
                        logger.info(f"Answers generated and saved to: {output_file}")
                    else:
                        logger.error(f"Generated file not found: {generated_data_path}")
                        continue
                except Exception as e:
                    logger.error(f"Failed to process generated answers for {model_info['name']}: {e}")
                    continue

                generated_files.append({
                    "model_name": model_info['name'],
                    "model_path": model_info['path'],
                    "file_path": output_file
                })

            except Exception as e:
                logger.error(f"Failed to generate answers for model {model_info['name']}: {e}")
                continue

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

    def _show_no_models_help(self):
        """显示无模型时的帮助信息"""
        print("No available models detected for evaluation")
        print()
        print("Solutions:")
        print("1. Train models:")
        print("   dataflow text2model init && dataflow text2model train")
        print()
        print("2. Manually specify models - edit config file:")
        print("   TARGET_MODELS = {")
        print("       'auto_detect': False,")
        print("       'models': [")
        print("           'Qwen/Qwen2.5-7B-Instruct',")
        print("           'meta-llama/Llama-3-8B-Instruct'")
        print("       ]")
        print("   }")


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
            logger.info("\nYou can edit these files to customize evaluation configuration")
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

        # 验证 TARGET_MODELS 结构
        target_models = config.get("TARGET_MODELS", {})
        if not isinstance(target_models, dict):
            logger.error("TARGET_MODELS must be a dictionary")
            return False

        if "auto_detect" not in target_models:
            logger.error("TARGET_MODELS missing 'auto_detect' parameter")
            return False

        # 验证 JUDGE_MODEL_CONFIG - 根据评估类型区分
        judge_config = config.get("JUDGE_MODEL_CONFIG", {})
        if not isinstance(judge_config, dict):
            logger.error("JUDGE_MODEL_CONFIG must be a dictionary")
            return False

        # 修复：确保正确处理不同的评估类型
        if eval_type == "api":
            # API 模式验证 - 需要 model_name, api_url, api_key_env
            required_judge_keys = ["model_name", "api_url", "api_key_env"]
            for key in required_judge_keys:
                if key not in judge_config:
                    logger.error(f"JUDGE_MODEL_CONFIG missing required key for API mode: {key}")
                    return False

            # 验证API密钥环境变量
            api_key_env = judge_config.get("api_key_env")
            if api_key_env and api_key_env not in os.environ:
                logger.error(f"❌ API key environment variable not set: {api_key_env}")
                logger.info(f"💡 Please set: export {api_key_env}='your_api_key'")
                return False

        elif eval_type == "local":
            # Local 模式验证 - 需要 model_path（不是 model_name！）
            required_judge_keys = ["model_path"]
            for key in required_judge_keys:
                if key not in judge_config:
                    logger.error(f"JUDGE_MODEL_CONFIG missing required key for Local mode: {key}")
                    logger.info(f"💡 Local mode should use 'model_path', not 'model_name'")
                    return False

            # 验证本地模型路径
            model_path = judge_config.get("model_path")
            if model_path and not model_path.startswith(("Qwen", "meta-llama", "microsoft", "google")):
                # 如果是本地路径，检查是否存在
                model_path_obj = Path(model_path)
                if not model_path_obj.exists():
                    logger.error(f"❌ Local model path does not exist: {model_path}")
                    logger.info(f"💡 Please check your model path in eval_local.py")
                    return False
        else:
            logger.warning(f"Unknown eval_type: {eval_type}, skipping specific validation")

        logger.debug("✅ Configuration validation passed")
        return True

    def run_eval_file(self, eval_type: str, eval_file: str, cli_args):
        """运行评估 - 修复版本，确保正确传递评估类型"""
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
                config["TARGET_MODELS"]["auto_detect"] = False
                config["TARGET_MODELS"]["models"] = model_list
                logger.info(f"Using specified models: {model_list}")

            # 处理 --no-auto 参数
            if hasattr(cli_args, 'no_auto') and cli_args.no_auto:
                config["TARGET_MODELS"]["auto_detect"] = False
                logger.info("Auto-detection disabled by --no-auto parameter")

                # 如果既没有 --models 也没有配置文件中的模型，给出警告
                if not config["TARGET_MODELS"].get("models"):
                    logger.warning("❌ No models specified and auto-detection disabled")
                    logger.info("💡 Solutions:")
                    logger.info("   1. Add models to config file: TARGET_MODELS['models'] = ['model1', 'model2']")
                    logger.info("   2. Use --models parameter: dataflow eval local --models 'model1,model2'")
                    logger.info("   3. Remove --no-auto to enable auto-detection")
                    return False

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

    def list_eval_files(self):
        """列出当前目录下的评估文件"""
        eval_files = list(self.current_dir.glob("eval_*.py"))
        if eval_files:
            logger.info("Found following evaluation config files:")
            for eval_file in eval_files:
                logger.info(f"  - {eval_file.name}")
        else:
            logger.info("No evaluation config files found in current directory")
            logger.info("Please run 'dataflow eval init' to generate config files")


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


def cli_eval():
    """评估命令行入口函数"""
    parser = argparse.ArgumentParser(description="DataFlow Evaluation Tool")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # init 子命令 - 简化，不需要type参数
    init_parser = subparsers.add_parser("init",
                                        help="Initialize evaluation config files (creates eval_api.py and eval_local.py)")

    # api 子命令
    api_parser = subparsers.add_parser("api", help="API model evaluation")
    api_parser.add_argument("--models", help="Comma-separated list of models")
    api_parser.add_argument("eval_file", nargs='?', default="eval_api.py", help="Evaluation config file")

    # local 子命令
    local_parser = subparsers.add_parser("local", help="Local model evaluation")
    local_parser.add_argument("--models", help="Comma-separated list of models")
    local_parser.add_argument("eval_file", nargs='?', default="eval_local.py", help="Evaluation config file")

    # list 子命令
    list_parser = subparsers.add_parser("list", help="List evaluation config files")

    # 新增：清除缓存命令
    clear_parser = subparsers.add_parser("clear-cache", help="Clear evaluation cache files")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    cli = DataFlowEvalCLI()

    try:
        if args.command == "init":
            success = cli.init_eval_files()
            if success:
                logger.info("Config files initialization successful")

        elif args.command in ["api", "local"]:
            success = cli.run_eval_file(args.command, args.eval_file, args)
            if success:
                logger.info("Evaluation completed")
            else:
                logger.error("Evaluation failed")

        elif args.command == "list":
            cli.list_eval_files()

        elif args.command == "clear-cache":
            cli.clear_evaluation_cache()

    except KeyboardInterrupt:
        logger.info("User interrupted operation")
    except Exception as e:
        logger.error(f"Error occurred during execution: {str(e)}")


# 为DataFlowEvalCLI类添加清除缓存方法
def clear_evaluation_cache(self):
    """清除评估缓存文件"""
    cache_paths = [
        "./.cache/eval",
        "./eval_results"
    ]

    cleared_count = 0
    for cache_path in cache_paths:
        cache_dir = Path(cache_path)
        if cache_dir.exists():
            try:
                shutil.rmtree(cache_dir)
                logger.info(f"Cleared cache directory: {cache_path}")
                cleared_count += 1
            except Exception as e:
                logger.error(f"Failed to clear {cache_path}: {e}")

    if cleared_count > 0:
        logger.info(f"Successfully cleared {cleared_count} cache directories")
    else:
        logger.info("No cache directories found to clear")


# 添加方法到类中
DataFlowEvalCLI.clear_evaluation_cache = clear_evaluation_cache

if __name__ == "__main__":
    cli_eval()