# dataflow/cli_funcs/cli_eval.py
"""DataFlow 评估工具 - 重构版本"""

import os
import json
import shutil
import subprocess
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import importlib.util

from dataflow import get_logger
from dataflow.serving import LocalModelLLMServing_vllm

logger = get_logger()

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
            logger.info("开始模型检测...")
            self.detected_models = self._detect_models()
            if not self.detected_models:
                self._show_no_models_help()
                return False
                
            self.prepared_models = self._prepare_models()
            
            # 2. 生成答案
            logger.info("开始生成答案...")
            self.generated_files = self._generate_answers()
            
            # 3. 执行评估
            logger.info("开始执行评估...")
            results = self._run_evaluation()
            
            # 4. 生成报告
            logger.info("生成评估报告...")
            self._generate_report(results)
            
            logger.info("评估流程完成")
            return True
            
        except Exception as e:
            logger.error(f"评估失败：{str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _detect_models(self) -> List[Dict[str, Any]]:
        """检测可用的模型"""
        models = []
        
        # Check command line arguments
        if hasattr(self.cli_args, 'models') and self.cli_args.models:
            logger.info("Using models specified via command line...")
            model_list = self.cli_args.models.split(',')
            for model_spec in model_list:
                model_spec = model_spec.strip()
                models.append({
                    "name": Path(model_spec).name,
                    "path": model_spec,
                    "type": "manual",
                    "needs_merge": False
                })
        elif self.config.get("TARGET_MODELS", {}).get("auto_detect", True):
            logger.info("Auto-detection mode...")
            models = self._auto_detect_models()
        else:
            # Use manually specified models from config file
            manual_models = self.config.get("TARGET_MODELS", {}).get("models", [])
            for model_spec in manual_models:
                models.append({
                    "name": Path(model_spec).name,
                    "path": model_spec,
                    "type": "manual",
                    "needs_merge": False
                })
        
        if models:
            logger.info(f"Detected {len(models)} models:")
            for i, model in enumerate(models, 1):
                logger.info(f"  {i}. {model['name']}")
        
        return models
    
    def _auto_detect_models(self) -> List[Dict[str, Any]]:
        """自动检测模型 - 优先使用已merge的模型，避免重复merge"""
        all_models = []
        lora_models = []
        merged_models = []
        
        # 检测所有模型
        saves_dir = Path("./.cache/saves")
        if saves_dir.exists():
            for model_dir in saves_dir.iterdir():
                if model_dir.is_dir():
                    model_info = self._analyze_model_dir(model_dir)
                    if model_info:
                        all_models.append(model_info)
                        if model_info['type'] == 'lora_adapter':
                            lora_models.append(model_info)
                        elif model_info['name'].endswith('_merged'):
                            merged_models.append(model_info)
        
        # 优选策略：优先使用已merged的模型，避免重复merge
        final_models = []
        
        # 找到一个base model
        base_model = None
        if lora_models:
            base_model = self._find_base_model_for_finetuned(lora_models[0])
        elif merged_models:
            # 从merged模型推断base model
            base_model = self._find_base_model_for_finetuned(merged_models[0])
        
        if base_model:
            final_models.append(base_model)
        
        # 优先选择已merged的模型
        if merged_models:
            # 按时间排序，选择最新的一个merged模型进行对比
            merged_models.sort(key=lambda x: x['name'])
            final_models.append({
                **merged_models[-1],  # 选择最新的
                'needs_merge': False  # 确保不再merge
            })
            
            if base_model:
                logger.info(f"Detected comparison pair (using existing merged model):")
                logger.info(f"  Base model: {base_model['name']}")
                logger.info(f"  Fine-tuned (merged): {merged_models[-1]['name']}")
        elif lora_models:
            # If no merged models exist, select one LoRA for merging
            lora_models.sort(key=lambda x: x['name'])
            final_models.append(lora_models[-1])  # Select the latest LoRA
            
            if base_model:
                logger.info(f"Detected comparison pair (requires merging):")
                logger.info(f"  Base model: {base_model['name']}")
                logger.info(f"  Fine-tuned: {lora_models[-1]['name']}")
        
        return final_models
    
    def _analyze_model_dir(self, model_dir: Path) -> Optional[Dict[str, Any]]:
        """分析模型目录，判断模型类型"""
        if not model_dir.exists():
            return None
            
        # 检查LoRA适配器文件
        adapter_files = ["adapter_config.json", "adapter_model.bin", "adapter_model.safetensors"]
        has_adapter = any((model_dir / f).exists() for f in adapter_files)
        
        # 检查基础模型文件
        model_files = ["config.json", "pytorch_model.bin", "model.safetensors"]
        has_model = any((model_dir / f).exists() for f in model_files)
        
        if has_adapter and not model_dir.name.endswith('_merged'):
            # 原始LoRA适配器
            return {
                "name": model_dir.name,
                "path": str(model_dir),
                "type": "lora_adapter",
                "needs_merge": True
            }
        elif has_model:
            # 完整模型（可能是base model或merged model）
            if model_dir.name.endswith('_merged'):
                return {
                    "name": model_dir.name,
                    "path": str(model_dir),
                    "type": "merged_model",
                    "needs_merge": False
                }
            else:
                return {
                    "name": model_dir.name,
                    "path": str(model_dir),
                    "type": "full_model",
                    "needs_merge": False
                }
        
        return None
    
    def _find_base_model_for_finetuned(self, fine_tuned_model: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """为微调模型寻找对应的base model"""
        if fine_tuned_model.get("type") == "lora_adapter":
            adapter_config_file = Path(fine_tuned_model["path"]) / "adapter_config.json"
            if adapter_config_file.exists():
                try:
                    with open(adapter_config_file, 'r') as f:
                        adapter_config = json.load(f)
                    base_model_path = adapter_config.get("base_model_name_or_path")
                    
                    if base_model_path:
                        return {
                            "name": Path(base_model_path).name,
                            "path": base_model_path,
                            "type": "base_model",
                            "needs_merge": False
                        }
                except Exception as e:
                    logger.warning(f"读取adapter配置失败: {e}")
        
        return None
    
    def _prepare_models(self) -> List[Dict[str, Any]]:
        """准备模型：处理merge等预处理步骤"""
        prepared = []
        
        for model in self.detected_models:
            try:
                if model.get("needs_merge"):
                    logger.info(f"Merging model: {model['name']}")
                    merged_path = self._merge_lora_model(model["path"])
                    model["path"] = merged_path
                    model["name"] = Path(merged_path).name
                    model["needs_merge"] = False
                    logger.info(f"Merge completed: {merged_path}")
                
                prepared.append(model)
                
            except Exception as e:
                logger.error(f"Model {model['name']} preparation failed, skipping: {e}")
                continue
        
        return prepared
    
    def _merge_lora_model(self, adapter_path: str, force_remerge: bool = True) -> str:
        """合并LoRA适配器 - 支持强制重新merge"""
        try:
            adapter_dir = Path(adapter_path)
            merged_dir = adapter_dir.parent / f"{adapter_dir.name}_merged"
            
            # 如果强制重新merge或目录不存在，则删除已有目录重新merge
            if force_remerge or not (merged_dir.exists() and (merged_dir / "config.json").exists()):
                if merged_dir.exists():
                    logger.info(f"删除已存在的merged目录进行重新merge: {merged_dir}")
                    shutil.rmtree(merged_dir)
                else:
                    logger.info(f"merged目录不存在，开始新的merge: {merged_dir}")
            else:
                logger.info(f"使用已存在的merged模型: {merged_dir}")
                return str(merged_dir)
            
            # 读取adapter配置
            adapter_config_file = adapter_dir / "adapter_config.json"
            if not adapter_config_file.exists():
                raise Exception(f"找不到adapter_config.json: {adapter_config_file}")
                
            with open(adapter_config_file, 'r') as f:
                adapter_config = json.load(f)
            base_model_path = adapter_config.get("base_model_name_or_path")
            
            if not base_model_path:
                raise Exception("adapter_config.json中缺少base_model_name_or_path")
            
            logger.info(f"开始merge LoRA适配器: {adapter_dir.name}")
            
            # 使用transformers进行merge
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            import torch
            
            # 加载base model
            logger.info(f"加载base model: {base_model_path}")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # 加载LoRA适配器
            logger.info(f"加载LoRA适配器: {adapter_dir}")
            model = PeftModel.from_pretrained(
                base_model,
                str(adapter_dir),
                torch_dtype=torch.bfloat16,
                is_trainable=False
            )
            
            # 执行merge
            logger.info("执行merge_and_unload...")
            model = model.merge_and_unload(progressbar=True)
            
            # 保存merged模型
            logger.info(f"保存merged模型到: {merged_dir}")
            merged_dir.mkdir(parents=True, exist_ok=True)
            
            model.save_pretrained(
                str(merged_dir),
                torch_dtype=torch.bfloat16,
                safe_serialization=True,
                max_shard_size="2GB"
            )
            
            # 保存tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_path,
                trust_remote_code=True
            )
            tokenizer.save_pretrained(str(merged_dir))
            
            # 验证merge结果
            if not (merged_dir / "config.json").exists():
                raise Exception("Merge完成但config.json不存在")
            
            logger.info(f"LoRA merge成功完成: {merged_dir}")
            
            # 清理内存
            del model, base_model, tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return str(merged_dir)
                
        except Exception as e:
            logger.error(f"LoRA merge失败: {e}")
            # 清理可能的失败残留
            if merged_dir.exists():
                shutil.rmtree(merged_dir, ignore_errors=True)
            raise e
        
    def _generate_answers(self) -> List[Dict[str, str]]:
        """为所有模型生成答案"""
        data_config = self.config.get("DATA_CONFIG", {})
        input_file = data_config.get("input_file", "./.cache/data/qa.json")
        
        if not Path(input_file).exists():
            raise FileNotFoundError(f"数据文件不存在：{input_file}")
        
        # 读取问题数据
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        question_key = data_config.get("question_key", "input")
        questions = [item[question_key] for item in data]
        
        generated_files = []
        batch_size = self.config.get("EVAL_CONFIG", {}).get("batch_size", 8)
        max_tokens = self.config.get("EVAL_CONFIG", {}).get("max_tokens", 512)
        
        for model in self.prepared_models:
            logger.info(f"Generating answers for model: {model['name']}")
            
            try:
                # Get VLLM configuration
                eval_config = self.config.get("EVAL_CONFIG", {})
                
                # Load model with supported GPU memory management configuration only
                model_serving = LocalModelLLMServing_vllm(
                    hf_model_name_or_path=model["path"],
                    vllm_tensor_parallel_size=1,
                    vllm_max_tokens=max_tokens,
                    # Only use supported GPU memory management parameters
                    vllm_gpu_memory_utilization=eval_config.get("vllm_gpu_memory_utilization", 0.75),
                    vllm_max_model_len=eval_config.get("vllm_max_model_len", 16384),
                )
                
                # Generate answers in batches
                answers = []
                for i in range(0, len(questions), batch_size):
                    batch_questions = questions[i:i+batch_size]
                    batch_answers = model_serving.generate_from_input(batch_questions)
                    answers.extend(batch_answers)
                    logger.info(f"  Progress: {min(i+batch_size, len(questions))}/{len(questions)}")
                
                # Save data with answers
                output_data = data.copy()
                for i, item in enumerate(output_data):
                    item["model_generated_answer"] = answers[i]
                
                # Generate safe filename
                safe_name = "".join(c for c in model['name'] if c.isalnum() or c in ('-', '_'))
                output_file = f"qa_{safe_name}_answers.json"
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
                
                generated_files.append({
                    "model_name": model['name'],
                    "file_path": output_file
                })
                
                logger.info(f"Answers saved: {output_file}")
                
            except Exception as e:
                logger.error(f"Answer generation failed for model {model['name']}: {e}")
                continue
        
        return generated_files

    def _run_evaluation(self) -> List[Dict[str, Any]]:
        """运行评估"""
        data_config = self.config.get("DATA_CONFIG", {})
        
        # 创建评估器LLM服务
        logger.info("创建评估器LLM服务...")
        try:
            judge_serving = self.config["create_judge_serving"]()
            logger.info("评估器服务创建成功")
        except Exception as e:
            logger.error(f"评估器服务创建失败: {e}")
            return []
        
        results = []
        
        for file_info in self.generated_files:
            logger.info(f"评估模型：{file_info['model_name']}")
            
            try:
                # 创建存储
                cache_name = "".join(c for c in file_info['model_name'] if c.isalnum() or c in ('-', '_'))
                storage = self.config["create_storage"](
                    file_info["file_path"],
                    f"./eval_cache_{cache_name}"
                )
                
                # 创建评估器
                result_file = f"./eval_result_{cache_name}.json"
                evaluator = self.config["create_evaluator"](judge_serving, result_file)
                
                # 运行评估
                evaluator.run(
                    storage=storage.step(),
                    input_test_answer_key="model_generated_answer",
                    input_gt_answer_key=data_config.get("reference_answer_key", "output"),
                    input_question_key=data_config.get("question_key", "input")
                )
                
                # 读取结果
                if Path(result_file).exists():
                    with open(result_file, 'r', encoding='utf-8') as f:
                        result_data = json.load(f)
                        if isinstance(result_data, list) and result_data:
                            result_data[0]["model_name"] = file_info['model_name']
                            results.append(result_data[0])
                            
            except Exception as e:
                logger.error(f"模型 {file_info['model_name']} 评估失败：{e}")
                continue
        
        return results

    def _generate_report(self, results: List[Dict[str, Any]]):
        """生成对比报告"""
        if not results:
            logger.warning("没有有效的评估结果")
            return
        
        print("\n" + "="*60)
        print("模型评估结果对比")
        print("="*60)
        
        # 按准确率排序
        sorted_results = sorted(results, key=lambda x: x.get("accuracy", 0), reverse=True)
        
        for i, result in enumerate(sorted_results, 1):
            print(f"{i}. {result.get('model_name', 'Unknown')}")
            print(f"   准确率: {result.get('accuracy', 0):.3f}")
            print(f"   样本数: {result.get('total_samples', 0)}")
            print(f"   匹配数: {result.get('matched_samples', 0)}")
            print()
        
        # 保存详细报告
        report = {
            "evaluation_summary": {
                "total_models": len(results),
                "results": sorted_results
            }
        }
        
        report_file = "model_comparison_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"详细报告已保存：{report_file}")
        print("="*60)

    def _show_no_models_help(self):
        """显示无模型时的帮助信息"""
        print("未检测到可用的模型进行评估")
        print()
        print("解决方案:")
        print("1. 训练模型:")
        print("   dataflow text2model init && dataflow text2model train")
        print()
        print("2. 手动指定模型 - 编辑配置文件:")
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

# 内置模板
EVAL_API_TEMPLATE = '''# eval_api.py - 评估配置文件
"""DataFlow API评估配置"""

import os
from dataflow.operators.core_text import BenchDatasetEvaluator
from dataflow.serving import APILLMServing_request
from dataflow.utils.storage import FileStorage

class FairAnswerJudgePrompt:
    """公平的答案评判提示词模板"""
    
    def build_prompt(self, question, answer, reference_answer):
        prompt = f"""
You are an expert evaluator assessing answer quality.

**Question:**
{question}

**Answer to Evaluate:**
{answer}

Judge based on: factual accuracy, completeness, relevance.
Reference: {reference_answer}

Return JSON: {{"judgement_result": true/false}}

Your judgment:
"""
        return prompt

# 配置参数
JUDGE_MODEL_CONFIG = {
    "model_name": "gpt-4o-mini",
    "api_url": "http://123.129.219.111:3000/v1/chat/completions",
    "api_key_env": "DF_API_KEY",
    "max_workers": 3
}

TARGET_MODELS = {
    "auto_detect": True,
    "models": []
}

DATA_CONFIG = {
    "input_file": "./.cache/data/qa.json",
    "question_key": "input",
    "reference_answer_key": "output"
}

EVAL_CONFIG = {
    "compare_method": "semantic",
    "batch_size": 8,
    "max_tokens": 512,
    # VLLM GPU memory management configuration - only supported parameters
    "vllm_gpu_memory_utilization": 0.75,  # GPU memory utilization (reduced to 75%)
    "vllm_max_model_len": 16384,           # Maximum sequence length (reduced for memory)
}

def create_judge_serving():
    api_key_env = JUDGE_MODEL_CONFIG["api_key_env"]
    if api_key_env not in os.environ:
        raise ValueError(f"环境变量 {api_key_env} 未设置")
    
    return APILLMServing_request(
        api_url=JUDGE_MODEL_CONFIG["api_url"],
        key_name_of_api_key=api_key_env,
        model_name=JUDGE_MODEL_CONFIG["model_name"],
        max_workers=JUDGE_MODEL_CONFIG["max_workers"]
    )

def create_evaluator(judge_serving, eval_result_path):
    return BenchDatasetEvaluator(
        compare_method=EVAL_CONFIG["compare_method"],
        llm_serving=judge_serving,
        prompt_template=FairAnswerJudgePrompt(),
        eval_result_path=eval_result_path
    )

def create_storage(data_file, cache_path):
    return FileStorage(
        first_entry_file_name=data_file,
        cache_path=cache_path,
        file_name_prefix="eval_result",
        cache_type="json"
    )

def get_evaluator_config():
    return {
        "JUDGE_MODEL_CONFIG": JUDGE_MODEL_CONFIG,
        "TARGET_MODELS": TARGET_MODELS,
        "DATA_CONFIG": DATA_CONFIG,
        "EVAL_CONFIG": EVAL_CONFIG,
        "create_judge_serving": create_judge_serving,
        "create_evaluator": create_evaluator,
        "create_storage": create_storage
    }

if __name__ == "__main__":
    from dataflow.cli_funcs.cli_eval import run_evaluation
    config = get_evaluator_config()
    run_evaluation(config)
'''

class DataFlowEvalCLI:
    """DataFlow 评估命令行工具"""
    
    def __init__(self):
        self.current_dir = Path.cwd()
    
    def init_eval_file(self, eval_type: str = "api", output_file: str = None):
        """生成评估配置文件"""
        if eval_type not in ["api", "local"]:
            logger.error("评估类型必须是 'api' 或 'local'")
            return False
        
        if output_file is None:
            output_file = f"eval_{eval_type}.py"
        
        output_path = self.current_dir / output_file
        
        # 检查是否已存在
        if output_path.exists():
            logger.warning(f"文件 {output_file} 已存在，是否覆盖？(y/n)")
            user_input = input().strip().lower()
            if user_input != 'y':
                logger.info("操作取消")
                return False
        
        try:
            # 写入模板内容
            template_content = EVAL_API_TEMPLATE if eval_type == "api" else EVAL_API_TEMPLATE
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(template_content)
                
            logger.info(f"评估配置文件已生成：{output_path}")
            logger.info(f"请编辑 {output_file} 文件配置您的评估参数，然后运行：")
            logger.info(f"python {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"生成文件失败：{str(e)}")
            return False
    
    def run_eval_file(self, eval_type: str, eval_file: str, cli_args=None):
        """运行评估文件"""
        eval_path = Path(eval_file)
        
        if not eval_path.exists():
            logger.error(f"评估文件不存在：{eval_file}")
            logger.info(f"请先运行 'dataflow eval init --type {eval_type}' 生成配置文件")
            return False
        
        try:
            logger.info(f"开始运行 {eval_type} 模型评估：{eval_file}")
            
            # 动态导入配置文件
            spec = importlib.util.spec_from_file_location("user_eval_config", eval_path.resolve())
            if spec is None or spec.loader is None:
                logger.error(f"无法加载配置文件：{eval_file}")
                return False
                
            user_config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(user_config_module)
            
            # 获取配置并运行
            if hasattr(user_config_module, 'get_evaluator_config'):
                config = user_config_module.get_evaluator_config()
                pipeline = EvaluationPipeline(config, cli_args)
                return pipeline.run()
            else:
                logger.error(f"配置文件 {eval_file} 中没有找到 get_evaluator_config 函数")
                return False
                
        except Exception as e:
            logger.error(f"运行评估文件失败：{str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def list_eval_files(self):
        """列出当前目录下的评估文件"""
        eval_files = list(self.current_dir.glob("eval_*.py"))
        if eval_files:
            logger.info("找到以下评估配置文件：")
            for eval_file in eval_files:
                logger.info(f"  - {eval_file.name}")
        else:
            logger.info("当前目录下没有找到评估配置文件")
            logger.info("请运行 'dataflow eval init' 生成配置文件")

# =============================================================================
# 简化的执行接口
# =============================================================================

def run_evaluation(config, cli_args=None):
    """运行评估"""
    logger.info("DataFlow 模型评估开始")
    
    try:
        pipeline = EvaluationPipeline(config, cli_args)
        success = pipeline.run()
        
        if success:
            logger.info("模型评估完成")
        else:
            logger.error("模型评估失败")
            
        return success
        
    except Exception as e:
        logger.error(f"评估失败：{str(e)}")
        raise

def cli_eval():
    """评估命令行入口函数 - 简化版"""
    parser = argparse.ArgumentParser(description="DataFlow 评估工具")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # init 子命令
    init_parser = subparsers.add_parser("init", help="生成评估配置文件")
    init_parser.add_argument("--type", choices=["api", "local"], default="api", help="配置类型")
    init_parser.add_argument("--output", help="输出文件名")
    
    # api 子命令
    api_parser = subparsers.add_parser("api", help="使用API模型评估")
    api_parser.add_argument("--models", help="逗号分隔的模型列表")
    api_parser.add_argument("eval_file", nargs='?', default="eval_api.py", help="评估配置文件")
    
    # local 子命令
    local_parser = subparsers.add_parser("local", help="使用本地模型评估")
    local_parser.add_argument("--models", help="逗号分隔的模型列表")
    local_parser.add_argument("eval_file", nargs='?', default="eval_local.py", help="评估配置文件")
    
    # list 子命令
    list_parser = subparsers.add_parser("list", help="列出评估配置文件")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = DataFlowEvalCLI()
    
    try:
        if args.command == "init":
            success = cli.init_eval_file(args.type, args.output)
            if success:
                logger.info("配置文件初始化成功")
                
        elif args.command in ["api", "local"]:
            success = cli.run_eval_file(args.command, args.eval_file, args)
            if success:
                logger.info("评估完成")
            else:
                logger.error("评估失败")
                
        elif args.command == "list":
            cli.list_eval_files()
            
    except KeyboardInterrupt:
        logger.info("用户中断操作")
    except Exception as e:
        logger.error(f"执行过程中发生错误: {str(e)}")

if __name__ == "__main__":
    cli_eval()