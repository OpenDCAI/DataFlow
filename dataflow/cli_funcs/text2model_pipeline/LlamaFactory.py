#!/usr/bin/env python3
"""
LlamaFactory Training Script with YAML Configuration Management
Complete "check-create-read-train" workflow
"""

import yaml
import os
import sys
import subprocess
import argparse
from pathlib import Path


class LlamaFactoryTrainer:
    def __init__(self, config_path=None, cache_base="./"):
        # 处理cache_base相对路径 - 基于调用者工作目录
        cache_path = Path(cache_base)
        if not cache_path.is_absolute():
            caller_cwd = Path(os.environ.get('PWD', os.getcwd()))
            cache_path = caller_cwd / cache_path

        self.cache_path = cache_path

        # 如果没有指定config_path，使用cache目录下的默认路径
        if config_path is None:
            config_path = str(cache_path / ".cache" / "train_config.yaml")
        else:
            # 处理config_path相对路径
            config_path_obj = Path(config_path)
            if not config_path_obj.is_absolute():
                caller_cwd = Path(os.environ.get('PWD', os.getcwd()))
                config_path = str(caller_cwd / config_path_obj)

        self.config_path = Path(config_path)

    def get_default_config(self):
        """Get default configuration - 基于最新LlamaFactory标准"""
        return {
            # === 基础配置 ===
            "stage": "sft",
            "do_train": True,

            # === 模型配置 ===
            "model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
            "template": "qwen",
            "trust_remote_code": True,

            # === 微调方法配置 ===
            "finetuning_type": "lora",
            "lora_target": "all",  # 使用 "all" 而不是具体层名
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,

            # === 数据集配置 ===
            "dataset": "kb_qa",
            "dataset_dir": str(self.cache_path / ".cache" / "data"),
            "cutoff_len": 1024,
            "max_samples": None,
            "overwrite_cache": True,
            "preprocessing_num_workers": 4,

            # === 训练配置 ===
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "learning_rate": 5e-5,
            "num_train_epochs": 2.0,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.05,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "optim": "adamw_torch",

            # === 输出配置 ===
            "output_dir": str(self.cache_path / ".cache" / "saves" / "qwen2.5_7b_sft_model"),
            "overwrite_output_dir": True,
            "save_only_model": True,
            "plot_loss": True,

            # === 日志和检查点 ===
            "logging_steps": 10,
            "save_steps": 300,
            "save_total_limit": 2,

            # === 评估配置 ===
            "val_size": 0.1,
            "per_device_eval_batch_size": 2,
            "eval_strategy": "steps",
            "eval_steps": 300,

            # === 硬件配置 ===
            "fp16": False,
            "bf16": True,
            "tf32": True,
            "dataloader_num_workers": 2,

            # === 其他配置 ===
            "seed": 42,
            "ddp_timeout": 1800,
            "report_to": "none",
            "run_name": "qwen2.5_7b_sft_training",
        }

    def check_and_create_config(self):
        """
        Core logic: check config file, read if exists, create if not
        """
        # Ensure directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        if self.config_path.exists():
            # Config file exists, read it
            print(f"Found config file: {self.config_path}")
            return self._load_existing_config()
        else:
            # Config file doesn't exist, create default config
            print(f"Creating new config file: {self.config_path}")
            return self._create_new_config()

    def _load_existing_config(self):
        """Load existing config file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print("Config file loaded successfully")

            # Check if new default parameters need to be added
            default_config = self.get_default_config()
            updated = False

            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
                    updated = True
                    print(f"Added missing parameter: {key}")

            # 移除过时的formatting参数
            if "formatting" in config:
                del config["formatting"]
                updated = True
                print("Removed deprecated parameter: formatting")

            # Save config if updated
            if updated:
                self._save_config(config)
                print("Config file updated")

            return config

        except Exception as e:
            print(f"Failed to read config: {e}")
            print("Creating new default config")
            return self._create_new_config()

    def _create_new_config(self):
        """Create new default config file"""
        default_config = self.get_default_config()

        if self._save_config(default_config):
            print("Default config file created successfully")
            return default_config
        else:
            print("Failed to create config file")
            return None

    def _save_config(self, config):
        """Save config to file"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f,
                          default_flow_style=False,
                          allow_unicode=True,
                          sort_keys=False,
                          indent=2,
                          width=80)
            return True
        except Exception as e:
            print(f"Failed to save config: {e}")
            return False

    def update_config(self, updates):
        """Update config parameters"""
        config = self.check_and_create_config()
        if config is None:
            return None

        config.update(updates)
        if self._save_config(config):
            print(f"Config updated: {list(updates.keys())}")
            return config
        return None

    def print_config_info(self, config):
        """Print config information"""
        if not config:
            return

        print("\n" + "=" * 50)
        print("Training Configuration")
        print("=" * 50)

        key_info = [
            ("Model", "model_name_or_path"),
            ("Dataset", "dataset"),
            ("Template", "template"),
            ("LoRA Rank", "lora_rank"),
            ("Learning Rate", "learning_rate"),
            ("Epochs", "num_train_epochs"),
            ("Batch Size", "per_device_train_batch_size"),
            ("Output Dir", "output_dir"),
        ]

        for name, key in key_info:
            print(f"{name:<12}: {config.get(key, 'N/A')}")

        effective_batch = config.get("per_device_train_batch_size", 1) * config.get("gradient_accumulation_steps", 1)
        print(f"Effective Batch: {effective_batch}")
        print("=" * 50)

    def check_environment(self):
        """Check training environment"""
        print("Checking training environment...")

        # 使用绝对路径 - 直接获取调用者工作目录
        caller_cwd = Path(os.environ.get('PWD', os.getcwd()))
        data_dir = caller_cwd / ".cache" / "data"

        # Check data files with detailed info
        data_files = [
            str(data_dir / "qa.json"),
            str(data_dir / "dataset_info.json")
        ]

        for file_path in data_files:
            if not Path(file_path).exists():
                print(f"❌ Missing data file: {file_path}")
                return False
            else:
                file_size = Path(file_path).stat().st_size
                print(f"✅ Found data file: {file_path} ({file_size} bytes)")

        # 检查qa.json的内容
        try:
            import json
            qa_file = str(data_dir / "qa.json")
            with open(qa_file, 'r', encoding='utf-8') as f:
                qa_data = json.load(f)
            print(f"✅ QA data loaded: {len(qa_data)} samples")

            if len(qa_data) == 0:
                print("❌ QA data is empty! Please check data generation pipeline.")
                return False

            # 检查第一个样本的格式
            if qa_data:
                sample = qa_data[0]
                required_keys = ["instruction", "input", "output"]
                missing_keys = [key for key in required_keys if key not in sample]
                if missing_keys:
                    print(f"❌ Missing keys in QA sample: {missing_keys}")
                    return False
                else:
                    print("✅ QA data format is correct")

        except Exception as e:
            print(f"❌ Error checking QA data: {e}")
            return False

        print("✅ Data files check passed")

        # Check LlamaFactory
        try:
            import llamafactory
            print("✅ LlamaFactory is installed")
        except ImportError:
            print("❌ LlamaFactory not installed")
            print("Please run: pip install llamafactory[torch,metrics]")
            return False

        return True

    def start_training(self):
        """Start training"""
        if not self.check_environment():
            print("Environment check failed")
            return False

        # Load config
        config = self.check_and_create_config()
        if not config:
            print("Failed to load config")
            return False

        # Show config info
        self.print_config_info(config)

        # Build training command
        train_cmd = f"llamafactory-cli train {self.config_path}"

        # Start training
        print(f"\nStarting training...")
        print(f"Config file: {self.config_path}")
        print(f"Command: {train_cmd}")

        try:
            result = subprocess.run(train_cmd, shell=True, check=True)
            print("Training completed!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Training failed: {e}")
            return False
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            return False


def main():
    parser = argparse.ArgumentParser(description="LlamaFactory Training Script")
    parser.add_argument("--config", default=None,
                        help="Config file path (default: cache_dir/.cache/train_config.yaml)")
    parser.add_argument("--cache", default="./", help="Cache directory path")
    parser.add_argument("--update-lr", type=float,
                        help="Update learning rate")
    parser.add_argument("--update-epochs", type=int,
                        help="Update training epochs")
    parser.add_argument("--update-batch-size", type=int,
                        help="Update batch size")

    args = parser.parse_args()

    # 处理cache_base相对路径
    cache_path = Path(args.cache)
    if not cache_path.is_absolute():
        caller_cwd = Path(os.environ.get('PWD', os.getcwd()))
        cache_path = caller_cwd / cache_path

    # Create trainer - 如果没有指定config，会自动使用cache目录下的默认路径
    config_path = args.config
    if config_path is None:
        config_path = str(cache_path / ".cache" / "train_config.yaml")

    trainer = LlamaFactoryTrainer(config_path, cache_base=str(cache_path))

    # Update parameters if specified
    updates = {}
    if args.update_lr:
        updates["learning_rate"] = args.update_lr
    if args.update_epochs:
        updates["num_train_epochs"] = args.update_epochs
    if args.update_batch_size:
        updates["per_device_train_batch_size"] = args.update_batch_size

    if updates:
        print("Updating config parameters...")
        trainer.update_config(updates)

    # Start training
    success = trainer.start_training()

    if success:
        print(f"\nOperation completed!")
        print(f"Model saved to: {cache_path / '.cache' / 'saves' / 'qwen2.5_7b_sft_model'}")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()