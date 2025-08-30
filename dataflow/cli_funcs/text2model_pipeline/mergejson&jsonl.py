#!/usr/bin/env python3
"""
JSON & JSONL Merger - mergejson&jsonl.py
将多个 JSON/JSONL/Parquet 文件聚合成单个 pt_input.jsonl 文件供 DataFlow 处理
支持保持原始字段结构
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Union, Dict, Any


class TextDataAggregator:
    """文本数据聚合器"""

    def __init__(self, output_file: str = "./.cache/pt_input.jsonl"):
        # 处理输出路径
        output_path = Path(output_file)
        if not output_path.is_absolute():
            caller_cwd = Path(os.getcwd())
            output_file = str(caller_cwd / output_file)
        self.output_file = output_file
        self.input_files = []
        self.supported_formats = {'.json', '.jsonl', '.parquet'}

    def scan_directory(self, directory: Union[str, Path], recursive: bool = True) -> List[str]:
        """扫描目录中的支持格式文件"""
        directory = Path(directory)

        if not directory.exists():
            print(f"错误：目录 '{directory}' 不存在")
            return []

        if not directory.is_dir():
            print(f"错误：'{directory}' 不是有效目录")
            return []

        found_files = []

        # 排除的目录
        exclude_dirs = {'.cache', '__pycache__', '.git', 'node_modules', '.venv', 'venv', '.env', 'cache'}

        if recursive:
            patterns = ["**/*.*"]
        else:
            patterns = ["*.*"]

        for pattern in patterns:
            for file_path in directory.glob(pattern):
                # 跳过排除目录
                if any(exclude_dir in file_path.parts for exclude_dir in exclude_dirs):
                    continue

                # 跳过隐藏目录
                if any(part.startswith('.') and part != '.' for part in file_path.parts):
                    continue

                # 检查支持的格式
                if file_path.suffix.lower() in self.supported_formats and file_path.is_file():
                    found_files.append(str(file_path.resolve()))
                    print(f"找到文件: {file_path}")

        self.input_files.extend(found_files)
        return found_files

    def extract_data_from_json(self, data: Any, text_keys: List[str]) -> List[Dict]:
        """从JSON数据中提取数据，保持原始结构"""
        results = []

        if isinstance(data, dict):
            # 过滤掉空值字段，保持原始结构
            filtered_data = {}
            for key, value in data.items():
                if isinstance(value, str) and value.strip():
                    filtered_data[key] = value.strip()
                elif not isinstance(value, str) and value is not None:
                    filtered_data[key] = value

            if filtered_data:  # 只有非空数据才添加
                results.append(filtered_data)

        elif isinstance(data, list):
            for item in data:
                results.extend(self.extract_data_from_json(item, text_keys))

        return results

    def process_json_file(self, file_path: str, text_keys: List[str]) -> List[Dict]:
        """处理JSON文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return self.extract_data_from_json(data, text_keys)
        except Exception as e:
            print(f"处理JSON文件 {file_path} 时出错: {e}")
            return []

    def process_jsonl_file(self, file_path: str, text_keys: List[str]) -> List[Dict]:
        """处理JSONL文件"""
        results = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            results.extend(self.extract_data_from_json(data, text_keys))
                        except json.JSONDecodeError as e:
                            print(f"跳过 {file_path} 第{line_num}行（JSON解析错误）: {e}")
                            continue
            return results
        except Exception as e:
            print(f"处理JSONL文件 {file_path} 时出错: {e}")
            return []

    def process_parquet_file(self, file_path: str, text_keys: List[str]) -> List[Dict]:
        """处理Parquet文件"""
        try:
            import pandas as pd
            df = pd.read_parquet(file_path)

            results = []
            # 保持原始结构：转换整个DataFrame
            for _, row in df.iterrows():
                row_dict = {}
                for col in df.columns:
                    if pd.notna(row[col]) and str(row[col]).strip():
                        if isinstance(row[col], str):
                            row_dict[col] = row[col].strip()
                        else:
                            row_dict[col] = row[col]

                if row_dict:  # 只添加非空行
                    results.append(row_dict)

            return results
        except ImportError:
            print(f"需要安装 pandas 来处理 Parquet 文件: pip install pandas pyarrow")
            return []
        except Exception as e:
            print(f"处理Parquet文件 {file_path} 时出错: {e}")
            return []

    def validate_text_length(self, item: Dict, min_length: int) -> bool:
        """验证数据项的文本长度"""
        # 检查所有字符串字段的总长度
        total_length = 0
        for value in item.values():
            if isinstance(value, str):
                total_length += len(value.strip())
        return total_length >= min_length

    def aggregate_files(self, text_keys: List[str] = None, min_length: int = 10) -> bool:
        """聚合所有文件，保持原始JSON结构"""
        if text_keys is None:
            text_keys = ['raw_content']  # 默认字段

        if not self.input_files:
            print("没有找到要处理的文件")
            return False

        # 确保输出目录存在
        output_path = Path(self.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total_items = 0
        processed_files = 0
        all_field_names = set()

        with open(self.output_file, 'w', encoding='utf-8') as f:
            for file_path in self.input_files:
                print(f"处理文件: {file_path}")

                file_path_obj = Path(file_path)
                suffix = file_path_obj.suffix.lower()

                # 根据文件格式选择处理方法
                if suffix == '.json':
                    items = self.process_json_file(file_path, text_keys)
                elif suffix == '.jsonl':
                    items = self.process_jsonl_file(file_path, text_keys)
                elif suffix == '.parquet':
                    items = self.process_parquet_file(file_path, text_keys)
                else:
                    print(f"跳过不支持的文件格式: {file_path}")
                    continue

                # 写入有效数据
                file_item_count = 0
                for item in items:
                    if self.validate_text_length(item, min_length):
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                        total_items += 1
                        file_item_count += 1

                        # 收集所有字段名用于统计
                        all_field_names.update(item.keys())

                if file_item_count > 0:
                    processed_files += 1
                    print(f"  提取数据: {file_item_count} 条")
                else:
                    print(f"  未找到有效数据")

        print(f"\n聚合完成！")
        print(f"处理文件: {processed_files}/{len(self.input_files)}")
        print(f"总数据条目: {total_items}")
        print(f"输出文件: {self.output_file}")
        print(f"包含字段: {sorted(list(all_field_names))}")

        return total_items > 0

    def preview_files(self, max_files: int = 10):
        """预览找到的文件"""
        if not self.input_files:
            print("没有找到文件")
            return

        print(f"\n找到 {len(self.input_files)} 个文件:")
        print("-" * 60)

        for i, file_path in enumerate(self.input_files[:max_files]):
            file_size = Path(file_path).stat().st_size
            print(f"{i + 1:3d}. {file_path} ({file_size:,} bytes)")

        if len(self.input_files) > max_files:
            print(f"... 还有 {len(self.input_files) - max_files} 个文件")
        print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description='聚合多个文本数据文件为单个 pt_input.jsonl')
    parser.add_argument('input_dir', nargs='?', default='./',
                        help='输入目录路径 (默认: ./)')
    parser.add_argument('-o', '--output', default='./.cache/pt_input.jsonl',
                        help='输出文件路径 (默认: ./.cache/pt_input.jsonl)')
    parser.add_argument('-k', '--keys', nargs='+', default=['raw_content'],
                        help='要提取的文本字段 (默认: raw_content)')
    parser.add_argument('-r', '--recursive', action='store_true', default=True,
                        help='递归扫描子目录')
    parser.add_argument('--no-recursive', action='store_false', dest='recursive',
                        help='不递归扫描子目录')
    parser.add_argument('--min-length', type=int, default=10,
                        help='最小文本长度 (默认: 10)')
    parser.add_argument('--preview', action='store_true',
                        help='只预览文件，不执行聚合')

    args = parser.parse_args()

    # 验证输入目录
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"错误：输入目录 '{input_path}' 不存在")
        return

    if not input_path.is_dir():
        print(f"错误：'{input_path}' 不是目录")
        return

    # 创建聚合器
    aggregator = TextDataAggregator(args.output)

    # 扫描文件
    print(f"扫描目录: {input_path}")
    print(f"递归模式: {'启用' if args.recursive else '禁用'}")
    print(f"提取字段: {args.keys}")
    print(f"最小长度: {args.min_length}")

    aggregator.scan_directory(input_path, args.recursive)

    if args.preview:
        # 只预览
        aggregator.preview_files()
    else:
        # 预览并聚合
        aggregator.preview_files()
        print("\n开始聚合...")

        if aggregator.aggregate_files(args.keys, args.min_length):
            print(f"\n✅ 聚合成功！现在可以运行: dataflow text2model")
        else:
            print(f"\n❌ 聚合失败，请检查输入文件")


if __name__ == "__main__":
    main()