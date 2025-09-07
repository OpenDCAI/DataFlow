#!/usr/bin/env python3
import json
import os
import argparse
from pathlib import Path


def extract_text_from_json(json_file_path):
    """从JSON/JSONL文件中提取文本内容"""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            if json_file_path.suffix == '.jsonl':
                # 处理JSONL文件（每行一个JSON对象）
                texts = []
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        text_content = extract_text_from_data(data)
                        if text_content:
                            texts.append(text_content)
                    except json.JSONDecodeError as e:
                        print(f"  ⚠️  Warning: Invalid JSON at line {line_num} in {json_file_path}: {e}")
                        continue
                return texts
            else:
                # 处理标准JSON文件
                data = json.load(f)
                text_content = extract_text_from_data(data)
                return [text_content] if text_content else []
                
    except Exception as e:
        print(f"  ❌ Error reading {json_file_path}: {e}")
        return []


def extract_text_from_data(data):
    """从JSON数据中提取文本内容，确保返回tokenizer友好的格式"""
    text_fields = ['text', 'content', 'body', 'raw_text', 'message', 'description', 'summary']
    
    def clean_text(text):
        """清理文本，确保tokenizer兼容"""
        if not isinstance(text, str):
            return None
        
        # 移除或替换问题字符
        text = text.replace('\x00', '')  # 移除null字符
        text = text.replace('\ufeff', '')  # 移除BOM
        text = text.replace('\r\n', '\n')  # 统一换行符
        text = text.replace('\r', '\n')
        
        # 移除控制字符（保留常见的换行、制表符）
        import re
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # 确保文本不为空
        text = text.strip()
        if not text:
            return None
            
        # 限制文本长度（防止过长文本导致问题）
        max_length = 50000  # 50K字符限制
        if len(text) > max_length:
            print(f"  ⚠️  Warning: Text too long ({len(text)} chars), truncating to {max_length}")
            text = text[:max_length]
        
        return text
    
    if isinstance(data, dict):
        # 查找文本字段
        for field in text_fields:
            if field in data and data[field]:
                content = data[field]
                if isinstance(content, str):
                    cleaned = clean_text(content)
                    if cleaned:
                        return cleaned
                elif isinstance(content, (list, dict)):
                    # 递归处理嵌套结构
                    nested_text = extract_text_from_data(content)
                    if nested_text:
                        return clean_text(nested_text)
        
        # 如果没找到标准字段，尝试查找所有字符串值
        for key, value in data.items():
            if isinstance(value, str) and len(value.strip()) > 50:
                cleaned = clean_text(value)
                if cleaned:
                    return cleaned
                    
    elif isinstance(data, list):
        # 如果是列表，处理每个元素
        texts = []
        for item in data:
            if isinstance(item, dict):
                text_content = extract_text_from_data(item)
                if text_content:
                    texts.append(text_content)
            elif isinstance(item, str):
                cleaned = clean_text(item)
                if cleaned:
                    texts.append(cleaned)
        
        if texts:
            combined = "\n\n".join(texts)
            return clean_text(combined)
            
    elif isinstance(data, str):
        return clean_text(data)
    
    return None


def create_text_files(json_files, cache_path_obj):
    """将JSON文件中的文本内容转换为独立的文本文件，确保格式正确"""
    temp_dir = cache_path_obj / ".cache" / "gpu" / "temp_texts"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    text_paths = []
    processed_count = 0
    
    print("\nExtracting text content from JSON files...")
    print("-" * 50)
    
    for i, json_file in enumerate(json_files):
        print(f"Processing: {json_file.name}")
        
        # 提取文本内容
        text_contents = extract_text_from_json(json_file)
        
        if not text_contents:
            print(f"  ⚠️  No valid text content found in {json_file}")
            continue
        
        # 为每个文本内容创建文件
        for j, text_content in enumerate(text_contents):
            if not text_content or not isinstance(text_content, str) or not text_content.strip():
                continue
            
            # 最终验证和清理文本
            final_text = validate_and_clean_text(text_content)
            if not final_text:
                print(f"  ⚠️  Text content failed validation")
                continue
                
            # 创建文本文件名
            base_name = json_file.stem
            if len(text_contents) > 1:
                temp_file = temp_dir / f"{base_name}_{j}.txt"
            else:
                temp_file = temp_dir / f"{base_name}.txt"
            
            try:
                with open(temp_file, 'w', encoding='utf-8') as tf:
                    tf.write(final_text)
                
                # 验证写入的文件
                if verify_text_file(temp_file):
                    text_paths.append(str(temp_file))
                    processed_count += 1
                    print(f"  ✅ Created: {temp_file.name} ({len(final_text)} chars)")
                else:
                    print(f"  ❌ Failed validation: {temp_file.name}")
                    temp_file.unlink()  # 删除无效文件
                    
            except Exception as e:
                print(f"  ❌ Error creating {temp_file}: {e}")
    
    print(f"\n✅ Successfully processed {processed_count} text contents from {len(json_files)} JSON files")
    return text_paths


def validate_and_clean_text(text):
    """验证并清理文本，确保tokenizer兼容"""
    if not isinstance(text, str):
        return None
    
    # 基本清理
    text = text.strip()
    if not text:
        return None
    
    # 移除或替换问题字符
    import re
    
    # 移除null字符和其他控制字符
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # 移除BOM
    text = text.replace('\ufeff', '')
    
    # 统一换行符
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # 移除过多的连续空白字符
    text = re.sub(r'\n{3,}', '\n\n', text)  # 最多保留两个连续换行
    text = re.sub(r'[ \t]{2,}', ' ', text)  # 多个空格/制表符替换为单个空格
    
    # 确保文本不为空
    text = text.strip()
    if not text:
        return None
    
    # 检查文本长度
    if len(text) < 10:  # 太短的文本可能无意义
        return None
    
    # 限制最大长度
    max_length = 50000
    if len(text) > max_length:
        text = text[:max_length]
        # 尝试在句号处截断，避免截断句子
        last_period = text.rfind('.')
        if last_period > max_length * 0.8:  # 如果句号位置合理
            text = text[:last_period + 1]
    
    return text


def verify_text_file(file_path):
    """验证文本文件是否可以被正确读取和tokenize"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 基本检查
        if not content or not content.strip():
            return False
        
        # 检查是否包含非法字符
        if '\x00' in content:
            return False
        
        # 检查是否是纯文本（不是JSON等结构化数据）
        content_stripped = content.strip()
        if (content_stripped.startswith('{') and content_stripped.endswith('}')) or \
           (content_stripped.startswith('[') and content_stripped.endswith(']')):
            # 可能是JSON，需要进一步验证
            try:
                import json
                json.loads(content)
                return False  # 如果是有效JSON，则不是纯文本
            except:
                pass  # 不是有效JSON，可能是包含{}[]的普通文本，继续
        
        return True
        
    except Exception as e:
        print(f"  ⚠️  File verification failed: {e}")
        return False


def scan_json_files(base_dir, cache_path="./", mode="paths"):
    """
    扫描目录下的JSON/JSONL文件，排除.cache目录
    mode: 'paths' - 创建文件路径索引, 'texts' - 提取文本内容
    """
    base_path = Path(base_dir)
    cache_path_obj = Path(cache_path)
    if not cache_path_obj.is_absolute():
        caller_cwd = Path(os.environ.get('PWD', os.getcwd()))
        cache_path_obj = caller_cwd / cache_path_obj

    # 确保缓存目录存在
    gpu_cache_dir = cache_path_obj / ".cache" / "gpu"
    gpu_cache_dir.mkdir(parents=True, exist_ok=True)

    text_input_file = gpu_cache_dir / "text_input.jsonl"

    print(f"Scanning directory: {base_path}")
    print(f"Output file: {text_input_file}")
    print(f"Mode: {'Extract text content' if mode == 'texts' else 'File paths only'}")
    print("-" * 50)

    json_files = []

    # 遍历所有文件和目录
    for root, dirs, files in os.walk(base_path):
        # 排除.cache目录
        dirs[:] = [d for d in dirs if not d.startswith('.cache')]

        # 检查每个文件
        for file in files:
            if file.endswith(('.json', '.jsonl')):
                file_path = Path(root) / file
                json_files.append(file_path.absolute())
                print(f"Found: {file_path}")

    if not json_files:
        print("No JSON/JSONL files found!")
        return False

    print(f"\nTotal found: {len(json_files)} files")
    
    # 根据模式处理
    if mode == "texts":
        print("Mode: Extract text content and create text files...")
        text_paths = create_text_files(json_files, cache_path_obj)
        
        if not text_paths:
            print("❌ No valid text content found in any JSON files!")
            return False
        
        # 创建text_input.jsonl文件，包含文本文件路径
        try:
            with open(text_input_file, 'w', encoding='utf-8') as f:
                for text_path in text_paths:
                    record = {"text_path": text_path}
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')

            print(f"✅ Created: {text_input_file}")
            print(f"✅ Contains {len(text_paths)} text file paths")
            return True

        except Exception as e:
            print(f"❌ Error creating file: {e}")
            return False
            
    else:
        print("Mode: Create file path index...")
        # 创建text_input.jsonl文件，包含JSON文件路径
        try:
            with open(text_input_file, 'w', encoding='utf-8') as f:
                for json_file in json_files:
                    record = {"text_path": str(json_file)}
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')

            print(f"✅ Created: {text_input_file}")
            print(f"✅ Contains {len(json_files)} file paths")
            return True

        except Exception as e:
            print(f"❌ Error creating file: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description='Scan JSON/JSONL files and create input for text processing pipeline')
    parser.add_argument('input_dir', nargs='?', default='./', help='Input directory to scan (default: ./)')
    parser.add_argument('--cache', default='./', help='Cache directory path (default: ./)')
    parser.add_argument('--mode', choices=['paths', 'texts'], default='texts', 
                       help='Processing mode: "paths" for file paths only, "texts" for extract text content (default: texts)')

    args = parser.parse_args()

    print("Enhanced JSON/JSONL File Scanner & Text Processor")
    print("=" * 60)

    success = scan_json_files(args.input_dir, args.cache, args.mode)

    if success:
        print("✅ Processing completed successfully!")
        if args.mode == 'texts':
            print("Ready for CorpusTextSplitterBatch processing with extracted text files.")
            print("\nNext steps:")
            print("1. Run your text2qa pipeline")
            print("2. The pipeline will process the text files automatically")
        else:
            print("Ready for CorpusTextSplitterBatch processing with JSON file paths.")
            print("\nNote: Make sure your JSON files contain text fields like 'text', 'content', or 'body'")
    else:
        print("❌ Processing failed!")


if __name__ == "__main__":
    main()