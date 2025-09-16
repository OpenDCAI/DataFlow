#!/usr/bin/env python3
import json
import os
from pathlib import Path


def get_script_dir():
    """获取脚本文件所在目录的绝对路径"""
    return Path(__file__).parent.absolute()


def find_input_file():
    """相对于脚本位置查找输入文件"""
    script_dir = get_script_dir()
    print(f"📂 脚本位置: {script_dir}")

    # 相对于脚本位置的可能路径
    possible_paths = [
        script_dir / ".cache" / "gpu" / "batch_cleaning_step_step4.json",
        script_dir / "cache" / "gpu" / "batch_cleaning_step_step4.json",
        script_dir / "batch_cleaning_step_step4.json",
        script_dir.parent / ".cache" / "gpu" / "batch_cleaning_step_step4.json",  # 上级目录
        script_dir / ".." / ".cache" / "gpu" / "batch_cleaning_step_step4.json",  # 相对路径形式
    ]

    print("🔍 搜索输入文件...")
    for path in possible_paths:
        abs_path = path.resolve()  # 转换为绝对路径
        if abs_path.exists():
            size = abs_path.stat().st_size
            print(f"✅ 找到输入文件: {abs_path} ({size} 字节)")
            return abs_path
        else:
            print(f"❌ 未找到: {abs_path}")

    print("❌ 找不到输入文件！")
    return None


def convert_to_alpaca(input_file, output_dir=None):
    """转换为Alpaca格式"""
    script_dir = get_script_dir()

    # 如果没有指定输出目录，使用脚本同级的data目录
    if output_dir is None:
        output_dir = script_dir / "data"
    else:
        output_dir = Path(output_dir)

    print(f"📖 读取数据文件: {input_file}")
    print(f"📁 输出目录: {output_dir}")

    results = []

    # 读取数据
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ 成功读取数据，类型: {type(data)}, 长度: {len(data) if hasattr(data, '__len__') else 'N/A'}")
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return None

    # 学术论文专用instruction
    instruction = (
        "Please answer the following question based on the provided academic literature. "
        "Your response should:\n"
        "1. Provide accurate information from the source material\n"
        "2. Include relevant scientific reasoning and methodology\n"
        "3. Reference specific findings, data, or conclusions when applicable\n"
        "4. Maintain academic rigor and precision in your explanation\n\n"
        "Focus on delivering factual, evidence-based answers suitable for academic research."
    )

    # 处理每个QA对
    processed_items = 0
    total_qa_pairs = 0

    print("🔄 处理QA对...")

    for i, item in enumerate(data):
        print(f"处理项目 {i + 1}/{len(data)}: ", end="")

        # 检查数据结构
        if not isinstance(item, dict):
            print("跳过（非字典格式）")
            continue

        if "MultiHop_QA" not in item:
            print("跳过（无MultiHop_QA字段）")
            # 打印可用字段供调试
            if i == 0:  # 只打印第一个的字段
                print(f"   可用字段: {list(item.keys())}")
            continue

        multihop_qa = item.get("MultiHop_QA", {})
        if not isinstance(multihop_qa, dict):
            print("跳过（MultiHop_QA不是字典）")
            continue

        qa_pairs = multihop_qa.get("qa_pairs", [])
        if not qa_pairs:
            print("跳过（无qa_pairs）")
            continue

        print(f"找到 {len(qa_pairs)} 个QA对")
        processed_items += 1

        for qa in qa_pairs:
            if not isinstance(qa, dict):
                continue

            question = qa.get("question", "").strip()
            answer_text = qa.get("answer", "").strip()

            # 跳过空问题或答案
            if not question or not answer_text:
                continue

            # 合并推理步骤
            reasoning_steps = qa.get("reasoning_steps", [])
            reasoning_text = "\n".join(
                [step.get("step", "").strip() for step in reasoning_steps if
                 isinstance(step, dict) and step.get("step", "").strip()])

            # 构建输出（推理过程 + 答案）
            if reasoning_text:
                output_text = f"{reasoning_text}\n\n{answer_text}"
            else:
                output_text = answer_text

            results.append({
                "instruction": instruction,
                "input": question,
                "output": output_text
            })

            total_qa_pairs += 1

    print(f"\n📊 处理统计:")
    print(f"总数据项: {len(data)}")
    print(f"有效项目: {processed_items}")
    print(f"转换QA对: {total_qa_pairs}")

    if not results:
        print("❌ 没有转换任何QA对！请检查数据格式")

        # 显示第一个数据项的结构供调试
        if data and isinstance(data[0], dict):
            print("📋 第一个数据项的结构:")
            print(json.dumps(data[0], indent=2, ensure_ascii=False)[:500] + "...")

        return None

    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存为qa.json（LlamaFactory标准格式）
    qa_file = output_dir / "qa.json"
    try:
        with open(qa_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        file_size = qa_file.stat().st_size
        print(f"✅ 转换完成: {len(results)} 个QA对 -> {qa_file} ({file_size} 字节)")

        return qa_file
    except Exception as e:
        print(f"❌ 保存文件失败: {e}")
        return None


def create_llamafactory_config(output_dir=None):
    """Create dataset_info.json for LlamaFactory"""
    script_dir = get_script_dir()

    if output_dir is None:
        output_dir = script_dir / "data"
    else:
        output_dir = Path(output_dir)

    print("📋 创建LlamaFactory配置...")

    # LlamaFactory dataset configuration
    dataset_info = {
        "kb_qa": {
            "file_name": "qa.json",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output"
            }
        }
    }

    config_file = output_dir / "dataset_info.json"
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)

        print(f"✅ LlamaFactory配置创建: {config_file}")
        print(f"数据集名称: kb_qa")
        return config_file
    except Exception as e:
        print(f"❌ 创建配置失败: {e}")
        return None


def verify_output(output_dir=None):
    """验证输出文件"""
    script_dir = get_script_dir()

    if output_dir is None:
        output_dir = script_dir / "data"
    else:
        output_dir = Path(output_dir)

    print(f"\n🔍 验证输出文件 (目录: {output_dir})...")

    qa_file = output_dir / "qa.json"
    config_file = output_dir / "dataset_info.json"

    # 检查qa.json
    if qa_file.exists():
        size = qa_file.stat().st_size
        print(f"✅ qa.json: {size} 字节")

        try:
            with open(qa_file, 'r', encoding='utf-8') as f:
                qa_data = json.load(f)
            print(f"✅ qa.json包含 {len(qa_data)} 个样本")

            if qa_data:
                sample = qa_data[0]
                print(f"📋 样本字段: {list(sample.keys())}")
        except Exception as e:
            print(f"❌ qa.json验证失败: {e}")
    else:
        print(f"❌ 未找到 qa.json")

    # 检查dataset_info.json
    if config_file.exists():
        print(f"✅ dataset_info.json 存在")
    else:
        print(f"❌ 未找到 dataset_info.json")


if __name__ == "__main__":
    print("🚀 QA数据转换工具（相对路径版）")
    print("=" * 50)

    script_dir = get_script_dir()
    print(f"📂 脚本所在目录: {script_dir}")

    # 查找输入文件（相对于脚本位置）
    input_file = find_input_file()
    if not input_file:
        print("\n💡 提示：")
        print("1. 确保已运行 Pdf2QAPipeline.py")
        print("2. 检查 .cache/gpu/ 目录是否存在")
        print("3. 如果文件在其他位置，请手动指定路径")
        exit(1)

    # 输出目录（相对于脚本位置）
    output_dir = script_dir / "data"

    print(f"\n开始转换...")
    print(f"输入: {input_file}")
    print(f"输出目录: {output_dir}")
    print("-" * 50)

    # Convert data
    qa_file = convert_to_alpaca(input_file, output_dir)

    if qa_file:
        # Create config file
        config_file = create_llamafactory_config(output_dir)

        if config_file:
            print(f"\n🎉 数据转换完成!")
            verify_output(output_dir)

            print(f"\n现在可以运行训练:")
            print(f"python LlamaFactory.py --dry-run  # 预览")
            print(f"python LlamaFactory.py           # 实际训练")
        else:
            print("❌ 配置文件创建失败")
    else:
        print("❌ 数据转换失败")

    print(f"\n📂 所有路径都相对于脚本位置: {script_dir}")