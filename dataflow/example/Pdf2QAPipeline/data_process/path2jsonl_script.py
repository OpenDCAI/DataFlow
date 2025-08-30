import os
import json
import argparse
from pathlib import Path
from typing import List, Union


class PDFDetector:
    """PDF文件检测器，用于扫描目录并生成JSONL配置文件"""

    def __init__(self, output_file: str = "../input/pdf_list.jsonl"):  # 修改默认输出路径
        self.output_file = output_file
        self.pdf_files = []

    def scan_directory(self, directory: Union[str, Path], recursive: bool = True) -> List[str]:
        """
        扫描目录中的PDF文件

        Args:
            directory: 要扫描的目录路径
            recursive: 是否递归扫描子目录

        Returns:
            找到的PDF文件路径列表
        """
        directory = Path(directory)

        if not directory.exists():
            print(f"错误: 目录 '{directory}' 不存在")
            return []

        if not directory.is_dir():
            print(f"错误: '{directory}' 不是一个有效目录")
            return []

        pdf_files = []

        if recursive:
            # 递归搜索所有子目录
            pattern = "**/*.pdf"
        else:
            # 只搜索当前目录
            pattern = "*.pdf"

        for pdf_path in directory.glob(pattern):
            if pdf_path.is_file():
                # 转换为相对路径或绝对路径
                pdf_files.append(str(pdf_path.resolve()))
                print(f"发现PDF: {pdf_path}")

        self.pdf_files.extend(pdf_files)
        return pdf_files

    def scan_multiple_directories(self, directories: List[Union[str, Path]], recursive: bool = True) -> List[str]:
        """
        扫描多个目录

        Args:
            directories: 目录路径列表
            recursive: 是否递归扫描

        Returns:
            所有找到的PDF文件路径列表
        """
        all_pdfs = []
        for directory in directories:
            pdfs = self.scan_directory(directory, recursive)
            all_pdfs.extend(pdfs)

        return all_pdfs

    def add_pdf_file(self, file_path: Union[str, Path]) -> bool:
        """
        手动添加单个PDF文件

        Args:
            file_path: PDF文件路径

        Returns:
            是否成功添加
        """
        file_path = Path(file_path)

        if not file_path.exists():
            print(f"错误: 文件 '{file_path}' 不存在")
            return False

        if not file_path.is_file():
            print(f"错误: '{file_path}' 不是文件")
            return False

        if file_path.suffix.lower() != '.pdf':
            print(f"错误: '{file_path}' 不是PDF文件")
            return False

        abs_path = str(file_path.resolve())
        if abs_path not in self.pdf_files:
            self.pdf_files.append(abs_path)
            print(f"添加PDF: {file_path}")
            return True
        else:
            print(f"PDF已存在: {file_path}")
            return False

    def generate_jsonl(self, output_file: str = None, use_relative_paths: bool = False, base_path: str = None) -> str:
        """
        生成JSONL配置文件

        Args:
            output_file: 输出文件路径，如果为None则使用初始化时的路径
            use_relative_paths: 是否使用相对路径
            base_path: 相对路径的基准目录

        Returns:
            生成的JSONL文件路径
        """
        if output_file is None:
            output_file = self.output_file

        if not self.pdf_files:
            print("警告: 没有找到任何PDF文件")
            return output_file

        # 验证和处理输出文件路径
        output_path = Path(output_file)

        # 如果输出路径是目录，自动添加默认文件名
        if output_path.exists() and output_path.is_dir():
            output_path = output_path / "pdf_list.jsonl"
            output_file = str(output_path)
            print(f"⚠️  输出路径是目录，自动生成文件名: {output_file}")
        elif output_path.suffix == "":
            # 如果没有扩展名，添加.jsonl
            output_path = output_path.with_suffix(".jsonl")
            output_file = str(output_path)
            print(f"⚠️  自动添加扩展名: {output_file}")

        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            for pdf_path in self.pdf_files:
                # 处理路径格式
                if use_relative_paths and base_path:
                    try:
                        # 计算相对路径
                        rel_path = os.path.relpath(pdf_path, base_path)
                        final_path = rel_path
                    except ValueError:
                        # 如果无法计算相对路径，使用绝对路径
                        final_path = pdf_path
                else:
                    final_path = pdf_path

                # 写入JSONL格式
                json_line = {"raw_content": final_path}
                f.write(json.dumps(json_line, ensure_ascii=False) + '\n')

        print(f"✅ 成功生成JSONL文件: {output_file}")
        print(f"📄 共包含 {len(self.pdf_files)} 个PDF文件")
        return output_file

    def preview_results(self, max_items: int = 10):
        """预览检测结果"""
        if not self.pdf_files:
            print("没有找到任何PDF文件")
            return

        print(f"\n📋 检测到 {len(self.pdf_files)} 个PDF文件:")
        print("-" * 50)

        for i, pdf_path in enumerate(self.pdf_files[:max_items]):
            print(f"{i + 1:3d}. {pdf_path}")

        if len(self.pdf_files) > max_items:
            print(f"... 还有 {len(self.pdf_files) - max_items} 个文件")
        print("-" * 50)

    def clear_results(self):
        """清空检测结果"""
        self.pdf_files.clear()
        print("已清空检测结果")


def main():
    parser = argparse.ArgumentParser(description='检测PDF文件并生成JSONL配置文件')
    parser.add_argument('input_dir', nargs='?', default='../input', help='要扫描的输入目录路径 (默认: ../input)')
    parser.add_argument('-o', '--output', default='../input/pdf_list.jsonl', help='输出JSONL文件路径 (默认: ../input/pdf_list.jsonl)')  # 修改默认输出
    parser.add_argument('-r', '--recursive', action='store_true', default=True, help='递归扫描子目录')
    parser.add_argument('--no-recursive', action='store_false', dest='recursive', help='不递归扫描子目录')
    parser.add_argument('--relative', action='store_true', help='使用相对路径')
    parser.add_argument('--base-path', help='相对路径的基准目录')
    parser.add_argument('-p', '--preview', action='store_true', help='预览结果')

    args = parser.parse_args()

    # 验证输入目录
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"❌ 错误: 输入目录 '{args.input_dir}' 不存在")
        return

    if not input_path.is_dir():
        print(f"❌ 错误: '{args.input_dir}' 不是一个有效目录")
        return

    # 创建检测器
    detector = PDFDetector(args.output)

    # 使用指定的输入目录
    input_directory = args.input_dir

    # 扫描目录
    print(f"🔍 开始扫描目录: {input_directory}")
    print(f"📁 递归模式: {'开启' if args.recursive else '关闭'}")

    detector.scan_directory(input_directory, args.recursive)

    # 预览结果
    if args.preview:
        detector.preview_results()

    # 生成JSONL文件
    detector.generate_jsonl(
        output_file=args.output,
        use_relative_paths=args.relative,
        base_path=args.base_path or os.getcwd()
    )


if __name__ == "__main__":
    main()