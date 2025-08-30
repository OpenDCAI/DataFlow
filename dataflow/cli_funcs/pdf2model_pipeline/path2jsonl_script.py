import os
import json
import argparse
from pathlib import Path
from typing import List, Union


class PDFDetector:
    """PDF file detector for scanning directories and generating JSONL config files"""

    def __init__(self, output_file: str = "./.cache/gpu/pdf_list.jsonl"):
        # Handle output path - based on caller's working directory
        output_path = Path(output_file)
        if not output_path.is_absolute():
            caller_cwd = Path(os.environ.get('PWD', os.getcwd()))
            output_file = str(caller_cwd / output_file)
        self.output_file = output_file
        self.pdf_files = []

    def scan_directory(self, directory: Union[str, Path], recursive: bool = True) -> List[str]:
        """
        Scan PDF files in directory

        Args:
            directory: Directory path to scan
            recursive: Whether to scan subdirectories recursively

        Returns:
            List of found PDF file paths
        """
        directory = Path(directory)

        if not directory.exists():
            print(f"Error: Directory '{directory}' does not exist")
            return []

        if not directory.is_dir():
            print(f"Error: '{directory}' is not a valid directory")
            return []

        pdf_files = []

        # Directories to exclude from scanning
        exclude_dirs = {'.cache', '__pycache__', '.git', 'node_modules', '.venv', 'venv', '.env'}

        if recursive:
            # Recursively search all subdirectories
            pattern = "**/*.pdf"
        else:
            # Only search current directory
            pattern = "*.pdf"

        for pdf_path in directory.glob(pattern):
            # Skip if path contains any excluded directory
            if any(exclude_dir in pdf_path.parts for exclude_dir in exclude_dirs):
                continue

            # Also skip hidden directories (starting with .)
            if any(part.startswith('.') and part != '.' for part in pdf_path.parts):
                continue

            if pdf_path.is_file():
                # Convert to absolute path
                pdf_files.append(str(pdf_path.resolve()))
                print(f"Found PDF: {pdf_path}")

        self.pdf_files.extend(pdf_files)
        return pdf_files

    def scan_multiple_directories(self, directories: List[Union[str, Path]], recursive: bool = True) -> List[str]:
        """
        Scan multiple directories

        Args:
            directories: List of directory paths
            recursive: Whether to scan recursively

        Returns:
            List of all found PDF file paths
        """
        all_pdfs = []
        for directory in directories:
            pdfs = self.scan_directory(directory, recursive)
            all_pdfs.extend(pdfs)

        return all_pdfs

    def add_pdf_file(self, file_path: Union[str, Path]) -> bool:
        """
        Manually add a single PDF file

        Args:
            file_path: PDF file path

        Returns:
            Whether successfully added
        """
        file_path = Path(file_path)

        if not file_path.exists():
            print(f"Error: File '{file_path}' does not exist")
            return False

        if not file_path.is_file():
            print(f"Error: '{file_path}' is not a file")
            return False

        if file_path.suffix.lower() != '.pdf':
            print(f"Error: '{file_path}' is not a PDF file")
            return False

        abs_path = str(file_path.resolve())
        if abs_path not in self.pdf_files:
            self.pdf_files.append(abs_path)
            print(f"Added PDF: {file_path}")
            return True
        else:
            print(f"PDF already exists: {file_path}")
            return False

    def generate_jsonl(self, output_file: str = None) -> str:
        """
        Generate JSONL config file

        Args:
            output_file: Output file path, if None use the initialized path

        Returns:
            Generated JSONL file path
        """
        if output_file is None:
            output_file = self.output_file
        else:
            # Handle output file relative path - based on caller's working directory
            output_path = Path(output_file)
            if not output_path.is_absolute():
                caller_cwd = Path(os.environ.get('PWD', os.getcwd()))
                output_path = caller_cwd / output_path
                output_file = str(output_path)

        if not self.pdf_files:
            print("Warning: No PDF files found")
            return output_file

        # Validate and process output file path
        output_path = Path(output_file)

        # If output path is directory, auto-generate filename
        if output_path.exists() and output_path.is_dir():
            output_path = output_path / "pdf_list.jsonl"
            output_file = str(output_path)
            print(f"Warning: Output path is directory, auto-generating filename: {output_file}")
        elif output_path.suffix == "":
            # If no extension, add .jsonl
            output_path = output_path.with_suffix(".jsonl")
            output_file = str(output_path)
            print(f"Warning: Auto-adding extension: {output_file}")

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            for pdf_path in self.pdf_files:
                # Write in JSONL format
                json_line = {"raw_content": pdf_path}
                f.write(json.dumps(json_line, ensure_ascii=False) + '\n')

        print(f"Successfully generated JSONL file: {output_file}")
        print(f"Contains {len(self.pdf_files)} PDF files")
        return output_file

    def preview_results(self, max_items: int = 10):
        """Preview detection results"""
        if not self.pdf_files:
            print("No PDF files found")
            return

        print(f"\nDetected {len(self.pdf_files)} PDF files:")
        print("-" * 50)

        for i, pdf_path in enumerate(self.pdf_files[:max_items]):
            print(f"{i + 1:3d}. {pdf_path}")

        if len(self.pdf_files) > max_items:
            print(f"... and {len(self.pdf_files) - max_items} more files")
        print("-" * 50)

    def clear_results(self):
        """Clear detection results"""
        self.pdf_files.clear()
        print("Detection results cleared")


def main():
    parser = argparse.ArgumentParser(description='Detect PDF files and generate JSONL config file')
    parser.add_argument('input_dir', nargs='?', default='./input',
                        help='Input directory path to scan (default: ./input)')
    parser.add_argument('-o', '--output', default='./.cache/gpu/pdf_list.jsonl',
                        help='Output JSONL file path (default: ./.cache/gpu/pdf_list.jsonl)')
    parser.add_argument('-r', '--recursive', action='store_true', default=True, help='Scan subdirectories recursively')
    parser.add_argument('--no-recursive', action='store_false', dest='recursive', help='Do not scan subdirectories')

    args = parser.parse_args()

    # Validate input directory - handle relative paths
    input_path = Path(args.input_dir)
    if not input_path.is_absolute():
        # If relative path, resolve based on caller's working directory
        caller_cwd = Path(os.environ.get('PWD', os.getcwd()))
        input_path = caller_cwd / input_path

    if not input_path.exists():
        print(f"Error: Input directory '{input_path}' does not exist")
        return

    if not input_path.is_dir():
        print(f"Error: '{input_path}' is not a valid directory")
        return

    # Create detector
    detector = PDFDetector(args.output)

    # Use resolved input directory
    input_directory = str(input_path)

    # Scan directory
    print(f"Starting directory scan: {input_directory}")
    print(f"Recursive mode: {'enabled' if args.recursive else 'disabled'}")

    detector.scan_directory(input_directory, args.recursive)

    # Generate JSONL file
    detector.generate_jsonl(output_file=args.output)


if __name__ == "__main__":
    main()