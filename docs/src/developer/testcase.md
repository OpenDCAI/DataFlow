## Testcase

testcase的写法可以参考CodePipeline/test/ast_checker_test.py
```python
import unittest
import tempfile
import os
import sys
import json
sys.path.append("..")
sys.path.append(".")
sys.path.append("../..")
from dataflow.utils.utils import get_generator

class TreeSitterParserTest(unittest.TestCase):
    
    def setUp(self):
        self.input_file = "/root/workspace/culfjk4p420c73amv510/herunming/DataFlow/CodePipeline/data/ast_data.jsonl"
        self.output_file = tempfile.NamedTemporaryFile(delete=False, mode='w+', suffix='.jsonl')
        
    def tearDown(self):
        os.unlink(self.output_file.name)
        
    def test_run(self):
        config = {
            "input_file": self.input_file,
            "output_file": self.output_file.name,
            "input_key": "content",
            "output_key": "",
            "vllm_used": False
        }
        generator = get_generator("TreeSitterParser", config)
        generator.run()
        with open(self.output_file.name, "r") as f:
            data = [json.loads(_) for _ in f]
        for i in range(5):
            self.assertEqual(data[i]['ast_error'], 0, f"Not Equal to 0 in data {i}")
        for i in range(5, 10):
            self.assertEqual(data[i]['ast_error'], 1, f"Not Equal to 1 in data {i}")
        
if __name__ == "__main__":
    unittest.main()
```
其中setUp是启动test前的准备工作，tearDown是后处理函数，test_run是运行的主函数，unittest.Testcase类内置如assertEqual等函数