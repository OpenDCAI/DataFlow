import os
import pathlib
import threading
import pandas as pd
import argparse
from concurrent.futures import ThreadPoolExecutor
from MCTS.task import MCTS_Task
from utils.json_operator import load_file, dump_json
import copy
import time


class MCTSRunner:
    def __init__(self, config, args=None):
        self.args = args
        self.file_lock = threading.Lock()
        self.output_list = []
        self.start_time = time.time()
        self.input_file = self.config.get("input_file")
        self.output_file = self.config.get("output_file")

    def _process_task(self, i, data_list):
        data_list = pd.read_json(self.input_file, lines=True)
        print(f'Begin to solve the problem {i + 1}...\n')
        data = data_list[i]['question']
        answer = data_list[i]['real_answer']

        Task = MCTS_Task(
            data, self.args.propose_method, self.args.value_method, self.args.branch, self.args.end_gate,
            self.args.roll_policy, self.args.roll_branch, self.args.roll_forward_steps, self.args.time_limit,
            self.args.iteration_limit, self.args.exploration_constant, self.args.alpha, self.args.inf,
            self.args.temperature, use_case_prompt=self.args.use_case_prompt, use_reflection=self.args.use_reflection,
            low=self.args.low, high=self.args.high, evaluate=self.args.evaluate, answer=answer, lang='en'
        )

        output, root = Task.run()
        print(f'The solution to problem {i + 1} is complete.\n')

        base_dir = os.getcwd()
        output_dir = pathlib.Path(f'{base_dir}/outputs/{self.args.task_name}/{self.args.file}/{Task.mode}')
        # output_file = f'{output_dir}/{Task.propose_method}_{Task.value_method}_{self.args.save_name}.json'
        output_file = self.output_file
        data_item = copy.deepcopy(data_list[i])
        data_item['mcts_output'] = output

        with self.file_lock:
            pathlib.Path.mkdir(output_dir, exist_ok=True, parents=True)
            self.output_list.append(data_item)
            dump_json(output_file, self.output_list)

    def run(self):
        print('-' * 30, 'Begin testing', '-' * 30, '\n')
        file = self.args.load_file_path
        print('** file_path: ', file)

        try:
            data_list = load_file(file)
            data_len = len(data_list)
        except Exception as e:
            print(f'File must be standardized json!\nError type:{e}\n')
            return

        assert data_len > 0, "Data list is empty!\n"

        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(self._process_task, i, data_list) for i in range(data_len)]
            for future in futures:
                future.result()

        print('_' * 60)
        print(f'Total number of questions: {data_len}\n')
        print('_' * 60)

        elapsed_time = time.time() - self.start_time
        print(f"程序运行时间: {elapsed_time:.2f} 秒")




def parse_args():
    base_args = argparse.ArgumentParser()
    base_args.add_argument('--load_file_path', type=str, default='scibench')
    base_args.add_argument('--task_name', type=str, default='scibench')
    base_args.add_argument('--file', type=str, default='thermo_standardized')  # json
    base_args.add_argument('--save_name', type=str, default='test')  # json
    base_args.add_argument('--propose_method', type=str, choices=['gpt', 'glm', 'llama', 'local'], default='glm')
    base_args.add_argument('--value_method', type=str, choices=['gpt', 'glm', 'local'], default='local')
    base_args.add_argument('--mode', type=str, choices=['cot', 'tot', 'mcts'], default='tot')
    base_args.add_argument('--temperature', type=float, default=0.7)
    base_args.add_argument('--time_limit', type=int, default=None)
    base_args.add_argument('--iteration_limit', type=int, default=100)
    base_args.add_argument('--roll_policy', type=str, choices=['random', 'greedy'], default='greedy')
    base_args.add_argument('--exploration_constant', type=float, default=0.4)
    base_args.add_argument('--roll_forward_steps', type=int, default=2)
    base_args.add_argument('--end_gate', type=float, default=0.9)  # End threshold
    base_args.add_argument('--branch', type=int, default=3)
    base_args.add_argument('--roll_branch', type=int, default=1)
    base_args.add_argument('--inf', type=float, default=0.8)
    base_args.add_argument('--evaluate', type=str, default='scibench')  # Whether to evaluate (empty means no evaluation)
    base_args.add_argument('--alpha', type=float, default=0.5)
    base_args.add_argument('--visualize', type=bool, default=False)  # visualization
    base_args.add_argument('--use_case_prompt', type=bool, default=False)  # Use sample prompts
    base_args.add_argument('--use_reflection', type=str, choices=['simple', 'common'], default='simple')  # Use reflective mode
    base_args.add_argument('--low', type=float, default=0)
    base_args.add_argument('--high', type=float, default=1)
    base_args.add_argument('--algorithm', type=str, choices=['dfs', 'bfs'], default='dfs')
    base_args.add_argument('--select_branch', type=int, default=2)
    base_args.add_argument('--max_depth', type=int, default=8)
    base_args.add_argument('--select_method', type=str, choices=['greedy', 'sample'], default='greedy')
    base_args.add_argument('--consistency', type=bool, default=True)

    arguments = base_args.parse_args()
    return arguments


if __name__ == '__main__':
    args = parse_args()
    runner = MCTSRunner(args=args)
    runner.run()
