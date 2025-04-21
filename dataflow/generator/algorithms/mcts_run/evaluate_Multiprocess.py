import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import pathlib
from MCTS.task import MCTS_Task
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor
from utils.visualize import visualize
from utils.json_operator import *
from utils.verify_answer import *
from utils.self_consistency import get_consistency_output_scibench
import requests
import random
from datasets import load_dataset
import copy
from tqdm import tqdm
import time

start_time = time.time()
# 创建一个锁对象，防止多个线程同时写文件
file_lock = threading.Lock()



def process_task(i, data_list, arguments, output_list):
    print(f'Begin to solve the problem {i+1}...\n')
    data = data_list[i]['question']
    answer = data_list[i]['real_answer']

    Task = MCTS_Task(data, arguments.propose_method, arguments.value_method, arguments.branch, arguments.end_gate,
                        arguments.roll_policy, arguments.roll_branch, arguments.roll_forward_steps, arguments.time_limit,
                        arguments.iteration_limit, arguments.exploration_constant, arguments.alpha, arguments.inf,
                        arguments.temperature, use_case_prompt=arguments.use_case_prompt, use_reflection=arguments.use_reflection,
                        low=arguments.low, high=arguments.high, evaluate=arguments.evaluate, answer=answer, lang='en')
    
    output, root = Task.run()
    print(f'The solution to problem {i+1} is complete.\n')

    base_dir = os.getcwd()
    output_dir = pathlib.Path(f'{base_dir}/outputs/{arguments.task_name}/{arguments.file}/{Task.mode}')
    
    # 修改文件名，加入任务索引，确保每个任务有独立文件
    output_file = f'{base_dir}/outputs/{arguments.task_name}/{arguments.file}/{Task.mode}/{Task.propose_method}_{Task.value_method}_{arguments.save_name}.json'
    
    data_item = copy.deepcopy(data_list[i])  # 创建深拷贝
    data_item['mcts_output'] = output

    output_list.append(data_item)
    
    # 使用锁确保文件写入时不会发生竞争
    with file_lock:
        pathlib.Path.mkdir(output_dir, exist_ok=True, parents=True)
        dump_json(output_file, output_list)
    
    



def run(arguments):
    print('-'*30, 'Begin testing', '-'*30, '\n')
    file = arguments.load_file_path
    print('** file_path: ', file)

    try:
        data_list = load_file(file)
        data_len = len(data_list)
    except Exception as e:
        print(f'File must be standardized json!\nError type:{e}\n')
        return

    assert data_len > 0, "Data list is empty!\n"
    
    output_list = []
    correct_count = 0

    # 使用线程池来处理每个任务
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(process_task, i, data_list, arguments, output_list) for i in range(data_len)]
        
        # 等待所有线程执行完成
        for future in futures:
            future.result()

    print('_' * 60)
    if arguments.evaluate:
        print(f'Test accuracy:{correct_count / data_len}\n')
        print(f'Correct number of problems:{correct_count}\nTotal number of questions:{data_len}\n')
    print('_' * 60)
    # 记录程序结束时间
    end_time = time.time()

    # 计算运行时间
    elapsed_time = end_time - start_time
    print(f"程序运行时间: {elapsed_time} 秒")




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
    run(args)
