import random
from tasks.science import SearchTask
from ToT.base import Node
from models.get_response import *
from ToT.bfs import BFS
from ToT.dfs import DFS
from utils.solution_summary_extractor import extract_summary_from_solution
from utils.verify_MATH import exact_match_score


class ToT_Task(SearchTask):
    def __init__(self, data, propose_method='glm', value_method='glm', algorithm='dfs', branch=3, select_branch=2,
                 max_depth=8, end_gate=0.9, select_method='greedy',
                 temperature=0.7, max_tokens=2048,
                 seed=170, max_length=2048, truncation=True,
                 do_sample=True, max_new_tokens=256, use_case_prompt=False, low=0, high=1, evaluate='', multiply_value=False, lang='zh', answer=None, verify_method='string'):
        super().__init__(data, propose_method, value_method)
        assert 0 <= low < high, "Inappropriate value range!"
        self.mode = 'tot'
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.max_length = max_length
        self.truncation = truncation
        self.do_sample = do_sample
        self.max_new_tokens = max_new_tokens
        self.algorithm = algorithm
        self.branch = branch
        self.select_branch = select_branch
        self.max_depth = max_depth
        self.use_case_prompt = use_case_prompt
        self.low = low
        self.high = high
        self.evaluate = evaluate
        self.select_method = select_method
        self.end_gate = end_gate
        self.node_count = 1
        self.multiply_value = multiply_value
        self.lang = lang
        self.answer = answer
        self.verify_method = verify_method

    def update_count(self):
        self.node_count += 1

    def clear_cache(self):
        self.value_cache = {}
        self.node_count = 1

    def get_next_step(self, y, step_n):
        if self.use_case_prompt:
            prompt = self.single_propose_prompt_wrap(self.question, y, step_n)
        else:
            prompt = self.zero_single_propose_wrap(self.question, y, step_n, self.lang)

        response = get_proposal(prompt, self.propose_method, self.temperature, self.max_tokens, self.seed,
                                self.max_length,
                                self.truncation, self.do_sample, self.max_new_tokens)
        if not response:
            print('Failed to get next step！\n')
            return ''

        if len(response) > 5:
            response = response[:5]

        p = ''
        for _ in response:
            p = p + _ + ' '
        p = p.strip()

        if self.lang == 'zh':
            if '下一步:' in p:
                stp = p.split('下一步:')[1].strip()
                if len(stp) < 2:
                    print('The output step is too short!\n')
                    return ''
                if stp in y:
                    print('Output step repeat!\n')
                    return ''

                revised_ = '步骤' + str(step_n) + ':' + stp
                print(f'New steps after standardization:{revised_}\n')
                return revised_ + '\n'

            elif '步骤' in p and ':' in p:
                pre_len = len(p.split(':')[0])
                p_ = p[pre_len:]
                p_ = p_.split('步骤')[0].strip()
                if len(p_) < 3:
                    print('The output step is too short！\n')
                    return ''
                if p_[1:] in y:
                    print('Output step repeat!\n')
                    return ''

                revised_ = '步骤' + str(step_n) + p_
                print(f'New steps after standardization:{revised_}\n')
                return revised_ + '\n'

            else:
                print('Incorrect output format!\n')
                return ''
        else:
            if "Next step:" in p:
                stp = p.split('Next step:')[1].strip()
                if len(stp) < 2:
                    print('The output step is too short！\n')
                    return ''
                if stp in y:
                    print('Output step repeat!\n')
                    return ''

                revised_ = 'Step ' + str(step_n) + ': ' + stp
                print(f'New steps after standardization:{revised_}\n')
                return revised_ + '\n'

            elif "Step" in p and ":" in p:
                pre_len = len(p.split(':')[0])
                p_ = p[pre_len:]
                p_ = p_.split('Step')[0].strip()
                if len(p_) < 4:
                    print('The output step is too short！\n')
                    return ''
                p_ = p_[1:].strip()
                if p_ in y:
                    print('Output step repeat!\n')
                    return ''

                revised_ = 'Step ' + str(step_n) + ': ' + p_
                print(f'New steps after standardization:{revised_}\n')
                return revised_ + '\n'

            else:
                p_ = p.strip()
                if len(p_) < 3:
                    print('The output step is too short！\n')
                    return ''
                if p_ in y:
                    print('Output step repeat!\n')
                    return ''

                revised_ = 'Step ' + str(step_n) + ': ' + p_
                print(f'New steps after standardization:{revised_}\n')
                return revised_ + '\n'

    def get_step_value(self, y):
        if y in self.value_cache.keys():
            return self.value_cache[y]

        if self.value_method == 'local':
            if self.lang == 'zh':
                prompt_answer = '问题:' + self.question + '\n步骤:\n' + '【答案】' + y
            else:
                prompt_answer = 'Problem: ' + self.question + '\nSolution:\n' + y

            value = get_value(prompt_answer, self.value_method, self.temperature, self.max_tokens, self.seed,
                              self.max_length, self.low, self.high)
            print(f'Get a score:{value}\n')
            self.value_cache.update({y: value})
            return value

        else:
            prompt = self.value_prompt_wrap(self.question, y)
            response = get_value(prompt, self.value_method, self.temperature, self.max_tokens, self.seed,
                                 self.max_length, self.low, self.high)
            value = self.value_outputs_unwrap(response, self.low, self.high)
            print(f'Get a score:{value}\n')
            self.value_cache.update({y: value})
            return value

    def get_summary(self, y):
        if self.lang == 'zh':
            if self.evaluate == 'scibench':
                prompt = self.evaluate_summary_prompt_wrap(self.question, y)
            elif self.evaluate == 'scieval':
                prompt = self.general_evaluate_summary_prompt_wrap(self.question, y)
            else:
                prompt = self.summary_prompt_wrap(self.question, y)

            response = get_proposal(prompt, self.propose_method, self.temperature, self.max_tokens, self.seed,
                                    self.max_length,
                                    self.truncation, self.do_sample, 128)

            if not response:
                print('Failed to get a summary!\n')
                return ''
            p = ''
            for _ in response:
                p = p + _ + ' '
            p = p.strip()

            if self.evaluate:
                if len(p) < 1:
                    print('Get the summary too short!\n')
                    return ''

                if '综上所述，最终答案是:' not in p:
                    summ = '综上所述，最终答案是:' + p
                    print(f'Get summary:{summ}\n')
                    return summ
                else:
                    summ = '综上所述，最终答案是:' + p.split('综上所述，最终答案是:')[-1]
                    print(f'Get summary:{summ}\n')
                    return summ

            else:
                if len(p) < 1:
                    print('Get the summary too short!\n')
                    return ''

                if '综上所述，' not in p:
                    summ = '综上所述，' + p
                    print(f'Get summary:{summ}\n')
                    return summ
                else:
                    summ = '综上所述，' + p.split('综上所述，')[-1]
                    print(f'Get summary:{summ}\n')
                    return summ

        else:
            prompt = self.MATH_summary_prompt_wrap(self.question, y)
            response = get_proposal(prompt, self.propose_method, self.temperature, self.max_tokens, self.seed,
                                    self.max_length,
                                    self.truncation, self.do_sample, 128)
            if not response:
                print('Failed to get a summary!\n')
                return ''
            p = ''
            for _ in response:
                p = p + _ + ' '
            summ = p.strip()

            print(f'Get summary:{summ}\n')
            return summ

    def run(self):
        self.clear_cache()
        if self.algorithm == 'dfs':
            solution, root, final_node = DFS(self)
        elif self.algorithm == 'bfs':
            solution, root, final_node = BFS(self)
        else:
            print('Unsupported algorithm!\n')
            return {}

        cnt = 5
        summary = ''
        while cnt:
            summary = self.get_summary(solution)
            if summary:
                break
            else:
                cnt -= 1
        if not summary and self.lang == 'en':
            summary = extract_summary_from_solution(solution)

        if self.evaluate == 'math' or self.verify_method == 'string':
            result = exact_match_score(summary, self.answer)
            final_answer = {'content': self.question, 'solution': solution, 'summary': summary, 'accurate': result, 'real_answer': self.answer}
        else:
            final_answer = {'content': self.question, 'solution': solution, 'summary': summary}

        if self.multiply_value:
            multiply_v = final_node.get_multiply_value()
            final_answer.update({'multiply_value': multiply_v})

        return final_answer, root
