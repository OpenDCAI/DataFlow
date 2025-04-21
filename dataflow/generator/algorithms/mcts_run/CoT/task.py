import re
from tasks.science import SearchTask
from models.get_response import *
from utils.verify_MATH import exact_match_score
from utils.solution_summary_extractor import extract_summary_from_solution


class CoT_Task(SearchTask):
    def __init__(self, data, propose_method='glm', value_method='glm', temperature=0.7, max_tokens=2048, seed=170,
                 max_length=2048, truncation=True,
                 do_sample=True, max_new_tokens=1024, evaluate='', summary=False, lang='zh', answer=None,
                 verify_method='string', do_self_critic=False):
        super().__init__(data, propose_method, value_method)
        self.mode = 'cot'
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.max_length = max_length
        self.truncation = truncation
        self.do_sample = do_sample
        self.max_new_tokens = max_new_tokens
        self.evaluate = evaluate
        self.summary = summary
        self.lang = lang
        self.answer = answer
        self.verify_method = verify_method
        self.do_self_critic = do_self_critic

    def get_summary(self, solution: str):
        if self.lang == 'zh':
            if not self.summary:
                if "综上所述，" in solution:
                    summ = solution.split("综上所述，")[-1]
                    return "综上所述，" + summ
                elif '。' in solution:
                    summ = solution.split("。")[-2]
                    return "综上所述，" + summ + '。'
                else:
                    return ''
            else:
                if self.evaluate == 'scibench':
                    prompt = self.evaluate_summary_prompt_wrap(self.question, solution)
                elif self.evaluate == 'scieval':
                    prompt = self.general_evaluate_summary_prompt_wrap(self.question, solution)
                else:
                    prompt = self.summary_prompt_wrap(self.question, solution)

                response = get_proposal(prompt, self.propose_method, self.temperature, self.max_tokens, self.seed,
                                        self.max_length,
                                        self.truncation, self.do_sample, 128)

                if not response:
                    print('Get summary fail！\n')
                    return ''
                p = ''
                for _ in response:
                    p = p + _ + '\n'
                p = p.strip()

                if self.evaluate:
                    if len(p) < 1:
                        print('Get summary too short！\n')
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
                        print('Get summary too short！\n')
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
            if "Summary:" in solution:
                summ = solution.split("Summary:")[-1].strip()
            else:
                summ = ''
            return summ

    def get_MATH_summary(self, solution):
        prompt = self.MATH_summary_prompt_wrap(self.question, solution)
        response = get_proposal(prompt, self.propose_method, self.temperature, self.max_tokens, self.seed,
                                self.max_length,
                                self.truncation, self.do_sample, 128)
        if not response:
            print('Get summary fail!\n')
            return ''
        p = ''
        for _ in response:
            p = p + _ + '\n'
        p = p.strip()

        print(f'Get summary:{p}\n')
        return p

    def get_self_critic(self, solution):
        critic_prompt = self.self_critic_prompt_wrap(self.question, solution)
        output_score = get_proposal(critic_prompt, self.propose_method, self.temperature, self.max_tokens, self.seed,
                                    self.max_length, self.truncation, self.do_sample, 128)
        score_strs = ''
        for out in output_score:
            score_strs = score_strs + out + '\n'

        pattern = r'[0-9]+\.?[0-9]*'
        match = re.findall(pattern, score_strs)
        if not match:
            return None
        else:
            s = min(float(match[-1]), 1.0)
            s = max(s, 0)
            return s

    def run(self):
        self.clear_cache()
        if self.evaluate == 'math' or self.verify_method == 'string':
            prompt = self.cot_prompt_wrap(self.question, self.lang, True)
        else:
            prompt = self.cot_prompt_wrap(self.question, self.lang)
        out = get_proposal(prompt, self.propose_method, temperature=self.temperature,
                           max_tokens=self.max_tokens,
                           seed=self.seed, max_length=self.max_length, truncation=self.truncation,
                           do_sample=self.do_sample, max_new_tokens=self.max_new_tokens)
        solution = ''
        for _ in out:
            solution = solution + _ + '\n'
        solution = solution.strip()
        print(f'Get answers:{solution}\n')

        if self.evaluate == 'math' or self.verify_method == 'string':
            cnt = 5
            summary = ''
            while cnt and not summary:
                summary = self.get_MATH_summary(solution)
                cnt -= 1

            if not summary:
                summary = extract_summary_from_solution(solution)

            result = exact_match_score(summary, self.answer)
            output = {'content': self.question, 'solution': solution, 'summary': summary, 'accurate': result,
                      'real_answer': self.answer}

        else:
            cnt = 5
            summary = ''
            while cnt:
                summary = self.get_summary(solution)
                if summary:
                    break
                else:
                    cnt -= 1

            output = {'content': self.question, 'solution': solution, 'summary': summary}

        if self.do_self_critic:
            score = None
            cnt = 3
            while score is None and cnt:
                score = self.get_self_critic(solution)
                cnt -= 1
            if score is None:
                score = 0
            output.update({'self_critic': score})

        return output
