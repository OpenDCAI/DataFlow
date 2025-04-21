import re


def extract_summary_from_solution(solution: str):
    pattern = r"\\boxed\{(.*)\}"
    match = re.findall(pattern, solution)
    if match:
        summary = 'The final answer is ' + match[-1]
    elif '####' in solution:
        extracted = solution.split('####')[-1].strip()
        if len(extracted) > 1:
            if extracted[-1] == '.':
                extracted = extracted[:-1].strip()
            if len(extracted) > 1:
                if extracted[0] == ':':
                    extracted = extracted[1:].strip()
        summary = 'The final answer is ' + extracted
    elif 'The final answer is' in solution:
        extracted = solution.split('The final answer is')[-1].strip()
        if len(extracted) > 1:
            if extracted[-1] == '.':
                extracted = extracted[:-1].strip()
            if len(extracted) > 1:
                if extracted[0] == ':':
                    extracted = extracted[1:].strip()
        summary = 'The final answer is ' + extracted
    elif 'The answer is' in solution:
        extracted = solution.split('The answer is')[-1].strip()
        if len(extracted) > 1:
            if extracted[-1] == '.':
                extracted = extracted[:-1].strip()
            if len(extracted) > 1:
                if extracted[0] == ':':
                    extracted = extracted[1:].strip()
        summary = 'The final answer is ' + extracted
    elif 'final answer is' in solution:
        extracted = solution.split('final answer is')[-1].strip()
        if len(extracted) > 1:
            if extracted[-1] == '.':
                extracted = extracted[:-1].strip()
            if len(extracted) > 1:
                if extracted[0] == ':':
                    extracted = extracted[1:].strip()
        summary = 'The final answer is ' + extracted
    elif 'answer is' in solution:
        extracted = solution.split('answer is')[-1].strip()
        if len(extracted) > 1:
            if extracted[-1] == '.':
                extracted = extracted[:-1].strip()
            if len(extracted) > 1:
                if extracted[0] == ':':
                    extracted = extracted[1:].strip()
        summary = 'The final answer is ' + extracted
    else:
        summary = ''
    print('** Extracted summary: ', summary, '\n')
    return summary
