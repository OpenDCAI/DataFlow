import math
import re


# only support float answer verification
def verify_float(answer: float, output: str):
    if not output:
        print(f'The output is empty and cannot match the answer!\n')
        return False

    if '综上所述，' in output:
        spl_ans = output.split('综上所述，')[-1]
        spl_ans = spl_ans.strip()
    else:
        spl_ans = output.strip()

    try:
        match = re.findall(r'[^^{.\-0123456789]-?[0-9]+\.?[0-9]*[^^}.0123456789]', spl_ans)[-1][1:][:-1]
        model_ans = float(match)

        # standard (adjustable)
        if abs(answer) >= 1:
            result = math.isclose(model_ans, answer, abs_tol=0.1)
        else:
            result = math.isclose(model_ans, answer, rel_tol=0.1)

        print(f'The ans of model is:{model_ans}, while the ground truth is {answer}.\n')
        return result

    except Exception as e:
        try:
            match = re.findall(r'-?[0-9]+\.?[0-9]*', spl_ans)[-1]
            model_ans = float(match)

            # standard (adjustable)
            if abs(answer) >= 1:
                result = math.isclose(model_ans, answer, abs_tol=0.1)
            else:
                result = math.isclose(model_ans, answer, rel_tol=0.1)

            print(f'The ans of model is:{model_ans}, while the ground truth is {answer}.\n')
            return result
        except Exception as e:
            print(f'Result not matched, error type:{e}\n')
            print(f'The ans of model is:{spl_ans}, while the ground truth is {answer}.\n')
            return False


# only support choice answer verification
def verify_choice(answer: str, output: str):
    if not output:
        print(f'The output is empty and cannot match the answer!\n')
        return False

    check_list = ['A', 'B', 'C', 'D', 'E']

    if '综上所述，最终答案是:' in output:
        spl_ans = output.split('综上所述，最终答案是:')[-1]
        spl_ans = spl_ans.strip()
    elif '综上所述，' in output:
        spl_ans = output.split('综上所述，')[-1]
        spl_ans = spl_ans.strip()
    else:
        spl_ans = output.strip()

    # standard (adjustable)
    for choice in check_list:
        if choice in answer and choice not in spl_ans:
            print(f'The ans of model is:{spl_ans}, while the ground truth is {answer}.\n')
            return False
        if choice not in answer and choice in spl_ans:
            print(f'The ans of model is:{spl_ans}, while the ground truth is {answer}.\n')
            return False

    print(f'The ans of model is:{spl_ans}, while the ground truth is {answer}.\n')
    return True


# for scieval
def verify_scieval(answer, output, q_type):
    print(f'The ans of model is:"{output}", while the ground truth is {answer}.\n')
    if q_type == "multiple-choice":
        try:
            match = re.findall(r'[A-E]', output)[0]
        except Exception as e:
            print(f"Result not matched, error type:{e}\n")
            return False
        if answer.lower() == match.lower():
            return True
    elif q_type == "judge":
        if answer.lower() in output.lower():
            return True
    elif q_type == "filling":
        if answer.lower() in output.lower():
            return True
    else:
        print('Type error!\n')
        return False
    return False
