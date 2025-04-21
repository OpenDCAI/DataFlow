import re


def get_consistency_output_scibench(outputs):
    output_count = {}
    for output in outputs:
        summ = output['summary'].strip()
        try:
            match = re.findall(r'[^^{.\-0123456789]-?[0-9]+\.?[0-9]*[^^}.0123456789]', summ)[-1][1:][:-1]
            model_ans = float(match)

        except Exception as e:
            try:
                match = re.findall(r'-?[0-9]+\.?[0-9]*', summ)[-1]
                model_ans = float(match)
            except Exception as e:
                print(f'Extract the answer error! Error type:{e}\n')
                continue

        if model_ans not in output_count.keys():
            output_count.update({model_ans: [1, output]})
        else:
            output_count[model_ans][0] += 1

    if not output_count:
        return outputs[0]

    most_cons_count = 0
    most_cons_output = {}
    for ans, info in output_count.items():
        if info[0] > most_cons_count:
            most_cons_count = info[0]
            most_cons_output = info[1]
    return most_cons_output


def get_consistency_output_scieval(outputs, q_type):
    output_count = {}
    for output in outputs:
        summ = output['summary'].strip()
        if q_type == "multiple-choice":
            try:
                model_ans = re.findall(r'[A-E]', summ)[0]
            except Exception as e:
                print(f"Extract the answer error! Error type:{e}\n")
                continue
        elif q_type == "judge":
            model_ans = summ
        elif q_type == "filling":
            model_ans = summ
        else:
            break

        if model_ans not in output_count.keys():
            output_count.update({model_ans: [1, output]})
        else:
            output_count[model_ans][0] += 1

    if not output_count:
        return outputs[0]

    most_cons_count = 0
    most_cons_output = {}
    for ans, info in output_count.items():
        if info[0] > most_cons_count:
            most_cons_count = info[0]
            most_cons_output = info[1]
    return most_cons_output
