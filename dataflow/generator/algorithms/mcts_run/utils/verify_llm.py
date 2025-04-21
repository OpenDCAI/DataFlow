from models.get_response import *


def llm_verify(ans, real_ans, judge_model='gpt-4-1106-preview'):
    prompt = '下面将输入两段文字，第一段文字为某道理科题目的一个解答或答案（不一定正确），第二段是这道题目的标准答案。请判断第一段解答得到的答案与标准答案在数学意义上是否一致，并根据判断直接输出‘0’或’1‘，不需要输出任何别的信息。如果答案一致，请输出‘1’；否则，只要答案不匹配，或者第一个文段中没有明确指出答案也没有输出latex表达式，请输出‘0’；如果第一段解答与标准答案之间关系模糊，请输出‘0’。\n'
    qry = prompt + '文段1:' + ans + '\n' + '文段2:' + real_ans + '\n输出:'
    lbl = ''
    cnt = 5
    while lbl == '' and cnt:
        out = ''
        try:
            chat_comp = openai.ChatCompletion.create(model=judge_model, messages=[{"role": "user", "content": qry}])
            out = chat_comp.choices[0].message.content[0]
        except Exception as e:
            print(f'Error:{e}\n')
        if out == '0' or out == '1':
            lbl = out
        else:
            cnt -= 1
    if not cnt:
        return 0
    return int(lbl)
