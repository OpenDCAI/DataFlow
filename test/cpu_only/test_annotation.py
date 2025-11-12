from dataflow.core.prompt import PromptABC, DIYPromptABC
from dataflow.prompts.chemistry import ExtractSmilesFromTextPrompt

if __name__ == "__main__":
    class MyClass:
        def my_method(self, pram = ExtractSmilesFromTextPrompt):
            print(pram)

    diy = DIYPromptABC()

    # MyClass().my_method("hi!")  # 输出: hi!
    my = MyClass()
    my.my_method()  # 输出: hi!
    my.my_method(diy)