from dataflow.core.prompt import PromptABC, DIYPromptABC, prompt_restrict
import pytest
import typing
from typing import get_type_hints, get_origin, get_args, Union
import inspect

# ==== 测试用到的 Prompt 类 ====

class APrompt(PromptABC):
    pass

class BPrompt(PromptABC):
    pass

class OtherPrompt(PromptABC):
    pass

class CustomDIY(DIYPromptABC):
    pass

# ==== 被装饰的“算子/类” ====

@prompt_restrict(APrompt, BPrompt)
class OperatorKwOnly:
    """只从 kwargs 读取 prompt_template 的被测类"""
    def __init__(self, prompt_template=None):
        self.prompt_template = prompt_template

@prompt_restrict(APrompt, BPrompt)
class OperatorWithPositional:
    """用于演示‘位置参数不被检查’的行为"""
    def __init__(self, prompt_template=None):
        self.prompt_template = prompt_template

# ==== 单元测试 ====
@pytest.mark.cpu
def test_accept_allowed_prompt_instances():
    """允许类型（APrompt/BPrompt）的实例应当被接受"""
    op1 = OperatorKwOnly(prompt_template=APrompt())
    op2 = OperatorKwOnly(prompt_template=BPrompt())
    assert isinstance(op1.prompt_template, APrompt)
    assert isinstance(op2.prompt_template, BPrompt)
@pytest.mark.cpu
def test_accept_diy_subclass_instance():
    """任意 DIYPromptABC 的子类实例也应当被接受"""
    op = OperatorKwOnly(prompt_template=CustomDIY())
    assert isinstance(op.prompt_template, CustomDIY)
@pytest.mark.cpu
def test_accept_none():
    """prompt_template=None（省略也等价）应当被接受"""
    op1 = OperatorKwOnly()
    op2 = OperatorKwOnly(prompt_template=None)
    assert op1.prompt_template is None
    assert op2.prompt_template is None
@pytest.mark.cpu
def test_reject_other_prompt_subclass():
    """不是白名单（APrompt/BPrompt）且不是 DIYPromptABC 子类的 PromptABC 子类应当被拒绝"""
    with pytest.raises(TypeError) as ei:
        OperatorKwOnly(prompt_template=OtherPrompt())
    msg = str(ei.value)
    # 友好错误信息：包含类名与白名单
    assert "[OperatorKwOnly] Invalid prompt_template type:" in msg
    assert "OtherPrompt" in msg
    assert "APrompt" in msg and "BPrompt" in msg
@pytest.mark.cpu
def test_reject_non_prompt_object():
    """非 Prompt 类型（例如普通 object 实例）应当被拒绝"""
    with pytest.raises(TypeError):
        OperatorKwOnly(prompt_template=object())
@pytest.mark.cpu
def test_annotations_updated_union():
    """
    __annotations__ / get_type_hints 应该把 prompt_template 暴露为
    Union[APrompt, BPrompt, DIYPromptABC, NoneType]
    """
    hints = get_type_hints(OperatorKwOnly)
    assert "prompt_template" in hints
    anno = hints["prompt_template"]
    origin = get_origin(anno)
    assert origin is typing.Union
    args = set(get_args(anno))
    # 期望四个构成要素
    assert APrompt in args
    assert BPrompt in args
    assert DIYPromptABC in args
    assert type(None) in args
@pytest.mark.cpu
def test_allowed_prompts_attribute():
    """装饰器应当在类上设置 ALLOWED_PROMPTS 为元组"""
    assert hasattr(OperatorKwOnly, "ALLOWED_PROMPTS")
    assert OperatorKwOnly.ALLOWED_PROMPTS == (APrompt, BPrompt)

@pytest.mark.cpu
def test_kwargs_only_behavior_no_check_on_positional():
    """
    当前实现只从 kwargs 获取 prompt_template。
    使用‘位置参数’传入时不会触发检查（这是一个已知行为/潜在坑）。
    下面这行不会抛异常：
    """
    # 尽管 OtherPrompt() 不在白名单里，但因为是位置参数，未被检查
    with pytest.raises(TypeError):
        op = OperatorWithPositional(OtherPrompt())


@pytest.mark.cpu
@pytest.mark.parametrize(
    "value,should_pass",
    [
        (APrompt(), True),
        (BPrompt(), True),
        (CustomDIY(), True),
        (OtherPrompt(), False),
        (object(), False),
    ],
)
def test_matrix_allowed_and_rejected(value, should_pass):
    """参数化覆盖更多输入组合"""
    if should_pass:
        OperatorKwOnly(prompt_template=value)  # 不应抛异常
    else:
        with pytest.raises(TypeError):
            OperatorKwOnly(prompt_template=value)


# ================== 不同签名形式的被测类（prompt_template 的位置不同） ==================

@prompt_restrict(APrompt, BPrompt)
class OpFirst:
    def __init__(self, prompt_template=None, x=0):
        self.prompt_template = prompt_template
        self.x = x

@prompt_restrict(APrompt, BPrompt)
class OpMiddle:
    def __init__(self, x, prompt_template, y=1):
        self.x = x
        self.prompt_template = prompt_template
        self.y = y

@prompt_restrict(APrompt, BPrompt)
class OpKWOnly:
    def __init__(self, *, prompt_template=None, x=0):
        self.prompt_template = prompt_template
        self.x = x

@prompt_restrict(APrompt, BPrompt)
class OpDefaultNone:
    def __init__(self, x=0, y=1, prompt_template=None):
        self.x, self.y = x, y
        self.prompt_template = prompt_template

@prompt_restrict(APrompt, BPrompt)
class OpNoPromptParam:
    def __init__(self, x, y=1):
        self.x, self.y = x, y  # 没有 prompt_template —— 不检查


# ================== 基础通过用例（位置 & 关键字 & None） ==================

@pytest.mark.parametrize("Cls", [OpFirst, OpMiddle, OpKWOnly, OpDefaultNone])
@pytest.mark.parametrize("value", [APrompt(), BPrompt(), CustomDIY(), None])
def test_allowed_values_positional_and_keyword(Cls, value):
    # 先测位置参数（若签名允许）
    sig = inspect.signature(Cls.__init__)
    if "prompt_template" in sig.parameters:
        p = list(sig.parameters).index("prompt_template")
        # 根据 prompt_template 的位置构造实参
        if Cls is OpFirst:
            obj = Cls(value, 123) if value is not None else Cls(None, 123)
        elif Cls is OpMiddle:
            obj = Cls(999, value) if value is not None else Cls(999, None)
        elif Cls is OpDefaultNone:
            # OpDefaultNone: (x=0, y=1, prompt_template=None) —— 允许位置传参
            obj = Cls(1, 2, value)
        else:
            obj = None  # OpKWOnly 不接受位置参数

        if obj is not None:
            assert getattr(obj, "prompt_template") is value

    # 再测关键字参数（所有都支持）
    obj2 = Cls(prompt_template=value) if value is not None else Cls(prompt_template=None)
    assert getattr(obj2, "prompt_template") is value

# ================== 非允许类型：位置 & 关键字都应抛错 ==================

@pytest.mark.parametrize("Cls, args_builder", [
    (OpFirst,      lambda bad: (bad, 0)),          # prompt_template 在首位
    (OpMiddle,     lambda bad: (0, bad)),          # 在中间
    (OpDefaultNone,lambda bad: (0, 1, bad)),       # 在末尾
])
@pytest.mark.parametrize("bad_value", [OtherPrompt(), object()])
def test_disallowed_values_positional_raise(Cls, args_builder, bad_value):
    with pytest.raises(TypeError) as ei:
        Cls(*args_builder(bad_value))
    msg = str(ei.value)
    assert "[{}] Invalid prompt_template type:".format(Cls.__name__) in msg
    assert "APrompt" in msg and "BPrompt" in msg

@pytest.mark.parametrize("Cls", [OpFirst, OpMiddle, OpKWOnly, OpDefaultNone])
@pytest.mark.parametrize("bad_value", [OtherPrompt(), object()])
def test_disallowed_values_keyword_raise(Cls, bad_value):
    with pytest.raises(TypeError):
        Cls(prompt_template=bad_value)

# ================== 位置 + 关键字重复传参：Python 自身错误 ==================

def test_multiple_values_for_argument_python_error():
    with pytest.raises(TypeError):
        OpFirst(APrompt(), prompt_template=BPrompt())  # 同一参数重复提供

# ================== 注解与 ALLOWED_PROMPTS ==================

@pytest.mark.parametrize("Cls", [OpFirst, OpMiddle, OpKWOnly, OpDefaultNone])
def test_annotations_and_allowed_prompts(Cls):
    hints = get_type_hints(Cls)
    assert "prompt_template" in hints
    anno = hints["prompt_template"]
    assert get_origin(anno) is Union
    args = set(get_args(anno))
    assert APrompt in args and BPrompt in args
    assert DIYPromptABC in args and type(None) in args

    assert hasattr(Cls, "ALLOWED_PROMPTS")
    assert Cls.ALLOWED_PROMPTS == (APrompt, BPrompt)

# ================== 继承类行为：应保持相同（位置也检查） ==================

class SubOpFirst(OpFirst): pass
class SubOpMiddle(OpMiddle): pass
class SubOpKWOnly(OpKWOnly): pass

@pytest.mark.parametrize("Cls, args_builder", [
    (SubOpFirst,  lambda bad: (bad, 0)),
    (SubOpMiddle, lambda bad: (0, bad)),
])
@pytest.mark.parametrize("bad_value", [OtherPrompt(), object()])
def test_subclass_positional_still_checked(Cls, args_builder, bad_value):
    with pytest.raises(TypeError):
        Cls(*args_builder(bad_value))

@pytest.mark.parametrize("Cls", [SubOpFirst, SubOpMiddle, SubOpKWOnly])
def test_subclass_keyword_still_checked(Cls):
    with pytest.raises(TypeError):
        Cls(prompt_template=OtherPrompt())

# ================== 边界：没有 prompt_template 参数的类 —— 不检查 ==================

def test_class_without_prompt_param_is_not_checked():
    # 即使传入奇怪对象也不会检查，因为签名不含 prompt_template
    obj = OpNoPromptParam(1, y=2)
    assert obj.x == 1 and obj.y == 2