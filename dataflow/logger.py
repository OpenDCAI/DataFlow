import os
import sys
import logging
import colorlog

# ---------- 1) 自定义 SUCCESS 等级 ----------
SUCCESS_LEVEL_NUM = 25
logging.addLevelName(SUCCESS_LEVEL_NUM, "SUCCESS")

def success(self, message, *args, **kwargs):
    if self.isEnabledFor(SUCCESS_LEVEL_NUM):
        self._log(SUCCESS_LEVEL_NUM, message, args, **kwargs)

logging.Logger.success = success

# ---------- 2) stdout / stderr 分流的过滤器 ----------
class MaxLevelFilter(logging.Filter):
    """仅放行 level < max_level 的日志（用于 stdout）。"""
    def __init__(self, max_level):
        super().__init__()
        self.max_level = max_level
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno < self.max_level

class MinLevelFilter(logging.Filter):
    """仅放行 level >= min_level 的日志（用于 stderr）。"""
    def __init__(self, min_level):
        super().__init__()
        self.min_level = min_level
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= self.min_level

# ---------- 3) 贴近 loguru 的 ColoredFormatter ----------
def _make_colored_formatter():
    # 颜色映射（loguru 的感觉：DEBUG/INFO 偏冷色，WARNING 黄，ERROR/CRITICAL 红，SUCCESS 绿）
    log_colors = {
        "DEBUG":    "blue",
        "INFO":     "white",
        "SUCCESS":  "green",
        "WARNING":  "yellow",
        "ERROR":    "red",
        "CRITICAL": "red,bg_white",
    }
    # 二级着色：让 levelname 与 message 按级别着色；name/func/line 用青色
    secondary = {
        "levelname": log_colors,
        "message":   {
            "DEBUG":    "blue",
            "INFO":     "white",
            "SUCCESS":  "green",
            "WARNING":  "yellow",
            "ERROR":    "red",
            "CRITICAL": "red",
        },
        # 这些是固定青色，模拟 loguru 的 <cyan> 标签
        "name":      {k: "cyan" for k in log_colors},
        "funcName":  {k: "cyan" for k in log_colors},
        "lineno":    {k: "cyan" for k in log_colors},
        "asctime":   {k: "green" for k in log_colors},  # 时间绿色
    }


    return colorlog.ColoredFormatter(
        fmt=(
            "%(asctime_log_color)s%(asctime)s.%(msecs)03d%(reset)s"
            " | %(levelname_log_color)s%(levelname)-8s%(reset)s"
            " | %(name_log_color)s%(name)s%(reset)s"
            ":%(funcName_log_color)s%(filename)s%(reset)s"
            ":%(funcName_log_color)s%(funcName)s%(reset)s"
            ":%(lineno_log_color)s%(lineno)d%(reset)s"
            " - %(message_log_color)s%(message)s%(reset)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors=log_colors,
        secondary_log_colors=secondary,
        style="%",
    )

# ---------- 4) 获取 logger（贴近 loguru 的 add 行为） ----------
def get_logger(level: str = None) -> logging.Logger:
    """返回一个名为 'DataFlow' 的 logger：
    - 控制台输出分流：<ERROR 到 stdout；>=ERROR 到 stderr
    - 颜色与格式尽量贴近 loguru 默认
    - 避免重复添加 handler
    """
    if level is None:
        level = os.getenv("DF_LOGGING_LEVEL", "INFO")

    logger = logging.getLogger("DataFlow")
    logger.setLevel(level)
    logger.propagate = False  # 避免向 root 传播造成重复输出

    if logger.handlers:
        # 已初始化过，直接返回（可根据需要改成更新 handler 的级别/格式）
        return logger

    colored_fmt = _make_colored_formatter()

    # stdout：DEBUG/INFO/SUCCESS/WARNING
    h_out = logging.StreamHandler(stream=sys.stdout)
    h_out.setLevel(level)
    h_out.addFilter(MaxLevelFilter(logging.ERROR))
    h_out.setFormatter(colored_fmt)

    # stderr：ERROR/CRITICAL
    h_err = logging.StreamHandler(stream=sys.stderr)
    h_err.setLevel(level)
    h_err.addFilter(MinLevelFilter(logging.ERROR))
    h_err.setFormatter(colored_fmt)

    logger.addHandler(h_out)
    logger.addHandler(h_err)
    return logger
