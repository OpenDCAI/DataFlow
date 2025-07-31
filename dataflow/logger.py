import os
import logging
import colorlog

# 读取logging level
default_log_level = os.getenv("DF_LOGGING_LEVEL", "INFO")

# 自定义日志等级
SUCCESS_LEVEL_NUM = 25
logging.addLevelName(SUCCESS_LEVEL_NUM, "SUCCESS")

def success(self, message, *args, **kwargs):
    if self.isEnabledFor(SUCCESS_LEVEL_NUM):
        self._log(SUCCESS_LEVEL_NUM, message, args, **kwargs)

logging.Logger.success = success  # 添加方法到 Logger 类

def get_logger(level=default_log_level) -> logging.Logger:
    # 创建logger对象
    logger = logging.getLogger("DataFlow")
    if not logger.handlers:
        logger.setLevel(level)
        # 创建控制台日志处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        # 定义颜色输出格式
        color_formatter = colorlog.ColoredFormatter(
            '%(log_color)s %(asctime)s | %(filename)-20s- %(module)-20s- %(funcName)-20s- %(lineno)5d - %(name)-10s | %(levelname)8s | Processno %(process)5d - Threadno %(thread)-15d : %(message)s', 
            log_colors={
                'DEBUG': 'cyan',
                # 'INFO': 'white',
                'SUCCESS': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
        # 将颜色输出格式添加到控制台日志处理器
        console_handler.setFormatter(color_formatter)
        # 将控制台日志处理器添加到logger对象
        logger.addHandler(console_handler)
    return logger