import logging
import sys

# 建立獨立的 Logger
logger = logging.getLogger("RL_logger")
logger.setLevel(logging.DEBUG)

# 建立輸出到 stdout 的 Handler
stdout_handler = logging.StreamHandler(stream=sys.stdout)
formatter = logging.Formatter(f"%(levelname)s - %(message)s")
stdout_handler.setFormatter(formatter)

# 避免重複添加 Handler
if not logger.hasHandlers():
    logger.addHandler(stdout_handler)

# 防止訊息傳播到全域 logging
logger.propagate = False


def configure_logger(level=logging.INFO):
    """
    配置 Logger 的函数。
    :param level: 日志级别，默认为 INFO。
    """
    # 获取 Logger 实例
    logger = logging.getLogger("RL_logger")

    # 如果已经配置过 Handler，就不再重复配置
    if logger.hasHandlers():
        return

    # 设置日志级别
    logger.setLevel(level)

    # 创建输出到 stdout 的 Handler
    stdout_handler = logging.StreamHandler(stream=sys.stdout)

    # 设置日志格式
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stdout_handler.setFormatter(formatter)

    # 添加 Handler 到 Logger
    logger.addHandler(stdout_handler)

    # 防止日志传播到全局 Logger，避免重复输出
    logger.propagate = False

    return logger
