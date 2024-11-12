import logging
import sys

# 建立獨立的 Logger
logger = logging.getLogger("RL_logger")
logger.setLevel(logging.INFO)

# 建立輸出到 stdout 的 Handler
stdout_handler = logging.StreamHandler(stream=sys.stdout)
formatter = logging.Formatter(f"%(levelname)s - %(message)s")
stdout_handler.setFormatter(formatter)

# 避免重複添加 Handler
if not logger.hasHandlers():
    logger.addHandler(stdout_handler)

# 防止訊息傳播到全域 logging
logger.propagate = False
