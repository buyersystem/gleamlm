"""入口脚本的日志可见性配置 —— 库层 print → logging 迁移 (T4) 的配套。

设计约定:
  - gleamlm 库内一律 ``logger = logging.getLogger(__name__)``, 不配置 root;
    库根 (__init__.py) 挂 NullHandler, pip 使用方未配置日志时默认静默。
  - CLI / 训练脚本是一等应用方: 调用 ``setup_cli_logging()`` 让 ``gleamlm.*``
    日志以接近原 print 的观感（``%(message)s`` 无前缀）输出到 stderr ——
    WebUI 以 stdout+stderr 合并重定向启动训练, 日志观感与迁移前一致;
    终端手动运行同样可见。
  - 只动 ``gleamlm`` 命名空间, 不碰 root logger: torch 等第三方库日志不受影响。
"""

import logging


def setup_cli_logging(level: int = logging.INFO) -> None:
    """让 gleamlm 库日志输出到 stderr（幂等: 已有 StreamHandler 时不重复挂载）。

    仅入口脚本（manual/ 训练、data_tools/ 数据准备、tools/ 工具）调用；
    库模块自身绝不调用 —— 库不该为使用方决定日志去向。
    """
    logger = logging.getLogger("gleamlm")
    if any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
