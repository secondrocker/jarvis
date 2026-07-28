"""进程内检查点存储工厂。"""

from langgraph.checkpoint.memory import MemorySaver


def create_checkpointer() -> MemorySaver:
    """为应用生命周期创建一个进程内 MemorySaver。

    不在模块导入阶段缓存实例；必须由 FastAPI 生命周期调用此函数，
    以便测试创建拥有独立检查点状态的应用。

    返回值:
        用于保存会话图状态的全新内存检查点存储。
    """
    return MemorySaver()
