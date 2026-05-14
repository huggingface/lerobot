from .configuration_molmoact2 import MolmoAct2Config

__all__ = ["MolmoAct2Config", "MolmoAct2Policy", "make_molmoact2_pre_post_processors"]


def __getattr__(name):
    if name == "MolmoAct2Policy":
        from .modeling_molmoact2 import MolmoAct2Policy

        return MolmoAct2Policy
    if name == "make_molmoact2_pre_post_processors":
        from .processor_molmoact2 import make_molmoact2_pre_post_processors

        return make_molmoact2_pre_post_processors
    raise AttributeError(name)
