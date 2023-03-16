from yacs.config import CfgNode as CN
_CN = CN()

_CN.name = ''

def get_cfg():
    return _CN.clone()
