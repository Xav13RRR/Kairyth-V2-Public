from .unirig_ar import UniRigAR

from .spec import ModelSpec

def get_model(**kwargs) -> ModelSpec:
    MAP = {
        'unirig_ar': UniRigAR,
    }
    __target__ = kwargs['__target__']
    del kwargs['__target__']
    if __target__ == 'unirig_skin':
        from .unirig_skin import UniRigSkin
        MAP['unirig_skin'] = UniRigSkin
    assert __target__ in MAP, f"expect: [{','.join(MAP.keys())}], found: {__target__}"
    return MAP[__target__](**kwargs)
