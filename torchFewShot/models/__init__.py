
from __future__ import absolute_import

from .net import ResNet12, AngleLinear, Classifier
from .mlp import MLPN

from .protohead import ProtoNetHead


__model_factory = {
        'ResNet12': ResNet12,
        'AngleLinear': AngleLinear,
        'Classifier': Classifier,

        'Proto': ProtoNetHead,

        'MLP': MLPN,
}



def get_names():
    return list(__model_factory.keys()) 


def init_model(name, *args, **kwargs):
    if name not in list(__model_factory.keys()):
        raise KeyError("Unknown model: {}".format(name))
    return __model_factory[name](*args, **kwargs)

