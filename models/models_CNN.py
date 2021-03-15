from tensorflow.keras.applications import *
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D


class Lenet(Model):

    def __init__(self):
        super(Lenet, self).__init__()
        self.conv1 = Conv2D(32, (3, 3), activation="relu")
        self.pool1 = MaxPooling2D()
        self.conv2 = Conv2D(64, (3, 3), activation="relu")
        self.pool2 = MaxPooling2D()
        self.conv3 = Conv2D(128, (3, 3), activation="relu")
        self.pool3 = MaxPooling2D()

    def call(self, input):
        y = self.conv1(input)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.pool2(y)
        y = self.conv3(y)
        y = self.pool3(y)
        return y


models = {
    'ResNet50': {
        'model': resnet50.ResNet50,
        'module': resnet50
    },
    'DenseNet121': {
        'model': densenet.DenseNet121,
        'module': densenet
    },
    'DenseNet169': {
        'model': densenet.DenseNet169,
        'module': densenet
    },
    'NASNet': {
        'model': nasnet.NASNetMobile,
        'module': nasnet
    },
    'VGG16': {
        'model': vgg16.VGG16,
        'module': vgg16
    },
    'Xception': {
        'model': xception.Xception,
        'module': xception
    },
    'customLenet': {
        'model': Lenet,
        'module': xception
    }
}


class Error(Exception):
    """
    Base of custom errors.

    **********************
    """

    pass


class UnknownModelError(Error):
    """
    Raise when trying to access unknown level.

    *********************************************
    """

    pass


def get_model(name):
    if name not in models:
        raise UnknownModelError("Model {} is not implemented!!!".format(name))
    else:
        return models[name]
