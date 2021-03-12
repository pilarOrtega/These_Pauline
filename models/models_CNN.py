from tensorflow.keras.applications import VGG16, Xception, ResNet50
from tensorflow.keras.applications import resnet50, xception, vgg16
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
        'model': ResNet50,
        'module': resnet50
    },
    'VGG16': {
        'model': VGG16,
        'module': vgg16
    },
    'Xception': {
        'model': Xception,
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
