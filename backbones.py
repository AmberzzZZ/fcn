from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50


def get_backbone(backbone_name, input_shape):
    vgg16 = VGG16(include_top=False, weights=None, input_shape=input_shape, pooling=None)
    resnet50 = ResNet50(include_top=False, weights=None, input_shape=input_shape, pooling=None)
    models = {'vgg16': vgg16, 'resnet50': resnet50}
    encoder_features = {'vgg16': ('block4_pool', 'block3_pool', 'block2_pool'),
                        'resnet50': ('activation_40', 'activation_22', 'activation_10', 'activation_1')}
    return models[backbone_name], encoder_features[backbone_name]



if __name__ == '__main__':
    pass

