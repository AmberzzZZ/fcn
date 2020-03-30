from keras.models import Model
from keras.layers import Conv2D, Dropout, Conv2DTranspose, add, UpSampling2D
from backbones import get_backbone
from keras.optimizers import Adam
import numpy as np

def fcn(backbone_name='vgg16', input_shape=(224,224,3), num_classes=1000, skip=True, fixed=True):
    # backbone
    backbone, encoder_features = get_backbone(backbone_name, input_shape)
    pool4 = backbone.get_layer(encoder_features[0]).output
    pool3 = backbone.get_layer(encoder_features[1]).output
    x = backbone.output

    # fc_convolutionalization
    x = Conv2D(4096, 7, activation='relu', padding='same')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(4096, 1, activation='relu', padding='same')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(num_classes, 1, activation='relu', padding='same')(x)

    # upsampling
    if not skip:
        x = Conv2DTranspose(num_classes, 64, strides=32, padding='same', activation='softmax',
                            kernel_initializer='bilinear')(x)
    else:
        x = Conv2DTranspose(num_classes, 4, strides=2, padding='same', kernel_initializer=bilinear)(x)
        pool4 = Conv2D(num_classes, 1, kernel_initializer='zero')(pool4)
        x = add([x, pool4])

        x = Conv2DTranspose(num_classes, 4, strides=2, padding='same', kernel_initializer=bilinear)(x)
        pool3 = Conv2D(num_classes, 1, kernel_initializer='zero')(pool3)
        x = add([x, pool3])

        x = Conv2DTranspose(num_classes, 16, strides=8, padding='same', kernel_initializer=bilinear,
                            activation='softmax', name='final_deconv')(x)

    model = Model(backbone.input, x)
    if skip and fixed:
        model.get_layer('final_deconv').trainable = False

    adam = Adam(lr=3e-4, decay=5e-3)
    model.compile(adam, loss='categorical_crossentropy', metrics=['acc'])

    return model


def bilinear(shape, dtype='float32'):
    in_channels, out_channels, kernel_size, kernel_size = shape
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=dtype)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return np.array(weight)


if __name__ == '__main__':
    model = fcn(backbone_name='vgg16', input_shape=(224,224,3), num_classes=21, skip=True)
    model.summary()


