from keras.models import Model
from keras.layers import Conv2D, Dropout, Conv2DTranspose, add
from backbones import get_backbone
from keras.optimizers import Adam


def fcn(backbone_name='vgg16', input_shape=(224,224,3), num_classes=1000, skip=True):
    # backbone
    backbone, encoder_features = get_backbone(backbone_name, input_shape)
    pool4, pool3 = backbone.get_layer(encoder_features[0]).output, backbone.get_layer(encoder_features[1]).output
    x = backbone.output

    # fc_convolutionalization
    x = Conv2D(4096, 7, activation='relu', padding='same')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(4096, 1, activation='relu', padding='same')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(num_classes, 1, activation='relu', padding='same')(x)

    # upsampling
    if not skip:
        x = Conv2DTranspose(num_classes, 64, strides=32, padding='same', activation='softmax')(x)
    else:
        x = Conv2DTranspose(num_classes, 4, strides=2, padding='same', activation='relu')(x)
        pool4 = Conv2D(num_classes, 1)(pool4)
        x = add([x, pool4])

        x = Conv2DTranspose(num_classes, 4, strides=2, padding='same')(x)
        pool3 = Conv2D(num_classes, 1)(pool3)
        x = add([x, pool3])

        x = Conv2DTranspose(num_classes, 16, strides=8, padding='same', activation='softmax')(x)

    model = Model(backbone.input, x)

    adam = Adam(lr=3e-4, decay=5e-3)
    model.compile(adam, loss='categorical_crossentropy', metrics=['acc'])

    return model


if __name__ == '__main__':
    model = fcn(backbone_name='vgg16', input_shape=(224,224,3), num_classes=1000, skip=True)
    model.summary()


