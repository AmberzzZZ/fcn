from model import fcn
from dataLoader import trainGenerator
from keras.callbacks import ModelCheckpoint
import os


if __name__ == '__main__':

    train_dir = "/Users/amber/workspace/unet_vnet_keras/data/disc/train/"
    val_dir = "/Users/amber/workspace/unet_vnet_keras/data/disc/test/"
    img_folder = 'image'
    label_folder = 'label'
    batch_size = 4
    num_classes = 1 + 1

    train_generator = trainGenerator(train_dir, img_folder, label_folder, batch_size)
    val_generator = trainGenerator(val_dir, img_folder, label_folder, batch_size)

    model = fcn(backbone_name='vgg16', input_shape=(512,512,1), num_classes=num_classes, skip=True)

    # checkpoint
    filepath = "fcn_disc_{epoch:02d}_val_loss_{val_loss:.3f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss',verbose=1, save_best_only=True, mode='min')

    steps = len(os.listdir(train_dir+img_folder)) // batch_size * 2
    model.fit_generator(train_generator,
                        steps_per_epoch=steps,
                        epochs=20,
                        verbose=1,
                        callbacks=[checkpoint],
                        validation_data=val_generator,
                        validation_steps=steps//5)




