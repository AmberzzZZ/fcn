import cv2
import numpy as np
import random
import os


def trainGenerator(data_pt, img_folder, label_folder, batch_size):
    fileLst = os.listdir(data_pt + img_folder)
    idx = [i for i in range(len(fileLst))]
    while 1:
        img_batch = []
        mask_batch = []
        random.shuffle(idx)
        for i in idx:
            img = cv2.imread(os.path.join(data_pt, img_folder, fileLst[i]), 0)
            mask_img = cv2.imread(os.path.join(data_pt, label_folder, fileLst[i]), 0)
            if np.max(img) > 1:
                img = img / 255.
            img = np.expand_dims(img, axis=-1)
            mask_img[mask_img>1] = 1
            mask = np.zeros((512,512,2))
            mask[:,:,0] = (mask_img==0).astype(np.uint8)
            mask[:,:,1] = (mask_img==1).astype(np.uint8)
            img_batch.append(img)
            mask_batch.append(mask)
            if len(img_batch) == batch_size:
                break
        yield np.array(img_batch), np.array(mask_batch)


if __name__ == '__main__':

    train_dir = "/Users/amber/workspace/unet_vnet_keras/data/disc/train/"
    val_dir = "/Users/amber/workspace/unet_vnet_keras/data/disc/test/"
    img_folder = 'image'
    label_folder = 'label'
    batch_size = 1

    train_generator = trainGenerator(train_dir, img_folder, label_folder, batch_size)

    for idx, data_batch in enumerate(train_generator):
        img_batch, mask_batch = data_batch
        print(img_batch.shape, np.max(img_batch[0]), np.min(img_batch[0]))
        print(mask_batch.shape, np.max(mask_batch[0]), np.min(mask_batch[0]))
        cv2.imshow("tmp", img_batch[0,:,:,0])
        cv2.waitKey(0)
        cv2.imshow("tmp", mask_batch[0,:,:,0]*255)
        cv2.waitKey(0)
        cv2.imshow("tmp", mask_batch[0,:,:,1]*255)
        cv2.waitKey(0)

        if idx > 2:
            break

