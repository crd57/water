import numpy as np
import ReadWrite as rw
import pandas as pd
from tqdm import trange


def gen_data_list(water_path, background_path, img_path, SLIC_path):
    im_proj, im_geotrans, im_data, im_width, im_height = rw.read_img(img_path)
    _, _, SLIC, _, _ = rw.read_img(SLIC_path)
    img = np.zeros([im_height, im_width, 4], dtype=np.float32)
    for i in range(4):
        img[:, :, i] = (im_data[i, :, :] - np.min(im_data[i, :, :])) / (
                    np.max(im_data[i, :, :]) - np.min(im_data[i, :, :]))
    water_list = pd.read_csv(water_path)
    background_list = pd.read_csv(background_path)
    Water = []
    BackGround = []
    for i in range(water_list['File X'].size):
        Water.append(SLIC[water_list[' File Y'][i], water_list['File X'][i],])
        Water = list(set(Water))
    for i in range(background_list['File X'].size):
        BackGround.append(SLIC[background_list[' File Y'][i], background_list['File X'][i],])
        BackGround = list(set(BackGround))
    label = list(np.ones([len(Water)]))
    label.extend(list(np.zeros([len(BackGround)])))
    data_ = Water
    data_.extend(BackGround)
    arr = np.arange(len(label))
    np.random.shuffle(arr)
    data_ = np.array(data_)
    label = np.array(label)
    data_ = data_[arr]
    label = label[arr]
    data = []
    for i in data_:
        position = np.where(SLIC == i)
        if len(position[0]) >=144:
            cluster = np.zeros([144, 4], dtype=np.float32)
            cluster[:,:] = img[position[0][0:144],position[1][0:144],:]
            cluster = np.reshape(cluster,(12,12,-1))
            data.append(cluster)
        else :
            cluster = np.zeros([144,4],dtype=np.float32)
            cluster[0:len(position[0]),:] = img[position[0],position[1],:]
            cluster = np.reshape(cluster, (12, 12, -1))
            data.append(cluster)
    label = one_hot(label)
    np.save("D:/Code/BeiLu/Water/data/summerGF/data.npy", data)
    np.save("D:/Code/BeiLu/Water/data/summerGF/label.npy", label)
    return data_, label, img, SLIC


def save_data(SLIC, img):
    SLIC_data = []
    for i in trange(int(np.max(SLIC)) + 1):
        position = np.where(SLIC == i)
        if len(position[0]) >= 144:
            cluster = np.zeros([144, 4], dtype=np.float32)
            cluster[:, :] = img[position[0][0:144], position[1][0:144], :]
            cluster = np.reshape(cluster, (12, 12, -1))
            SLIC_data.append(cluster)
        else:
            cluster = np.zeros([144, 4], dtype=np.float32)
            cluster[0:len(position[0]), :] = img[position[0], position[1], :]
            cluster = np.reshape(cluster, (12, 12, -1))
            SLIC_data.append(cluster)
    np.save("D:/Code/BeiLu/Water/data/summerGF/image_data.npy", SLIC_data)
    np.save("D:/Code/BeiLu/Water/data/summerGF/images_.npy", SLIC)


def one_hot(label):
    a = np.zeros([len(label), 2], dtype=np.float32)
    for i in range(len(label)):
        ind = int(label[i])
        a[i, ind] = 1
    return a


def divide(data, label, ratio=0.8):
    index = int(len(data) * ratio)
    index2 = (len(data) - index) // 2
    train_data = data[:index]
    test_data = data[index:index + index2]
    train_label = label[:index]
    test_label = label[index:index + index2]
    verification_data = data[index + index2:]
    verification_label = label[index + index2:]
    np.save("D:/Code/BeiLu/Water/data/ecognition/train_data.npy", train_data)
    np.save("D:/Code/BeiLu/Water/data/ecognition/test_data.npy", test_data)
    np.save("D:/Code/BeiLu/Water/data/ecognition/verification_data.npy", verification_data)
    np.save("D:/Code/BeiLu/Water/data/ecognition/train_label.npy", train_label)
    np.save("D:/Code/BeiLu/Water/data/ecognition/test_label.npy", test_label)
    np.save("D:/Code/BeiLu/Water/data/ecognition/verification_label.npy", verification_label)


def get_bath(data, label, batch_size):
    for i in range(len(data) // batch_size):
        pos = i * batch_size
        data_batch = data[pos:pos + batch_size, :, :, :]
        label_batch = label[pos:pos + batch_size, :]
        data_batchs = np.zeros([batch_size, 12, 12, 4])
        for j in range(len(data_batch)):
            data_batchs[j, :, :, :] = data_batch[j]
        yield data_batchs, label_batch
    remainder = len(data) % batch_size
    if remainder != 0:
        data_batch = data[-remainder:, :, :, :]
        label_batch = label[-remainder:, :]
        data_batchs = np.zeros([remainder, 12, 12, 4])
        for j in range(len(data_batch)):
            data_batchs[j, :, :, :] = data_batch[j]
        yield data_batchs, label_batch


if __name__ == '__main__':
    water_path = 'D:/Code/BeiLu/Water/data/summerGF/water.csv'
    background_path = 'D:/Code/BeiLu/Water/data/summerGF/background.csv'
    img_path = 'D:/Code/BeiLu/Water/data/summerGF/image.tif'
    segmentation_path = "D:/Code/BeiLu/Water/data/summerGF/segment.tif"
    data, label, img, SLIC = gen_data_list(water_path, background_path, img_path, segmentation_path)
    save_data(SLIC, img)
