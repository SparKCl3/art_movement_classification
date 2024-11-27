from preprocessing import preprocessing
from model import get_from_directory
import os

#Preprocessing
folder_path_train = preprocessing('train')
folder_path_test = preprocessing('test')
folder_path_val = preprocessing('valid')

#tensorflow object
batch_size = int(os.environ.get('BATCH_SIZE', 32))

train_ds = get_from_directory(folder_path_train, batch_size, 'rgb', image_size=(416,416))
val_ds = get_from_directory(folder_path_val, batch_size, 'rgb' , image_size=(416,416))
test_ds = get_from_directory(folder_path_test, batch_size, 'rgb', image_size=(416,416))


if __name__ == '__main__':
    pass
