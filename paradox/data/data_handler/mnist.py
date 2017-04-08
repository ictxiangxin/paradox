import os
import gzip
import struct
import numpy


train_image_gzip_file = 'train-images-idx3-ubyte.gz'
train_label_gzip_file = 'train-labels-idx1-ubyte.gz'
test_image_gzip_file = 't10k-images-idx3-ubyte.gz'
test_label_gzip_file = 't10k-labels-idx1-ubyte.gz'
train_image_ubyte_file = 'train-images.idx3-ubyte'
train_label_ubyte_file = 'train-labels.idx1-ubyte'
test_image_ubyte_file = 't10k-images.idx3-ubyte'
test_label_ubyte_file = 't10k-labels.idx1-ubyte'


class MNIST:
    def __init__(self, path: str):
        self.__path = path

    def read_image(self, file: str, is_gzip: bool=True):
        file_path = os.path.join(self.__path, file)
        images = []
        offset = 0
        with (gzip if is_gzip else os).open(file_path, 'rb') as fp:
            data = fp.read()
            magic, number, rows, columns = struct.unpack_from('>IIII', data, offset)
            offset += struct.calcsize('>IIII')
            if magic != 0x803:
                raise ValueError('This is not MNIST image file: {}'.format(file))
            image_size = rows * columns
            image_struct = '>{}B'.format(image_size)
            for i in range(number):
                image_data = struct.unpack_from(image_struct, data, offset)
                offset += struct.calcsize(image_struct)
                images.append(numpy.array(image_data).reshape(rows, columns))
        return numpy.array(images)

    def read_label(self, file: str, is_gzip: bool=True):
        file_path = os.path.join(self.__path, file)
        labels = []
        offset = 0
        with (gzip if is_gzip else os).open(file_path, 'rb') as fp:
            data = fp.read()
            magic, number = struct.unpack_from('>II', data, offset)
            offset += struct.calcsize('>II')
            if magic != 0x801:
                raise ValueError('This is not MNIST image file: {}'.format(file))
            for i in range(number):
                labels.append(struct.unpack_from('>1B', data, offset)[0])
                offset += struct.calcsize('>1B')
        return numpy.array(labels)

    def read(self):
        if os.path.exists(os.path.join(self.__path, train_image_gzip_file)):
            train_image = self.read_image(os.path.join(self.__path, train_image_gzip_file), is_gzip=True)
        elif os.path.exists(os.path.join(self.__path, train_image_ubyte_file)):
            train_image = self.read_image(os.path.join(self.__path, train_image_ubyte_file), is_gzip=False)
        else:
            raise FileNotFoundError('Can not find train image file.')
        if os.path.exists(os.path.join(self.__path, test_image_gzip_file)):
            test_image = self.read_image(os.path.join(self.__path, test_image_gzip_file), is_gzip=True)
        elif os.path.exists(os.path.join(self.__path, test_image_ubyte_file)):
            test_image = self.read_image(os.path.join(self.__path, test_image_ubyte_file), is_gzip=False)
        else:
            raise FileNotFoundError('Can not find test image file.')
        if os.path.exists(os.path.join(self.__path, train_label_gzip_file)):
            train_label = self.read_label(os.path.join(self.__path, train_label_gzip_file), is_gzip=True)
        elif os.path.exists(os.path.join(self.__path, train_label_ubyte_file)):
            train_label = self.read_label(os.path.join(self.__path, train_label_ubyte_file), is_gzip=False)
        else:
            raise FileNotFoundError('Can not find train label file.')
        if os.path.exists(os.path.join(self.__path, test_label_gzip_file)):
            test_label = self.read_label(os.path.join(self.__path, test_label_gzip_file), is_gzip=True)
        elif os.path.exists(os.path.join(self.__path, test_label_ubyte_file)):
            test_label = self.read_label(os.path.join(self.__path, test_label_ubyte_file), is_gzip=False)
        else:
            raise FileNotFoundError('Can not find test label file.')
        return {'train_image': train_image, 'train_label': train_label, 'test_image': test_image, 'test_label': test_label}
