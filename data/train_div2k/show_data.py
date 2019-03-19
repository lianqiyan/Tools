import tensorflow as tf
from collections import OrderedDict
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '6'
_batch_size = 1
_shuffle_buffer_size = 1
TRAINING_DATASET_INFO_PATH='dataset_info.txt'
TESTING_DATASET_PATH='datasets/test_div2k/dataset.tfrecords'
TESTING_DATASET_INFO_PATH='datasets/test_div2k/dataset_info.txt'
#get the information of the dataset
with open(TRAINING_DATASET_INFO_PATH) as dataset_info:
    example_num = int(dataset_info.readline())
    scale_factor = int(dataset_info.readline())
    input_info = OrderedDict()
    for line in dataset_info.readlines():
        items = line.split(',')
        input_info[items[0]] = [int(dim) for dim in items[1:]]


def _parse_tf_example( example_proto):
    features = dict([(key, tf.FixedLenFeature([], tf.string)) for key, _ in input_info.items()])
    # print(features['lr0'])
    parsed_features = tf.parse_single_example(example_proto, features=features)
    # parsed_features = [tf.reshape(tf.cast(tf.decode_raw(parsed_features[key], tf.uint8), tf.float32), value)
    #          for key, value in input_info.items()]
    # print(parsed_features)
    image_keys = list(input_info.keys())
    image_shape = list(input_info.values())
    images = []
    for i in range(4):
        images.append(tf.reshape(tf.decode_raw(parsed_features[image_keys[i]], tf.uint8), image_shape[i]))
    # print(parsed_features['lr0'])
    return images

    #
    # return [tf.reshape(tf.cast(tf.decode_raw(parsed_features[key], tf.uint8), tf.float32), value)
    #         for key, value in input_info.items()]


def read_by_seqence():
    image_keys = list(input_info.keys())
    image_shape = list(input_info.values())
    filename_queue = tf.train.string_input_producer(['dataset.tfrecords'])
    # Unlike the TFRecordWriter, the TFRecordReader is symbolic, 即所做的操作不会立即执行
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features=dict(
        [(key, tf.FixedLenFeature([], tf.string)) for key, _ in input_info.items()]))
    image_lr0 = tf.reshape(tf.cast(tf.decode_raw(features[image_keys[0]], tf.uint8), tf.float32), image_shape[0])


if __name__ == '__main__':
    dataset = tf.data.TFRecordDataset('dataset.tfrecords')
    images = dataset.map(_parse_tf_example)
    # print(images)

    dataset = images.batch(16)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    # if _shuffle_buffer_size != 0:
    #     dataset = dataset.shuffle(buffer_size=_shuffle_buffer_size)
    #     dataset = dataset.repeat()

    # print(hr_batch)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    # print([(key, value) for key, value in input_info.items()])

    with tf.Session() as sess:
        sess.run(iterator.initializer)
        temp = (sess.run(next_element))
        print(temp[1].shape)
        print(len(temp))
        print(type(temp))

        # print((sess.run(next_element)))



        # sess.run(init_op)
        # tf.train.start_queue_runners()
        # print(sess.run(image_lr0))
        # print(sess.run(image_lr0))
        # print(hr_batch)
    # print(keys)
    # print(data_batch)
    # print(input_info)
