import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def get_data():
    file_name = 'test.jpg'
    im = cv2.imread('test.jpg', 1)


    r = 600.0 / im.shape[1]
    dim = (600, int(im.shape[0] * r))

    # perform the actual resizing of the image and show it
    resized = cv2.resize(im, dim, interpolation=cv2.INTER_AREA)
    # imgYCC = cv2.cvtColor(resized, cv2.COLOR_BGR2YCR_CB)
    yuv = cv2.cvtColor(resized, cv2.COLOR_BGR2YUV)
    # print(resized.shape, yuv.shape)

    # cv2.imshow('image', yuv[:,:,0])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite("thumbnail.png", resized)

    input = yuv[:,:,0]
    input = np.reshape(input, [1,input.shape[0],input.shape[1], 1])
    # print(input.shape)

    dataset = tf.data.Dataset.from_tensor_slices(input)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    with tf.Session() as sess:
        sess.run(iterator.initializer)
        temp = (sess.run(next_element))
        # temp = sess.run(dataset)
        print(type(temp))
        print(temp.shape)
        plt.imshow(temp[:, :, 0], 'gray')
        plt.show()


def load_models():


if __name__ == '__main__':


