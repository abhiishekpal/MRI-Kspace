import tensorflow as tf
import numpy as np
import pickle
import os
from skimage.measure import compare_ssim as ssim


tf.reset_default_graph()

logs_path = "/home/dlagroup5/Project/network_codes/logs"
path1 = '/home/dlagroup5/Project/undersampled/network_codes/data'

batch_size = 1
epochs = 50
learning_rate = 0.0001
left_input = tf.placeholder(tf.float32, [None, 320, 320, 256*4], name='left')
right_input = tf.placeholder(tf.float32,[None, 320, 320, 256*4], name='right')
gt_ = tf.placeholder(tf.complex64, [None, 320, 320, 256*4], name='right')
lr = tf.Variable(0.1)

L = 0.3


def ssim_loss(img, img_gt):
    #loss = ssim(img, img_gt, data_range=img.max() - img.min())
    loss = tf.losses.mean_squared_error(predictions = img,labels = img_gt)
    return loss

def calc_loss(left_pred,right_pred, gt_):
    img_f = tf.complex(right_pred, left_pred)
    img = np.abs(np.fft.ifftn(img_f))
    img_gt = np.abs(np.fft.ifftn(gt_))
    loss = ssim_loss(img,img_gt)
    
    return loss

def next_batch_real(i):
    fo1 = os.path.join(path1,"input{}.npy".format(i))
    dat1 = np.load(fo1)
    dat11 = np.abs(np.real(dat1))
    return dat11

def next_batch_imaginary(i):
    fo1 = os.path.join(path1,"input{}.npy".format(i))
    dat1 = np.load(fo1)
    dat11 =np.abs( np.imag(dat1))
    return dat11

def next_gt(i):
    fo1 = os.path.join(path1,"output{}.npy".format(i))
    dat1 = np.load(fo1)
    return dat1

def SegNet(input_, reuse=False):

    '''Encoder'''

    conv1 = tf.layers.conv2d(inputs = input_, filters = 64, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "conv1", reuse=reuse)
    b_norm1 = tf.layers.batch_normalization(inputs = conv1, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "batch_norm1", reuse = reuse)
    pool1 = tf.layers.max_pooling2d(inputs = b_norm1, pool_size = 2, strides=2, name = "max_pool1")

    conv2 = tf.layers.conv2d(inputs = pool1, filters = 128, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "conv2", reuse=reuse)
    b_norm2 = tf.layers.batch_normalization(inputs = conv2, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "batch_norm2", reuse = reuse)
    pool2 = tf.layers.max_pooling2d(inputs = b_norm2, pool_size = 2, strides=2, name = "max_pool2")


    conv3 = tf.layers.conv2d(inputs = pool2, filters = 256, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "conv3", reuse=reuse)
    b_norm3 = tf.layers.batch_normalization(inputs = conv3, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "batch_norm3", reuse = reuse)
    pool3 = tf.layers.max_pooling2d(inputs = b_norm3, pool_size = 2, strides=2, name = "max_pool3")

    conv4 = tf.layers.conv2d(inputs = pool3, filters = 512, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "conv4", reuse=reuse)
    b_norm4 = tf.layers.batch_normalization(inputs = conv4, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "batch_norm4", reuse = reuse)
    pool4 = tf.layers.max_pooling2d(inputs = b_norm4, pool_size = 2, strides=2, name = "max_pool4")


    '''Decoder'''

    conv5 = tf.layers.conv2d(inputs = pool4, filters = 512, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "conv5", reuse=reuse)
    b_norm5 = tf.layers.batch_normalization(inputs = conv5, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "batch_norm5", reuse = reuse)

    up1 = tf.keras.layers.UpSampling2D(2)(b_norm5)
    conv6 = tf.layers.conv2d(inputs = up1, filters = 256, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "conv6", reuse=reuse)
    b_norm6 = tf.layers.batch_normalization(inputs = conv6, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "batch_norm6", reuse = reuse)

    up2 = tf.keras.layers.UpSampling2D(2)(b_norm6)
    conv7 = tf.layers.conv2d(inputs = up2, filters = 128, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "conv7", reuse=reuse)
    b_norm7 = tf.layers.batch_normalization(inputs = conv7, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "batch_norm7", reuse = reuse)

    up3 = tf.keras.layers.UpSampling2D(2)(b_norm7)
    conv8 = tf.layers.conv2d(inputs = up3, filters = 64, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "conv8", reuse=reuse)
    b_norm8 = tf.layers.batch_normalization(inputs = conv8, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "batch_norm8", reuse = reuse)

    up4 = tf.keras.layers.UpSampling2D(2)(b_norm8)


    conv9 = tf.layers.conv2d(inputs = up4, filters = 128, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "conv9", reuse=reuse)
    b_norm9 = tf.layers.batch_normalization(inputs = conv9, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "batch_norm9", reuse = reuse)

    conv10 = tf.layers.conv2d(inputs = b_norm9, filters = 256*4, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "conv10", reuse=reuse)
    b_norm10 = tf.layers.batch_normalization(inputs = conv10, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "batch_norm10", reuse = reuse)



    out1 = tf.reshape(b_norm10,(1,320,320,256*4))
    out_logit = tf.nn.softmax(out1, name="softmax_output")

    return out_logit

def nn():
    left1 =  SegNet(left_input, reuse=False)
    right1 = SegNet(right_input, reuse=True)
    
    cost = tf.reduce_mean(calc_loss(left1,right1,gt_))
    optimizer = tf.train.AdamOptimizer(lr).minimize(cost)

#     tf.summary.scalar("cost", cost)

#     summary_op = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
   # writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    with tf.Session() as sess:
        #saver.restore(sess,"./model.ckpt")
        sess.run(init)
        pt = 0
        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(1,20):
                if(i==12):
                    continue
                train_x = np.ndarray((1,320, 320, 256*4), dtype = float)
                train_y = np.ndarray((1,320, 320, 256*4), dtype = float)
                train_z = np.ndarray((1,320, 320, 256*4), dtype = complex)
                left_ =  next_batch_imaginary(i)
                right_ =  next_batch_real(i)
                gt = next_gt(i)

                train_x[0] = left_
                train_y[0] = right_
                train_z[0] = gt
                batch_x = np.array(train_x)
                batch_y = np.array(train_y)
                batch_z = np.array(train_z)
#                 print(batch_x.shape,batch_y.shape)
                c = 0
                _, c = sess.run([optimizer,cost], feed_dict={left_input: batch_x, right_input: batch_y, gt_: batch_z, lr: learning_rate})
               # writer.add_summary(summary, pt)
                pt += 1
                epoch_loss += c
                save_path = saver.save(sess, "./freeze/model.ckpt")
                    
                print("*"*10)
            print(epoch," ",epoch_loss)

def main():
    nn()

if __name__ == "__main__":
    main()
