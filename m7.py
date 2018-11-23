import tensorflow as tf
import numpy as np
import datetime
import timeit
import os


# Based off multiple online tutorials
# tensorboard --logdir="C:\Users\JP Kim\Desktop\GAN\tensorboard"
def discriminator(inputImages, reuse_variables=None):
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables) as scope:
        disW1 = tf.get_variable('disW1', [5, 5, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
        disB1 = tf.get_variable('disB1', [32], initializer=tf.constant_initializer(0))
        disW2 = tf.get_variable('disW2', [5, 5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
        disB2 = tf.get_variable('disB2', [64], initializer=tf.constant_initializer(0))
        disW3 = tf.get_variable('disW3', [7 * 7 * 64, 1024], initializer=tf.truncated_normal_initializer(stddev=0.02))
        disB3 = tf.get_variable('disB3', [1024], initializer=tf.constant_initializer(0))
        disW4 = tf.get_variable('disW4', [1024, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
        disB4 = tf.get_variable('disB4', [1], initializer=tf.constant_initializer(0))

        temp = tf.nn.conv2d(input=inputImages, filter=disW1, strides=[1, 1, 1, 1], padding='SAME')
        temp = tf.nn.leaky_relu(temp + disB1)
        temp = tf.nn.avg_pool(temp, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        temp = tf.nn.conv2d(input=temp, filter=disW2, strides=[1, 1, 1, 1], padding='SAME')
        temp = tf.nn.leaky_relu(temp + disB2)
        temp = tf.nn.avg_pool(temp, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        temp = tf.reshape(temp, [-1, 7 * 7 * 64])
        temp = tf.matmul(temp, disW3)
        temp = tf.nn.leaky_relu(temp + disB3)

        temp = tf.matmul(temp, disW4) + disB4

        return temp


def generator(noise, batchSize, noiseDim):
    genW1 = tf.get_variable('genW1', [noiseDim, 3136], dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.02))
    genB1 = tf.get_variable('genB1', [3136], initializer=tf.truncated_normal_initializer(stddev=0.02))
    genW2 = tf.get_variable('genW2', [3, 3, 1, noiseDim / 2], dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.02))
    genB2 = tf.get_variable('genB2', [noiseDim / 2], initializer=tf.truncated_normal_initializer(stddev=0.02))
    genW3 = tf.get_variable('genW3', [3, 3, noiseDim / 2, noiseDim / 4], dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.02))
    genB3 = tf.get_variable('genB3', [noiseDim / 4], initializer=tf.truncated_normal_initializer(stddev=0.02))
    genW4 = tf.get_variable('genW4', [1, 1, noiseDim / 4, 1], dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.02))
    genB4 = tf.get_variable('genB4', [1], initializer=tf.truncated_normal_initializer(stddev=0.02))

    temp = tf.matmul(noise, genW1) + genB1
    temp = tf.reshape(temp, [-1, 56, 56, 1])
    temp = tf.contrib.layers.batch_norm(temp, epsilon=1e-5, scope='genB1')
    temp = tf.nn.relu(temp)

    temp = tf.nn.conv2d(temp, genW2, strides=[1, 2, 2, 1], padding='SAME') + genB2
    temp = tf.contrib.layers.batch_norm(temp, epsilon=1e-5, scope='genB2')
    temp = tf.nn.relu(temp)
    temp = tf.image.resize_images(temp, [56, 56])

    temp = tf.nn.conv2d(temp, genW3, strides=[1, 2, 2, 1], padding='SAME') + genB3
    temp = tf.contrib.layers.batch_norm(temp, epsilon=1e-5, scope='genB3')
    temp = tf.nn.relu(temp)
    temp = tf.image.resize_images(temp, [56, 56])

    temp = tf.nn.conv2d(temp, genW4, strides=[1, 2, 2, 1], padding='SAME')

    # Pushes grey to black or white
    temp = tf.sigmoid(temp + genB4)

    return temp


def trainDis(data, epochCount, batchSize, noiseDim, sess, disTrainerReal, disTrainerFake, disRealLoss, disFakeLoss,
             inputImage, inputNoise, shouldSummary, merged, writer, outCount):
    for i in range(epochCount):
        noise = np.random.normal(0, 1, size=[batchSize, noiseDim])
        realBatch = data.train.next_batch(batchSize)[0].reshape([batchSize, 28, 28, 1])
        _, __, tempRealLoss, tempFakeLoss = sess.run([disTrainerReal, disTrainerFake, disRealLoss, disFakeLoss],
                                                     {inputImage: realBatch, inputNoise: noise})

        if shouldSummary:
            noise = np.random.normal(0, 1, size=[batchSize, noiseDim])
            summary = sess.run(merged, {inputNoise: noise, inputImage: realBatch})
            writer.add_summary(summary, outCount)


def trainGen(epochCount, batchSize, noiseDim, sess, genTrainer, inputNoise):
    for i in range(epochCount):
        noise = np.random.normal(0, 1, size=[batchSize, noiseDim])
        sess.run(genTrainer, feed_dict={inputNoise: noise})


def trainMain(startEpoch, data, epochCount, batchSize, noiseDim, sess, genTrainer, disTrainerReal, disTrainerFake,
              disRealLoss, disFakeLoss, inputImage, inputNoise, saver, merged, writer):
    start = timeit.default_timer()
    for i in range(int(startEpoch.eval(sess)), epochCount):
        if i % 10 == 0 and i != 0:
            trainDis(data, 1, batchSize, noiseDim, sess, disTrainerReal, disTrainerFake, disRealLoss, disFakeLoss,
                     inputImage, inputNoise, True, merged, writer, i)
        else:
            trainDis(data, 1, batchSize, noiseDim, sess, disTrainerReal, disTrainerFake, disRealLoss, disFakeLoss,
                     inputImage, inputNoise, False, merged, writer, i)

        trainGen(1, batchSize, noiseDim, sess, genTrainer, inputNoise)

        if i % 100 == 0 and i != 0:
            sess.run(tf.assign(startEpoch, i))
            saver.save(sess, "trained/checkpoint", global_step=i)
            print("")
            print("Epoch Count: %i" % i)
            stop = timeit.default_timer()
            print("Elapsed Time: " + str(stop - start))
            start = timeit.default_timer()


def main():
    data = tf.contrib.learn.datasets.load_dataset("mnist")

    noiseDim = 100
    batchSize = 50
    preTrainCount = 300
    epochCount = 100000
    start = tf.get_variable("start", dtype=tf.int32, trainable=False, initializer=0)

    inputImage = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='inputImage')
    inputNoise = tf.placeholder(tf.float32, [None, noiseDim], name='inputNoise')
    genImages = generator(inputNoise, batchSize, noiseDim)
    predictionsReal = discriminator(inputImage)
    predictionsFake = discriminator(genImages, reuse_variables=True)

    disRealLoss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=predictionsReal, labels=tf.ones_like(predictionsReal)))
    disFakeLoss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=predictionsFake, labels=tf.zeros_like(predictionsFake)))
    genLoss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=predictionsFake, labels=tf.ones_like(predictionsFake)))

    tvars = tf.trainable_variables()
    disVars = [var for var in tvars if 'dis' in var.name]
    genVars = [var for var in tvars if 'gen' in var.name]

    disTrainerFake = tf.train.AdamOptimizer(0.0003).minimize(disFakeLoss, var_list=disVars)
    disTrainerReal = tf.train.AdamOptimizer(0.0003).minimize(disRealLoss, var_list=disVars)
    genTrainer = tf.train.AdamOptimizer(0.0001).minimize(genLoss, var_list=genVars)

    tf.get_variable_scope().reuse_variables()

    sess = tf.Session()
    saver = tf.train.Saver(max_to_keep=0)

    tf.summary.scalar('Generator_loss', genLoss)
    tf.summary.scalar('Discriminator_loss_real', disRealLoss)
    tf.summary.scalar('Discriminator_loss_fake', disFakeLoss)

    displayImages = generator(inputNoise, batchSize, noiseDim)
    tf.summary.image('Generated_images', displayImages, 5)
    merged = tf.summary.merge_all()
    logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    writer = tf.summary.FileWriter(logdir, sess.graph)

    print(tf.train.latest_checkpoint('trained/'))

    os.system('cls')

    try:
        saver.restore(sess, tf.train.latest_checkpoint('trained/'))
    except:
        print("Checkpoint not found")
        print("")
        print("Starting pre-training")
        sess.run(tf.global_variables_initializer())
        trainDis(data, preTrainCount, batchSize, noiseDim, sess, disTrainerReal, disTrainerFake, disRealLoss,
                 disFakeLoss, inputImage, inputNoise, False, merged, writer, 0)
        print("Finished pre-training")
    else:
        print("Checkpoint found")
        print("Starting from epoch %e" % start.eval(sess))

    trainMain(start, data, epochCount, batchSize, noiseDim, sess, genTrainer, disTrainerReal, disTrainerFake,
              disRealLoss, disFakeLoss, inputImage, inputNoise, saver, merged, writer)

    saver.save(sess, './trained/trainedModel.ckpt')

    print("Finished training")


if __name__ == "__main__":
    main()