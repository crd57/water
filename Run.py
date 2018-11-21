from DataPretreatment import *
from Model import *
import time
from tqdm import trange

im_proj, im_geotrans, SLIC_data, im_width, im_height = rw.read_img("D:/Code/BeiLu/Water/data/summerGF/segment.tif")
# # SLIC = np.load("D:/Code/BeiLu/Water/data/data/SLIC_data.npy")
# img_path = 'D:/Code/BeiLu/Water/data/warp.tif'
# _,_,im_data,_,_ = rw.read_img(img_path)
# img = np.zeros([im_height,im_width,4],dtype=np.float32)
# for i in range(4):
#     img[:, :, i] = (im_data[i, :, :] - np.min(im_data[i, :, :])) / (np.max(im_data[i, :, :]) - np.min(im_data[i, :, :]))
train_data_path = "D:/Code/BeiLu/Water/data/summerGF/data.npy"
train_label_path = "D:/Code/BeiLu/Water/data/summerGF/label.npy"
datass = np.load(train_data_path)
labelss = np.load(train_label_path)

inputs, targets, keep_prob, prediction, train_step, merged, graph, accuracy, loss, learning_rate = build_graph(isTrain)
with tf.Session(graph=graph) as sess:
    summary_writer = tf.summary.FileWriter("D:/Code/BeiLu/Water/data/summerGF/graph", sess.graph)
    saver = tf.train.Saver(max_to_keep=10)
    step = 0
    init = tf.global_variables_initializer()
    sess.run(init)
    if isTrain == True:
        # test_data = np.load("D:/Code/BeiLu/Water/data/ecognition/image_data.npy")
        for i in range(epochs):
            for data_batch, label_batch in get_bath(datass, labelss, batch_size):
                r_t, _, loss_, acu, s_m, lr = sess.run([prediction, train_step, loss, accuracy, merged, learning_rate],
                                                       {inputs: data_batch, targets: label_batch, keep_prob: 1})
                saver.save(sess, "D:/Code/BeiLu/Water/data/summerGF/ckpt/model.ckpt", global_step=step)
                summary_writer.add_summary(s_m, global_step=step)
                if step % 100 == 0:
                    print(
                        "Step:{:>5}, Train Loss:{:>7.4f}, Train Accuracy:{:>7.2%}, Learning Rate：{:>7.5}".format(step,
                                                                                                                 loss_,
                                                                                                                 acu,
                                                                                                                 lr
                                                                                                                 )
                    )
                step += 1
        print("Training Finished, Loss:{:>7.4f}, Accuracy:{:>7.2%}".format(loss_, acu))
        summary_writer.close()
        #
        # test_step = 0
        #
        #
        # test_loss, test_acu = sess.run([prediction],
        #                                {inputs: test_data[128,:,:,:],  keep_prob: 1})
        # print("Step:{:>5}, Vail Loss:{:>7.4f}, Vail Accuracy:{:>7.2%}".format(test_step, test_loss, test_acu))
        # test_step += 1
        print("Test Finished!")

    else:
        start = time.clock()
        checkpoint = tf.train.latest_checkpoint('D:/Code/BeiLu/Water/data/summerGF/ckpt')
        saver = tf.train.import_meta_graph("D:/Code/BeiLu/Water/data/summerGF/ckpt/model.ckpt-5099.meta")
        saver.restore(sess, checkpoint)
        inputs = sess.graph.get_tensor_by_name('inputs:0')
        keep_prob = sess.graph.get_tensor_by_name('keep_prob:0')
        prediction = sess.graph.get_tensor_by_name('prediction:0')
        class_image = np.zeros_like(SLIC_data, dtype=np.int32)
        SLIc_image = np.load("D:/Code/BeiLu/Water/data/summerGF/image_data.npy")
        for i in trange(np.max(SLIC_data) + 1):
            # position = np.where(SLIC_data == i)
            # if len(position[0]) >= 144:
            #     cluster = np.zeros([144, 4], dtype=np.float32)
            #     cluster[:, :] = img[position[0][0:144], position[1][0:144], :]
            #     cluster = np.reshape(cluster, (1,12, 12, 4))
            # else:
            #     cluster = np.zeros([144, 4], dtype=np.float32)
            #     cluster[0:len(position[0]), :] = img[position[0], position[1], :]
            #     cluster = np.reshape(cluster, (1,12, 12, 4))
            cluster = np.zeros([1, 12, 12, 4], dtype=np.float32)
            cluster[:, :, :, :] = SLIc_image[i, :, :, :]
            prediction_value = sess.run(prediction,
                                            {inputs: cluster, keep_prob: 1})
            print(prediction_value)
            class_image[SLIC_data == i] = prediction_value
        rw.write_img('D:/Code/BeiLu/Water/data/summerGF/class.tif', im_proj, im_geotrans, class_image)
        elapsed = (time.clock() - start)
        print(("Time use: {} Prediction Finished!").format(elapsed))
