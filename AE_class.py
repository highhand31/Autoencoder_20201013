import cv2,sys,shutil,os,json,time,math
import numpy as np
import tensorflow
if tensorflow.__version__.startswith('1.'):
    import tensorflow as tf
else:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
print(tf.__version__)
sys.path.append(r'G:\我的雲端硬碟\Python\Code\Pycharm\utility')
from Utility import get_4D_data
from models_AE import AE_transpose_4layer,tf_mish,AE_transpose_4layer_2,\
    AE_incep_resnet_v1,AE_transpose_4layer_noPool,AE_Resnet_Rot,AE_pooling_net

img_format = {"jpg", 'png', 'bmp', 'JPG','tif','TIF'}

class AE():
    def __init__(self,para_dict):
        #----var parsing
        train_img_source = para_dict['train_img_source']
        recon_img_dir = para_dict['recon_img_dir']

        #----local var
        recon_flag = False


        #----get train image paths
        train_paths = self.get_paths(train_img_source)

        if recon_img_dir is not None:
            recon_paths = self.get_paths(recon_img_dir)
            if len(recon_paths) > 0:
                self.recon_paths = recon_paths
                recon_flag = True
            else:
                print("recon img dir:{} has no files".format(recon_img_dir))

        #----log update
        content = dict()
        content = self.log_update(content, para_dict)

        #----local var to global
        self.train_paths = train_paths
        self.print_out = True
        self.content = content
        self.recon_flag = recon_flag
        self.recon_img_dir = recon_img_dir

    def model_init(self,para_dict):
        #----var parsing
        model_shape = para_dict['model_shape']  # [N,H,W,C]
        infer_method = para_dict['infer_method']
        activation = para_dict['activation']
        kernel_list = para_dict['kernel_list']
        filter_list = para_dict['filter_list']
        conv_time = para_dict['conv_time']
        maxpool_kernel = para_dict['maxpool_kernel']
        loss_method = para_dict['loss_method']
        opti_method = para_dict['opti_method']
        lr = para_dict['learning_rate']
        save_dir = para_dict['save_dir']
        embed_length = para_dict['embed_length']
        scaler = para_dict['scaler']


        # ----tf_placeholder declaration
        tf_input = tf.placeholder(tf.float32, shape=model_shape, name='input')
        tf_keep_prob = tf.placeholder(dtype=np.float32, name="keep_prob")
        #tf_label_batch = tf.placeholder(dtype=tf.int32, shape=[None], name="label_batch")
        #tf_phase_train = tf.placeholder(tf.bool, name="phase_train")

        # ----activation selection
        if activation == 'relu':
            acti_func = tf.nn.relu
        elif activation == 'mish':
            acti_func = tf_mish
        else:
            acti_func = tf.nn.relu

        avepool_out = self.__avepool(tf_input, k_size=5, strides=1)

        #----inference selection
        if infer_method == "AE_transpose_4layer":
            recon = AE_transpose_4layer(tf_input, kernel_list, filter_list,activation=acti_func,
                                              maxpool_kernel=maxpool_kernel)
        elif infer_method == "AE_transpose_4layer_noPool":
            recon = AE_transpose_4layer_noPool(tf_input, kernel_list, filter_list,activation=acti_func,
                                              maxpool_kernel=maxpool_kernel)
        elif infer_method == "test":
            recon = self.__AE_transpose_4layer_test(tf_input, kernel_list, filter_list,
                                               conv_time=conv_time,maxpool_kernel=maxpool_kernel)
        elif infer_method == 'inception_resnet_v1_reduction':
            recon = AE_incep_resnet_v1(tf_input=tf_input,tf_keep_prob=tf_keep_prob,embed_length=embed_length,
                                       scaler=scaler,kernel_list=kernel_list,filter_list=filter_list,
                                       activation=acti_func,)
        elif infer_method == "Resnet_Rot":
            filter_list = [12, 16, 24, 36, 48, 196]
            recon = AE_Resnet_Rot(tf_input,filter_list,tf_keep_prob,embed_length,activation=acti_func,
                                  print_out=True,rot=True)

        #----loss method selection
        if loss_method == 'mse':
            loss_AE = tf.reduce_mean(tf.pow(recon - tf_input,2), name="loss_AE")
        elif loss_method =='ssim':
            # self.loss_AE = tf.reduce_mean(tf.image.ssim_multiscale(tf.image.rgb_to_grayscale(self.tf_input),tf.image.rgb_to_grayscale(self.recon),2),name='loss')
            loss_AE = tf.reduce_mean(tf.image.ssim(tf_input,recon, 2.0),name='loss_AE')
        elif loss_method == "huber":
            loss_AE = tf.reduce_sum(tf.losses.huber_loss(tf_input, recon, delta=1.35),name='loss_AE')
        elif loss_method == 'ssim+mse':
            loss_1 = tf.reduce_mean(tf.pow(recon - tf_input, 2))
            loss_2 = tf.reduce_mean(tf.image.ssim(tf_input, recon, 2.0))
            loss_AE = tf.subtract(loss_2,loss_1,name='loss_AE')
        elif loss_method == 'cross_entropy':
            loss_AE = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.layers.flatten(tf_input),
                                                           logits=tf.layers.flatten(recon)),name="loss_AE")
        elif loss_method == 'kl_d':
            epsilon = 1e-8
            # generation loss(cross entropy)
            loss_AE = tf.reduce_mean(
                tf_input * tf.subtract(tf.log(epsilon + tf_input), tf.log(epsilon + recon)) , name='loss_AE')

        #----optimizer selection
        if opti_method == "adam":
            if loss_method.find('ssim') >= 0:
                opt_AE = tf.train.AdamOptimizer(learning_rate=lr).minimize(-loss_AE)
            else:
                opt_AE = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_AE)

        # ----create the dir to save model weights(CKPT, PB)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)


        if self.recon_flag is True:
            new_recon_dir = "img_recon-" + os.path.basename(save_dir)
            self.new_recon_dir = os.path.join(self.recon_img_dir, new_recon_dir)
            if not os.path.exists(self.new_recon_dir):
                os.makedirs(self.new_recon_dir)

        out_dir_prefix = os.path.join(save_dir, "model")
        saver = tf.train.Saver(max_to_keep=2)

        # ----appoint PB node names
        pb_save_path = os.path.join(save_dir, "pb_model.pb")
        if activation == 'mish':
            node_name = 'output_AE/mul'
        else:
            node_name = 'output_AE/Relu'
        pb_save_list = [node_name, "embeddings","loss_AE"]

        # ----create the log(JSON)
        count = 0
        for i in range(100):
            log_path = os.path.join(save_dir, "train_result_" + str(count) + ".json")
            if not os.path.exists(log_path):
                break
            count += 1
        print("log_path: ", log_path)
        self.content = self.log_update(self.content, para_dict)

        # ----local var to global
        self.model_shape = model_shape
        self.tf_input = tf_input
        self.tf_keep_prob = tf_keep_prob
        self.avepool_out = avepool_out
        self.recon = recon
        self.loss_AE = loss_AE
        self.opt_AE = opt_AE
        self.out_dir_prefix = out_dir_prefix
        self.saver = saver
        self.save_dir = save_dir
        self.pb_save_path = pb_save_path
        self.pb_save_list = pb_save_list
        self.log_path = log_path

    def train(self,para_dict):
        # ----var parsing
        epochs = para_dict['epochs']
        GPU_ratio = para_dict['GPU_ratio']
        batch_size = para_dict['batch_size']
        # ratio = para_dict['ratio']
        process_dict = para_dict['process_dict']
        eval_epochs = para_dict['eval_epochs']
        setting_dict = None
        if 'setting_dict' in para_dict.keys():
            if para_dict['setting_dict'] is not None and len(para_dict['setting_dict']) > 0:
                setting_dict = para_dict['setting_dict']

        # ----local var
        train_loss_list = list()
        train_acc_list = list()
        test_loss_list = list()
        test_acc_list = list()
        epoch_time_list = list()
        img_quantity = 0
        aug_enable = False
        ratio = 1.0

        self.content = self.log_update(self.content, para_dict)

        # ----ratio
        if ratio <= 1.0:
            img_quantity = int(self.train_paths.shape[0] * ratio)
        else:
            img_quantity = self.train_paths.shape[0]

        # ----check if the augmentation(image processing) is enabled
        if isinstance(process_dict, dict):
            if True in process_dict.values():
                aug_enable = True
                batch_size = batch_size // 2  # the batch size must be integer!!

        # ----calculate iterations of one epoch
        train_ites = math.ceil(img_quantity / batch_size)
        # if self.test_img_dir is not None:
        #     test_ites = math.ceil(self.test_paths.shape[0] / batch_size)

        # ----GPU setting
        config = tf.ConfigProto(log_device_placement=False,
                                allow_soft_placement=True)
        if GPU_ratio is None:
            config.gpu_options.allow_growth = True
        else:
            config.gpu_options.per_process_gpu_memory_fraction = GPU_ratio

        with tf.Session(config=config) as sess:
            # ----tranfer learning check
            files = [file.path for file in os.scandir(self.save_dir) if file.name.split(".")[-1] == 'meta']
            if len(files) == 0:
                sess.run(tf.global_variables_initializer())
                print("no previous model param can be used!")
            else:
                check_name = files[-1].split("\\")[-1].split(".")[0]
                model_path = os.path.join(self.save_dir, check_name)
                self.saver.restore(sess, model_path)
                msg = "use previous model param:{}".format(model_path)
                print(msg)

            # ----info display
            print("img_quantity:", img_quantity)
            if aug_enable is True:
                print("aug_enable is True, the data quantity of one epoch is doubled")

            # ----epoch training
            for epoch in range(epochs):
                # ----record the start time
                d_t = time.time()

                train_loss = 0
                train_acc = 0
                test_loss = 0
                test_acc = 0

                # ----shuffle
                indice = np.random.permutation(self.train_paths.shape[0])
                self.train_paths = self.train_paths[indice]
                #self.train_labels = self.train_labels[indice]
                train_paths_ori = self.train_paths[:img_quantity]
                #train_labels_ori = self.train_labels[:img_quantity]

                if aug_enable is True:
                    train_paths_aug = train_paths_ori[::-1]
                    #train_labels_aug = train_labels_ori[::-1]

                # ----do optimizers(training by iteration)
                for index in range(train_ites):
                    # ----get image start and end numbers
                    num_start = index * batch_size
                    num_end = np.minimum(num_start + batch_size, self.train_paths.shape[0])

                    # ----get 4-D data
                    if aug_enable is True:
                        # ----ori data
                        ori_data = get_4D_data(train_paths_ori[num_start:num_end], self.model_shape[1:],
                                                    process_dict=None)
                        #ori_labels = train_labels_ori[num_start:num_end]
                        # ----aug data
                        aug_data = get_4D_data(train_paths_aug[num_start:num_end], self.model_shape[1:],
                                                    process_dict=process_dict,setting_dict=setting_dict)
                        #aug_labels = train_labels_aug[num_start:num_end]
                        # ----data concat
                        batch_data = np.concatenate([ori_data, aug_data], axis=0)
                        #batch_labels = np.concatenate([ori_labels, aug_labels], axis=0)

                    else:
                        batch_data = get_4D_data(train_paths_ori[num_start:num_end], self.model_shape[1:])
                        #batch_labels = train_labels_ori[num_start:num_end]

                    # ----put all data to tf placeholders
                    feed_dict = {self.tf_input: batch_data,self.tf_keep_prob:0.5}

                    # ----session run
                    sess.run(self.opt_AE, feed_dict=feed_dict)

                    # ----evaluation(training set)
                    feed_dict[self.tf_keep_prob] = 1.0
                    # feed_dict[self.tf_phase_train] = False
                    loss_temp = sess.run(self.loss_AE, feed_dict=feed_dict)

                    # ----calculate the loss and accuracy
                    train_loss += loss_temp
                    #train_acc += self.evaluation(predict_temp, batch_labels)

                train_loss /= train_ites
                #train_acc /= self.train_paths.shape[0]

                # ----save ckpt, pb files
                if (epoch + 1) % 2 == 0 or (epoch + 1) == epochs:
                    model_save_path = self.saver.save(sess, self.out_dir_prefix, global_step=epoch)
                    print("save model CKPT to ", model_save_path)

                    graph = tf.get_default_graph().as_graph_def()
                    output_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph, self.pb_save_list)
                    with tf.gfile.GFile(self.pb_save_path, 'wb')as f:
                        f.write(output_graph_def.SerializeToString())
                    print("save PB file to ", self.pb_save_path)

                # ----record the end time
                d_t = time.time() - d_t

                # ----save results in the log file
                train_loss_list.append(float(train_loss))
                #train_acc_list.append(float(train_acc))
                # if self.test_img_dir is not None:
                #     # test_loss_list.append(float(test_loss))
                #     test_acc_list.append(float(test_acc))

                self.content["train_loss_list"] = train_loss_list
                #self.content["train_acc_list"] = train_acc_list
                # if self.test_img_dir is not None:
                #     # self.content["test_loss_list"] = test_loss_list
                #     self.content["test_acc_list"] = test_acc_list

                epoch_time_list.append(d_t)
                self.content['ave_epoch_time'] = float(np.average(epoch_time_list))

                with open(self.log_path, 'w') as f:
                    json.dump(self.content, f)

                print("save the log file in ", self.log_path)

                # ----display training results
                print("Epoch: ", epoch)
                print("training loss:{}".format(train_loss))
                # if self.test_img_dir is not None:
                #     print("test set accuracy:{}".format(test_acc))

                print("Epoch time consumption:", d_t)

                # ----test image reconstruction
                if self.recon_flag is True:
                    # if (epoch + 1) % eval_epochs == 0 and train_loss > 0.80:
                    if (epoch + 1) % eval_epochs == 0 :
                        for filename in self.recon_paths:
                            test_img = self.__img_read(filename, self.model_shape[1:])
                            # ----session run
                            img_sess_out = sess.run(self.recon, feed_dict={self.tf_input: test_img,
                                                                           self.tf_keep_prob:1.0})
                            # ----process of sess-out
                            img_sess_out = img_sess_out[0] * 255
                            img_sess_out = cv2.convertScaleAbs(img_sess_out)
                            if self.model_shape[3] == 1:
                                img_sess_out = np.reshape(img_sess_out, (self.model_shape[1], self.model_shape[2]))
                            else:
                                img_sess_out = cv2.cvtColor(img_sess_out, cv2.COLOR_RGB2BGR)

                            # if loss_method != 'ssim':
                            #     img_sess_out = cv2.cvtColor(img_sess_out, cv2.COLOR_RGB2BGR)

                            # ----save recon image
                            splits = filename.split("\\")[-1]
                            new_filename = splits.split('.')[0] + '_sess-out.' + splits.split('.')[-1]

                            new_filename = os.path.join(self.new_recon_dir, new_filename)
                            # cv2.imwrite(new_filename, img_sess_out)
                            cv2.imencode('.'+splits.split('.')[-1], img_sess_out)[1].tofile(new_filename)
                            # ----img diff method
                            img_diff = self.__img_diff_method(filename, img_sess_out, diff_th=15, cc_th=15)
                            img_diff = cv2.convertScaleAbs(img_diff)
                            new_filename = filename.split("\\")[-1]
                            new_filename = new_filename.split(".")[0] + '_diff.' + new_filename.split(".")[-1]
                            new_filename = os.path.join(self.new_recon_dir, new_filename)
                            # cv2.imwrite(new_filename, img_diff)
                            cv2.imencode('.' + splits.split('.')[-1], img_diff)[1].tofile(new_filename)

                            # ----img avepool diff method
                            img_diff = self.__img_patch_diff_method(filename, img_sess_out, sess, diff_th=30, cc_th=15)
                            img_diff = cv2.convertScaleAbs(img_diff)
                            new_filename = filename.split("\\")[-1]
                            new_filename = new_filename.split(".")[0] + '_avepool_diff.' + new_filename.split(".")[-1]
                            new_filename = os.path.join(self.new_recon_dir, new_filename)
                            # cv2.imwrite(new_filename, img_diff)
                            cv2.imencode('.' + splits.split('.')[-1], img_diff)[1].tofile(new_filename)

                            # ----SSIM method
                            # img_ssim = self.__ssim_method(filename, img_sess_out)
                            # img_ssim = cv2.convertScaleAbs(img_ssim)
                            # new_filename = filename.split("\\")[-1]
                            # new_filename = new_filename.split(".")[0] + '_ssim.' + new_filename.split(".")[-1]
                            # new_filename = os.path.join(self.new_recon_dir, new_filename)
                            # cv2.imwrite(new_filename, img_ssim)

    #----functions
    def __img_patch_diff_method(self,img_source_1, img_source_2, sess,diff_th=30, cc_th=30):

        temp = np.array([1., 2., 3.])
        re = None
        # ----read img source 1
        if isinstance(temp, type(img_source_1)):
            img_1 = img_source_1
        elif os.path.isfile(img_source_1):
            # img_1 = cv2.imread(img_source_1)
            img_1 = cv2.imdecode(np.fromfile(img_source_1, dtype=np.uint8), 1)
            img_1 = cv2.resize(img_1,(self.model_shape[2],self.model_shape[1]))
            # img_1 = img_1.astype('float32')

        else:
            print("The type of img_source_1 is not supported")

        # ----read img source 2
        if isinstance(temp, type(img_source_2)):
            img_2 = img_source_2
        elif os.path.isfile(img_source_2):
            # img_2 = cv2.imread(img_source_2)
            img_2 = cv2.imdecode(np.fromfile(img_source_2, dtype=np.uint8), 1)
            # img_2 = img_2.astype('float32')
        else:
            print("The type of img_source_2 is not supported")

        # ----subtraction
        if img_1 is not None and img_2 is not None:
            img_1_ave_pool = sess.run(self.avepool_out,feed_dict={self.tf_input:np.expand_dims(img_1,axis=0)})
            img_2_ave_pool = sess.run(self.avepool_out,feed_dict={self.tf_input:np.expand_dims(img_2,axis=0)})
            # print("img_1 shape = {}, dtype = {}".format(img_1.shape, img_1.dtype))
            # print("img_2 shape = {}, dtype = {}".format(img_2.shape, img_2.dtype))
            img_diff = cv2.absdiff(img_1_ave_pool[0], img_2_ave_pool[0])  # img_diff = np.abs(img - img_sess)#不能使用，因為相減若<0，會取補數
            if img_1.shape[-1] == 3:
                img_diff = cv2.cvtColor(img_diff, cv2.COLOR_BGR2GRAY)  # 利用轉成灰階，減少RGB的相加計算量

            # 連通
            img_compare = cv2.compare(img_diff, diff_th, cv2.CMP_GT)
            retval, labels = cv2.connectedComponents(img_compare)
            max_label_num = np.max(labels) + 1

            img_1_copy = img_1.copy()
            for i in range(0, max_label_num):  # label = 0是背景，所以從1開始
                y, x = np.where(labels == i)  # y,x代表類別=i的像素座標值
                if (y.shape[0] > cc_th) and (img_compare[y[0], x[0]] == 255):
                    for j in range(y.shape[0]):
                        img_1_copy.itemset((y[j], x[j], 0), 0)
                        img_1_copy.itemset((y[j], x[j], 1), 0)
                        img_1_copy.itemset((y[j], x[j], 2), 255)

            re = img_1_copy
            return re

    def __img_diff_method(self,img_source_1, img_source_2, diff_th=30, cc_th=30):

        temp = np.array([1., 2., 3.])
        re = None
        # ----read img source 1
        if isinstance(temp, type(img_source_1)):
            img_1 = img_source_1
        elif os.path.isfile(img_source_1):
            # img_1 = cv2.imread(img_source_1)
            img_1 = cv2.imdecode(np.fromfile(img_source_1, dtype=np.uint8), 1)
            img_1 = cv2.resize(img_1,(self.model_shape[2],self.model_shape[1]))
            # img_1 = img_1.astype('float32')

        else:
            print("The type of img_source_1 is not supported")

        # ----read img source 2
        if isinstance(temp, type(img_source_2)):
            img_2 = img_source_2
        elif os.path.isfile(img_source_2):
            # img_2 = cv2.imread(img_source_2)
            img_2 = cv2.imdecode(np.fromfile(img_source_2, dtype=np.uint8), 1)
            # img_2 = img_2.astype('float32')
        else:
            print("The type of img_source_2 is not supported")

        # ----subtraction
        if img_1 is not None and img_2 is not None:
            # print("img_1 shape = {}, dtype = {}".format(img_1.shape, img_1.dtype))
            # print("img_2 shape = {}, dtype = {}".format(img_2.shape, img_2.dtype))
            img_diff = cv2.absdiff(img_1, img_2)  # img_diff = np.abs(img - img_sess)#不能使用，因為相減若<0，會取補數
            if img_1.shape[-1] == 3:
                img_diff = cv2.cvtColor(img_diff, cv2.COLOR_BGR2GRAY)  # 利用轉成灰階，減少RGB的相加計算量

            # 連通
            img_compare = cv2.compare(img_diff, diff_th, cv2.CMP_GT)
            retval, labels = cv2.connectedComponents(img_compare)
            max_label_num = np.max(labels) + 1

            img_1_copy = img_1.copy()
            for i in range(0, max_label_num):  # label = 0是背景，所以從1開始
                y, x = np.where(labels == i)  # y,x代表類別=i的像素座標值
                if (y.shape[0] > cc_th) and (img_compare[y[0], x[0]] == 255):
                    for j in range(y.shape[0]):
                        img_1_copy.itemset((y[j], x[j], 0), 0)
                        img_1_copy.itemset((y[j], x[j], 1), 0)
                        img_1_copy.itemset((y[j], x[j], 2), 255)

            re = img_1_copy
            return re

    def __avepool(self,input_x,k_size=3,strides=1):
        kernel = [1,k_size,k_size,1]
        stride_kernel = [1,strides,strides,1]
        return tf.nn.avg_pool(input_x, ksize=kernel, strides=stride_kernel, padding='SAME')

    def get_paths(self,train_img_source):
        #----var
        paths = list()

        if isinstance(train_img_source,list):
            for img_dir in train_img_source:
                temp = [file.path for file in os.scandir(img_dir) if file.name[-3:] in img_format]
                if len(temp) == 0:
                    print("No files in ",img_dir)
                else:
                    paths.extend(temp)
        elif os.path.exists(train_img_source):
            temp = [file.path for file in os.scandir(train_img_source) if file.name[-3:] in img_format]
            if len(temp) == 0:
                print("No files in ",train_img_source)
            else:
                paths.extend(temp)

        print("length of path: ",len(paths))

        return np.array(paths)

    def __Conv(self,input_x,kernel=[3,3],filter=32,conv_times=2,stride=1):
        net = None
        for i in range(conv_times):
            if i == 0:
                net = tf.layers.conv2d(
                    inputs=input_x,
                    filters=filter,
                    kernel_size=kernel,
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                    strides=stride,
                    padding="same",
                    activation=tf.nn.relu)
            else:
                net = tf.layers.conv2d(
                    inputs=net,
                    filters=filter,
                    kernel_size=kernel,
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                    strides=stride,
                    padding="same",
                    activation=tf.nn.relu)
        return net

    def say_sth(self,msg, print_out=False):
        if print_out:
            print(msg)

    def log_update(self,content,para_dict):
        for key, value in para_dict.items():
            content[key] = value

        return content

    # def get_4D_data(self,paths, output_shape, process_dict=None):
    #     # ----var
    #     len_path = len(paths)
    #     processing_enable = False
    #     y_range = 5
    #     x_range = 5
    #     flip_list = [1, 0, -1, 2]
    #
    #     # ----create default np array
    #     batch_dim = [len_path]
    #     batch_dim.extend(output_shape)
    #     batch_data = np.zeros(batch_dim, dtype=np.float32)
    #
    #     # ----check process_dict
    #     if isinstance(process_dict, dict):
    #         if len(process_dict) > 0:
    #             processing_enable = True  # image processing is enabled
    #             if 'rdm_crop' in process_dict.keys():
    #                 x_start = np.random.randint(x_range, size=len_path)
    #                 y_start = np.random.randint(y_range, size=len_path)
    #
    #     for idx, path in enumerate(paths):
    #         img = cv2.imread(path)
    #         if img is None:
    #             print("read failed:", path)
    #         else:
    #             # ----image processing
    #             if processing_enable is True:
    #                 if 'rdm_crop' in process_dict.keys():
    #                     if process_dict['rdm_crop'] is True:
    #                         # ----From the random point, crop the image
    #                         img = img[y_start[idx]:, x_start[idx]:, :]
    #                 if 'rdm_br' in process_dict.keys():
    #                     if process_dict['rdm_br'] is True:
    #                         mean_br = np.mean(img)
    #                         br_factor = np.random.randint(mean_br * 0.9, mean_br * 1.1)
    #                         img = np.clip(img / mean_br * br_factor, 0, 255)
    #                         img = img.astype(np.uint8)
    #                 if 'rdm_flip' in process_dict.keys():
    #                     if process_dict['rdm_flip'] is True:
    #                         flip_type = np.random.choice(flip_list)
    #                         if flip_type != 2:
    #                             img = cv2.flip(img, flip_type)
    #                 if 'rdm_noise' in process_dict.keys():
    #                     if process_dict['rdm_noise'] is True:
    #                         uniform_noise = np.empty((img.shape[0], img.shape[1]), dtype=np.uint8)
    #                         cv2.randu(uniform_noise, 0, 255)
    #                         ret, impulse_noise = cv2.threshold(uniform_noise, 230, 255, cv2.THRESH_BINARY_INV)
    #                         img = cv2.bitwise_and(img, img, mask=impulse_noise)
    #                 if 'rdm_angle' in process_dict.keys():
    #                     if process_dict['rdm_angle'] is True:
    #                         angle = np.random.randint(-15, 15)
    #                         h, w = img.shape[:2]
    #                         M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    #                         img = cv2.warpAffine(img, M, (w, h))
    #
    #             # ----resize and change the color format
    #             img = cv2.resize(img, (output_shape[1], output_shape[0]))
    #             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #             batch_data[idx] = img
    #
    #     return batch_data / 255

    def __img_read(self, img_path, shape):

        # img = cv2.imread(img_path)
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 1)
        if img is None:
            print("Read failed:",img_path)
            return None
        else:
            img = cv2.resize(img,(shape[1],shape[0]))
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32)
            img /= 255

            return np.expand_dims(img,axis=0)


    #----models
    def __AE_transpose_4layer(self, input_x, kernel_list, filter_list,conv_time=1,maxpool_kernel=2):
        #----var
        maxpool_kernel = [1,maxpool_kernel,maxpool_kernel,1]
        transpose_filter = [1, 1]

        msg = '----AE_transpose_4layer_struct_2----'
        self.say_sth(msg, print_out=self.print_out)

        net = self.__Conv(input_x, kernel=kernel_list[0], filter=filter_list[0], conv_times=conv_time)
        U_1_point = net
        net = tf.nn.max_pool(net, ksize=maxpool_kernel, strides=[1, 2, 2, 1], padding='SAME')

        msg = "encode_1 shape = {}".format(net.shape)
        self.say_sth(msg, print_out=self.print_out)
        # -----------------------------------------------------------------------
        net = self.__Conv(net, kernel=kernel_list[1], filter=filter_list[1], conv_times=conv_time)
        U_2_point = net
        net = tf.nn.max_pool(net, ksize=maxpool_kernel, strides=[1, 2, 2, 1], padding='SAME')
        msg = "encode_2 shape = {}".format(net.shape)
        self.say_sth(msg, print_out=self.print_out)
        # -----------------------------------------------------------------------

        net = self.__Conv(net, kernel=kernel_list[2], filter=filter_list[2], conv_times=conv_time)
        U_3_point = net
        net = tf.nn.max_pool(net, ksize=maxpool_kernel, strides=[1, 2, 2, 1], padding='SAME')

        msg = "encode_3 shape = {}".format(net.shape)
        self.say_sth(msg, print_out=self.print_out)
        # -----------------------------------------------------------------------
        net = self.__Conv(net, kernel=kernel_list[3], filter=filter_list[3], conv_times=conv_time)
        U_4_point = net
        net = tf.nn.max_pool(net, ksize=maxpool_kernel, strides=[1, 2, 2, 1], padding='SAME')

        msg = "encode_4 shape = {}".format(net.shape)
        self.say_sth(msg, print_out=self.print_out)

        net = self.__Conv(net, kernel=kernel_list[4], filter=filter_list[4], conv_times=conv_time)


        flatten = tf.layers.flatten(net)

        # ----dropout
        net = tf.nn.dropout(net, keep_prob=0.8)

        # ----FC
        net = tf.layers.dense(inputs=net, units=128, activation=tf.nn.relu)
        print("FC shape:", net.shape)


        embeddings = tf.nn.l2_normalize(flatten, 1, 1e-10, name='embeddings')
        print("embeddings shape:",embeddings.shape)
        # net = tf.layers.dense(inputs=prelogits, units=units, activation=None)
        # print("net shape:",net.shape)
        # net = tf.reshape(net,shape)
        # -----------------------------------------------------------------------
        # --------Decode--------
        # -----------------------------------------------------------------------

        # data= 4 x 4 x 64

        net = tf.layers.conv2d_transpose(net, filter_list[3], transpose_filter, strides=2, padding='same')
        #net = tf.concat([net, U_4_point], axis=3)
        net = self.__Conv(net, kernel=kernel_list[3], filter=filter_list[3], conv_times=conv_time)
        msg = "decode_1 shape = {}".format(net.shape)
        self.say_sth(msg, print_out=self.print_out)
        # -----------------------------------------------------------------------
        # data= 8 x 8 x 64
        net = tf.layers.conv2d_transpose(net, filter_list[2], transpose_filter, strides=2, padding='same')
        # net = tf.concat([net, U_3_point], axis=3)
        net = self.__Conv(net, kernel=kernel_list[2], filter=filter_list[2], conv_times=conv_time)
        msg = "decode_2 shape = {}".format(net.shape)
        self.say_sth(msg, print_out=self.print_out)
        # -----------------------------------------------------------------------

        net = tf.layers.conv2d_transpose(net, filter_list[1], transpose_filter, strides=2, padding='same')
        # net = tf.concat([net, U_2_point], axis=3)
        net = self.__Conv(net, kernel=kernel_list[1], filter=filter_list[1], conv_times=conv_time)
        msg = "decode_3 shape = {}".format(net.shape)
        self.say_sth(msg, print_out=self.print_out)
        # -----------------------------------------------------------------------
        # data= 32 x 32 x 64

        net = tf.layers.conv2d_transpose(net, filter_list[0], transpose_filter, strides=2, padding='same')
        # net = tf.concat([net, U_1_point], axis=3)
        net = self.__Conv(net, kernel=kernel_list[0], filter=filter_list[0], conv_times=conv_time)
        msg = "decode_2 shape = {}".format(net.shape)
        self.say_sth(msg, print_out=self.print_out)

        net = tf.layers.conv2d(
            inputs=net,
            filters=3,
            kernel_size=kernel_list[0],
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
            padding="same",
            activation=tf.nn.relu,
            name='output_AE')
        msg = "output shape = {}".format(net.shape)
        self.say_sth(msg, print_out=self.print_out)
        # -----------------------------------------------------------------------
        # data= 64 x 64 x 3
        return net
    def __AE_transpose_4layer_test(self, input_x, kernel_list, filter_list,conv_time=1,maxpool_kernel=2):
        #----var
        maxpool_kernel = [1,maxpool_kernel,maxpool_kernel,1]
        transpose_filter = [1, 1]

        msg = '----AE_transpose_4layer_struct_2----'
        self.say_sth(msg, print_out=self.print_out)

        net = self.__Conv(input_x, kernel=kernel_list[0], filter=filter_list[0], conv_times=conv_time)
        U_1_point = net
        #net = tf.nn.max_pool(net, ksize=maxpool_kernel, strides=[1, 2, 2, 1], padding='SAME')
        net = tf.layers.max_pooling2d(net,pool_size=[2,2],strides=2,padding='SAME')

        msg = "encode_1 shape = {}".format(net.shape)
        self.say_sth(msg, print_out=self.print_out)
        # -----------------------------------------------------------------------
        net = self.__Conv(net, kernel=kernel_list[1], filter=filter_list[1], conv_times=conv_time)
        U_2_point = net
        # net = tf.nn.max_pool(net, ksize=maxpool_kernel, strides=[1, 2, 2, 1], padding='SAME')
        net = tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=2, padding='SAME')
        msg = "encode_2 shape = {}".format(net.shape)
        self.say_sth(msg, print_out=self.print_out)
        # -----------------------------------------------------------------------

        net = self.__Conv(net, kernel=kernel_list[2], filter=filter_list[2], conv_times=conv_time)
        U_3_point = net
        # net = tf.nn.max_pool(net, ksize=maxpool_kernel, strides=[1, 2, 2, 1], padding='SAME')
        net = tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=2, padding='SAME')

        msg = "encode_3 shape = {}".format(net.shape)
        self.say_sth(msg, print_out=self.print_out)
        # -----------------------------------------------------------------------
        net = self.__Conv(net, kernel=kernel_list[3], filter=filter_list[3], conv_times=conv_time)
        U_4_point = net
        # net = tf.nn.max_pool(net, ksize=maxpool_kernel, strides=[1, 2, 2, 1], padding='SAME')
        net = tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=2, padding='SAME')

        msg = "encode_4 shape = {}".format(net.shape)
        self.say_sth(msg, print_out=self.print_out)

        net = self.__Conv(net, kernel=kernel_list[4], filter=filter_list[4], conv_times=conv_time)


        flatten = tf.layers.flatten(net)

        embeddings = tf.nn.l2_normalize(flatten, 1, 1e-10, name='embeddings')
        print("embeddings shape:",embeddings.shape)
        # net = tf.layers.dense(inputs=prelogits, units=units, activation=None)
        # print("net shape:",net.shape)
        # net = tf.reshape(net,shape)
        # -----------------------------------------------------------------------
        # --------Decode--------
        # -----------------------------------------------------------------------

        # data= 4 x 4 x 64

        net = tf.layers.conv2d_transpose(net, filter_list[3], transpose_filter, strides=2, padding='same')
        #net = tf.concat([net, U_4_point], axis=3)
        net = self.__Conv(net, kernel=kernel_list[3], filter=filter_list[3], conv_times=conv_time)
        msg = "decode_1 shape = {}".format(net.shape)
        self.say_sth(msg, print_out=self.print_out)
        # -----------------------------------------------------------------------
        # data= 8 x 8 x 64
        net = tf.layers.conv2d_transpose(net, filter_list[2], transpose_filter, strides=2, padding='same')
        # net = tf.concat([net, U_3_point], axis=3)
        net = self.__Conv(net, kernel=kernel_list[2], filter=filter_list[2], conv_times=conv_time)
        msg = "decode_2 shape = {}".format(net.shape)
        self.say_sth(msg, print_out=self.print_out)
        # -----------------------------------------------------------------------

        net = tf.layers.conv2d_transpose(net, filter_list[1], transpose_filter, strides=2, padding='same')
        # net = tf.concat([net, U_2_point], axis=3)
        net = self.__Conv(net, kernel=kernel_list[1], filter=filter_list[1], conv_times=conv_time)
        msg = "decode_3 shape = {}".format(net.shape)
        self.say_sth(msg, print_out=self.print_out)
        # -----------------------------------------------------------------------
        # data= 32 x 32 x 64

        net = tf.layers.conv2d_transpose(net, filter_list[0], transpose_filter, strides=2, padding='same')
        # net = tf.concat([net, U_1_point], axis=3)
        net = self.__Conv(net, kernel=kernel_list[0], filter=filter_list[0], conv_times=conv_time)
        msg = "decode_2 shape = {}".format(net.shape)
        self.say_sth(msg, print_out=self.print_out)

        net = tf.layers.conv2d(
            inputs=net,
            filters=3,
            kernel_size=kernel_list[0],
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
            padding="same",
            activation=tf.nn.relu,
            name='output_AE')
        msg = "output shape = {}".format(net.shape)
        self.say_sth(msg, print_out=self.print_out)
        # -----------------------------------------------------------------------
        # data= 64 x 64 x 3
        return net

if __name__ == "__main__":
    para_dict = dict()
    dirs = list()
    root_dirs = [r"D:\dataset\optotech\test_img_2\train",
                # r"D:\dataset\optotech\CMIT_009IRC\009多分類data\enhance"
                 ]

    for root_dir in root_dirs:
        temp = [obj.path for obj in os.scandir(root_dir) if obj.is_dir()]
        dirs.extend(temp)
        # for obj in os.scandir(root_dir):
        #     if obj.is_dir():
        #         if obj.name.find("無分類") >= 0:
        #             pass
        #         else:
        #             dirs.append(obj.path)


    #----class init
    para_dict['train_img_source'] = dirs
    para_dict['recon_img_dir'] = r"D:\dataset\optotech\test_img_2\test\scratch"
    AE_train = AE(para_dict)

    #----model init
    para_dict['model_shape'] = [None, 96, 96, 3]
    para_dict['infer_method'] = "AE_transpose_4layer_noPool"#"AE_transpose_4layer"
    para_dict['kernel_list'] = [3,3,3,3,3]
    para_dict['filter_list'] = [32,64,96,128,72]
    para_dict['conv_time'] = 1
    para_dict['maxpool_kernel'] = 2
    para_dict['embed_length'] = 144
    para_dict['scaler'] = 1
    para_dict['maxpool_kernel'] = 2
    para_dict['activation'] = 'relu'
    para_dict['loss_method'] = "ssim"
    para_dict['opti_method'] = "adam"
    para_dict['learning_rate'] = 1e-4
    para_dict['save_dir'] = r"D:\code\model_saver\Opto_tech\AE_CMIT_009_6classes_192x192_Resnet_Rot"
    AE_train.model_init(para_dict)

    para_dict['epochs'] = 100
    #----train
    para_dict['eval_epochs'] = 2
    para_dict['GPU_ratio'] = None
    para_dict['batch_size'] = 32
    para_dict['ratio'] = 1.0
    para_dict['setting_dict'] = {'rdm_shift': 0.12, 'rdm_angle': 15}

    process_dict = {"rdm_flip": True, 'rdm_br': True, 'rdm_crop': False, 'rdm_blur': True,
                    'rdm_angle': True,
                    'rdm_noise': False,
                    'rdm_shift': True
                    }

    if True in process_dict.values():
        pass
    else:
        process_dict = None
    para_dict['process_dict'] = process_dict

    AE_train.train(para_dict)




