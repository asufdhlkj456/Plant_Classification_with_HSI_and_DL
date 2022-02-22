import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import os, random
import glob
import tensorflow as tf
from tensorflow.contrib.image import rotate
import cv2
import tensorflow.contrib.slim as slim
import scipy.io as sio
from func_for_v5 import *
from tqdm import tqdm
from scipy import misc
from net import *
from baseline_model import *
import time
from spectral import *
from sklearn.metrics import confusion_matrix ,accuracy_score,recall_score,f1_score,precision_score,cohen_kappa_score


""" 
%%%%%%%%% v5_1 %%%%%%
dataset 增加為6種
.mat檔改為直接從envi讀取
將from_generator 改為 from_tensor_slice --> 解決了一直重複讀取data的問題

加入PCA
def model_v2(木紋辨識的模型)
func_for_v5 的 crop 修改為同時剪RGB影像

def test --> 單張img done

將CE 改為用 focal loss
"""


#####################################################################################################################

class HSI(object):
    def __init__(self, args):
        self.args = args
        self.n_component = args.PCs
        self.nbofcub = args.nbofdata  # 每個cub取多少張data

        ## 不固定training dataset ##
        #self.ori_data_list = get_datalist(self.train_dir,"hdr")
        #self.total_data = self.nbofcub * len(self.ori_data_list)


        '''         ########  訓練 & 測試 資料夾  #######           '''
        ### raw
        # self.train_list = './data_new/new0702/train/uni_crop/raw_band30_txt/raw_band30_train_2.txt'
        # self.test_list = './data_new/new0702/train/uni_crop/raw_band30_txt/raw_band30_test_2.txt'

        ### raw_ROI
        # self.train_list = './data_new/new0702/train/uni_crop/raw_ROI_txt/raw_ROI_class_30_train_3.txt'
        # self.test_list = './data_new/new0702/train/uni_crop/raw_ROI_txt/raw_ROI_class_30_test_3.txt'

        ### raw_ROI  ## Uniflormly band selection
        #self.train_list = './data_new/new0702/train/uni_crop/band_selection/raw_ROI_band09_train_3.txt'
        #self.test_list = './data_new/new0702/train/uni_crop/band_selection/raw_ROI_band09_test_3.txt'

        ### pca
        # self.train_list = './data_new/new0702/train/uni_crop/pca_txt/pca_train_1.txt'
        # self.test_list = './data_new/new0702/train/uni_crop/pca_txt/pca_test_1.txt'

        # self.train_list = './data_new/new0702/train/uni_crop/pca_txt/pca_class_30_train_3.txt'
        # self.test_list = './data_new/new0702/train/uni_crop/pca_txt/pca_class_30_test_2.txt'

        ### pca after ROI
        #self.train_list = './data_new/new0702/train/uni_crop/pca_ROI_txt/pca_ROI_class_30_train_3.txt'
        #self.test_list = './data_new/new0702/train/uni_crop/pca_ROI_txt/pca_ROI_class_30_test_3.txt'


        ### rgb
        # self.train_list = './data_new/new0702/train/uni_crop/rgb_txt/rgb_train_2.txt'
        # self.test_list = './data_new/new0702/train/uni_crop/rgb_txt/rgb_test_2.txt'


        ### rgb_NIR     ## 6 bands
        # self.train_list = './data_new/new0702/train/uni_crop/rgbNIR_txt/rgbNIR_class_30_train_3.txt'
        # self.test_list = './data_new/new0702/train/uni_crop/rgbNIR_txt/rgbNIR_class_30_test_3.txt'

        ### NIR     ## 3 、 6 、 9bands
        # self.train_list = './data_new/new0702/train/uni_crop/NIR_txt/NIR_class_30_train_3.txt'
        # self.test_list = './data_new/new0702/train/uni_crop/NIR_txt/NIR_class_30_test_3.txt'

        # self.train_list = './data_new/new0702/train/uni_crop/NIR_6_bands_txt/NIR_6_bands_class_30_train_3.txt'
        # self.test_list = './data_new/new0702/train/uni_crop/NIR_6_bands_txt/NIR_6_bands_class_30_test_3.txt'

        # self.train_list = './data_new/new0702/train/uni_crop/NIR_9_bands_txt/NIR_9_bands_class_30_train_3.txt'
        # self.test_list = './data_new/new0702/train/uni_crop/NIR_9_bands_txt/NIR_9_bands_class_30_test_3.txt'

        ### OMP_9_bs
        # self.train_list = './data_new/new0702/train/uni_crop/omp_9_bs_txt/NIR_9_bands_class_30_train_2.txt'
        # self.test_list = './data_new/new0702/train/uni_crop/omp_9_bs_txt/NIR_9_bands_class_30_test_2.txt'
        #
        ###FNG_BS   ##3、6、9bands
        # self.train_list = 'J:/myproject/data_new/new0702/train/uni_crop/FNG_BS_txt/FNG_3bands_txt/FNG_3_bands_class_30_train_1.txt'
        # self.test_list = 'J:/myproject/data_new/new0702/train/uni_crop/FNG_BS_txt/FNG_3bands_txt/FNG_3_bands_class_30_test_1.txt'
        #
        ###Uniform BS
        # self.train_list = './data_new/new0702/train/uni_crop/band_selection/raw_ROI_band06_train_3.txt'
        # self.test_list = './data_new/new0702/train/uni_crop/band_selection/raw_ROI_band06_test_3.txt'
        #
        ###Average RGB
        # self.train_list = r'D:\deep_learning\data\FNG_BS_txt\FNG_60bands_txt\FNG_60_bands_class_30_train_1.txt'
        # self.test_list = r'D:\deep_learning\data\FNG_BS_txt\FNG_60bands_txt\FNG_60_bands_class_30_test_1.txt'

        ###OMP_BS
        self.train_list = 'D:\\deep_learning\\data\\OMP_BS_txt\\9bands\\OMP_9_bands_class_30_train_1.txt'
        self.test_list = 'D:\\deep_learning\\data\\OMP_BS_txt\\9bands\\OMP_9_bands_class_30_test_1.txt'
        ''' ########################################################## '''


        with open(self.train_list, 'r') as f:
            self.train_data_list = f.readlines()  # readlines --> out 已經是list

        self.total_data = len(self.train_data_list)

        self.band_select = args.num_bands

        self.crop_size = args.crop_size
        self.input_size = args.input_size

        ## Final version #30種
        self.dict_labels = {"Fittonia": 0,                "Carnosa": 1,                "Hoya_kerrii": 2,

                            "Ammocallis_rosea": 3,        "Peace_Lilies": 4,           "Zamioculcas": 5,

                            "Anyamanee": 6,               "Aglaonema": 7,              "Alocasia": 8,
                            "Hydrocotyle": 9,             "Guilfoyle":10,              "Scandent": 11,
                            "Sansevieria": 12,            "Dracaena": 13,              "Chlorophytum": 14,
                            "Begonia": 15,                "Bromeliads": 16,            "Clusia_rosea": 17,
                            "Davallia": 18,               "Myriophyllum": 19,          "Glechoma": 20,


                            "Plectranthus": 21,           "Variegata": 22,             "Pachira_aquatica": 23,
                            "Calathea_lancifolia": 24,    "Boston_Swordfern": 25,      "Spathoglottis": 26,
                            "Peperomia": 27,              "Podocarpaceae": 28,         "Aluminium_plant": 29
                            }


        ori_band_list = [i for i in range(161)]  ## 原始資料161個band
        deleted_band = [0, 1, 2, 3, 46, 47, 48, 49, 50, 156, 157, 158, 159, 160]  ##沒有用的band
        [ori_band_list.remove(cElement) for cElement in [ori_band_list[i] for i in deleted_band]]
        self.precessed_band = ori_band_list      ### 原始161-無用的(14) = 147個band

        self.NUM_CLASS = len(self.dict_labels)
        ################################
        self.checkpoint_dir = 'D:\\deep_learning\\test\\'  ## 儲存檢查點的資料夾，更換模型改最後面
        self.modelname = 'OMPBS'  ## checkpoint的儲存名稱

        ##################### 測試設定 ################################
        self.test_ckpt_dir = 'D:\\deep_learning\\test\\'
        self.test_model_step = args.test_step
        ################################################################


        #  self.step = 6                                   ## restore 的step

        self.epoch = args.epoch  # 先從程式裡設定
        self.batch_size = args.batch_size
        self.step_per_epoch = (self.total_data // self.batch_size)
        self.max_batch_num = int(self.step_per_epoch * self.epoch)

        self.lr = args.lr  # 先從程式裡設定
        self.decay_every_Nepoch = 5             #5

        self.Mode = args.Mode       ##Training or Testing

        if self.Mode:
            print('\033[33m ------------------------------------------', '\033[0m')
            print('\033[33m Now is Training Mode...', '\033[0m')
            print('\033[33m -----------------------', '\033[0m')
            # print('\033[33m PCA n_components:' + str(self.n_component), '\033[0m')
            print('\033[33m Total num of training data:' + str(self.total_data), '\033[0m')
            print('\033[33m Num fo class:' + str(self.NUM_CLASS), '\033[0m')
            print('\033[33m Crop Size:' + str(self.crop_size), '\033[0m')
            print('\033[33m Network input size:' + str(self.input_size), '\033[0m')
            print('\033[33m Initial learning rate:' + str(self.lr), '\033[0m')
            print('\033[33m Epoch:' + str(self.epoch), '\033[0m')
            print('\033[33m Step_per_epoch:' + str(self.step_per_epoch), '\033[0m')
            print('\033[33m Batch Size:' + str(self.batch_size), '\033[0m')
            print('\033[33m Max_batch_num(Last step):' + str(self.max_batch_num), '\033[0m')
            print('\033[33m ------------------------------------------', '\033[0m')

        else:
            print('\033[36m ------------------------------------------', '\033[0m')
            print('\033[36m Now is Testing Mode...', '\033[0m')
            print('\033[36m -----------------------', '\033[0m')
            print('\033[36m Load Model step:' + str(self.test_model_step), '\033[0m')
            # print('\033[36m PCA n_components:' + str(self.n_component), '\033[0m')
            print('\033[36m Num of class:' + str(self.NUM_CLASS), '\033[0m')
            print('\033[36m Crop Size:' + str(self.crop_size), '\033[0m')
            print('\033[36m Network input size:' + str(self.input_size), '\033[0m')
            print('\033[36m ------------------------------------------', '\033[0m')



    def preprocess(self, patch,label):  # patch 作resize ；label 作 one-hot


        patch = tf.image.random_flip_up_down(patch)       ##好像沒差
        patch = tf.image.random_flip_left_right(patch)

        H, W = self.input_size

        patch.set_shape([H,W,self.band_select])            # pca再開
        label = tf.one_hot(label, self.NUM_CLASS)
        return patch , label




    def train(self):

        NUM_CLASS = self.NUM_CLASS
        global_step = tf.Variable(initial_value=1, dtype=tf.int32, trainable=False)

        # lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars]) * 0.001
        '''#####     Training data  ####'''
        #隨機的
     #   train_dataset, train_label = self.creat_dataset()

        '''#############################'''

        # datalist = get_datalist('./data_new/new0702/train/cropped/raw_band60/', ".npy")
        # print(*datalist,sep='\n')

        def gener():
            datalist = self.train_data_list         ## 不知道為啥第二層def self還可以work

            random.shuffle(datalist)
            ne.set_num_threads(8)

            for path in datalist:
                path = path.strip('\n')
                train_data = np.load(path)

                ###  for PCA (PC1~6)
                # train_data = train_data[:,:,:1]                  #### PC1 -> :1 , PC1~2 -> :2 , PC1~3 -> :3 .......

                train_label = get_label(path,self.dict_labels)

                ### Resize
                # train_data = cv2.resize(train_data, self.input_size, interpolation=cv2.INTER_LINEAR)
                ### ramdom crop
                train_data = crop_img(train_data,self.input_size[0],self.input_size[1])

                yield train_data.astype(np.float32) , np.asarray(train_label).astype(np.uint8)


        ### 順序很重要 shuffle --> repeat --> batch
        ###  ver 1
        # train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label))
        # train_dataset = train_dataset.repeat().batch(self.batch_size)
        # 
        # iter = train_dataset.make_one_shot_iterator()
        # train_batch, label_batch = iter.get_next()

        ###  ver2
        train_dataset = tf.data.Dataset.from_generator(gener, (tf.float32, tf.uint8))
        train_dataset = train_dataset.map(self.preprocess, num_parallel_calls=8)            #經過map 要記得set shape
        #
        train_dataset = train_dataset.repeat().batch(self.batch_size)  ##用設定剛好epoch的有時候會出錯，不知為啥
        # train_dataset = train_dataset.prefetch(tf.contrib.data.AUTOTUNE)
        train_dataset = train_dataset.prefetch(1000)

        iter = train_dataset.make_one_shot_iterator()
        train_batch, label_batch = iter.get_next()


        # '''
        ####  Network  ##########

        final_layer = model_v9(train_batch, NUM_CLASS, keep_pro=0.4)
        #
        #### ↓↓↓↓↓↓  baseline   ↓↓↓↓↓↓↓↓
        # final_layer ,_ = vgg_16(train_batch, NUM_CLASS, keep_pro=0.4)
        # final_layer = V1_slim(train_batch,num_cls=NUM_CLASS,keep_prob=0.4)
        # final_layer, _ = alexnet_v2(train_batch, NUM_CLASS, keep_pro=0.4)
        #
        # learning rate
        init_lr = self.lr
        decay_step = (self.decay_every_Nepoch * self.step_per_epoch)
        lr = tf.train.exponential_decay(init_lr, global_step=global_step, decay_steps=decay_step, decay_rate=0.9)
        tf.summary.scalar('learning rate', lr)

        # Get true class from one-hot encoded format        #返回img_label每一"橫列"最大值之"索引值"，NOTE是索引而不是該數值(索引從0開始)
        image_true_class = tf.argmax(label_batch, axis=1)  # image_true_class --->> 原本的label，從onehotlabel轉回

        # '''
        def multi_category_focal_loss1(y_true, y_pred, alpha):      #gamma = 0 ,and a;pha = 1 ==> 原本的 CE
            epsilon = 1.e-7
            # gamma = 1.0
            gamma = 3.0     ## 4 5 loss會太早過低
            # gamma = 0
            # alpha = tf.constant([[2],[1],[1],[1],[1]], dtype=tf.float32)
            alpha = tf.constant(alpha, dtype=tf.float32)

            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
            y_t = tf.multiply(y_true, y_pred) + tf.multiply(1 - y_true, 1 - y_pred)
            ce = -tf.log(y_t)
            weight = tf.pow(tf.subtract(1., y_t), gamma)
            fl = tf.matmul(tf.multiply(weight, ce), alpha)
            f_loss = tf.reduce_mean(fl)
            return f_loss
        # '''
        # '''
        with tf.name_scope('focal_loss'):
            alpha = [[0.25] for _ in range(self.NUM_CLASS)]     #0.25

            focal_loss = multi_category_focal_loss1(label_batch, final_layer, alpha)
            tf.summary.scalar('focal_loss', focal_loss)
        # 
        # '''

        vars = tf.trainable_variables()
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name]) * 0.001
        # focal_loss = focal_loss + 0.01 * lossL2

        '''
        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=final_layer,
                                                                       ###Note :softmax_cross_entropy_with_logits_v2 是接沒有加softmax 的網路輸出
                                                                       labels=label_batch)
            focal_loss = tf.reduce_mean(cross_entropy)#+lossL2)
            tf.summary.scalar('entropy_loss', focal_loss)
        '''
        with tf.name_scope('adam_optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(focal_loss, global_step=global_step)
            # optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(focal_loss, global_step=global_step)


        with tf.name_scope('accuracy'):
            predict_class = tf.argmax(final_layer, axis=1)
            correction = tf.equal(predict_class, image_true_class)  # equal-->return True ,if not -->return False
            accuracy = tf.reduce_mean(tf.cast(correction, np.float32))
            tf.summary.scalar('accuracy', accuracy)
        
        
        # session
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess = sess
        sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver(max_to_keep=100)

        # keep training
        # self.load(sess, self.checkpoint_dir, step=12300)
        # self.load(sess, './All_Trained_model/ours_v5_1/ck', step=7500)

        # sess.run(vars)
        #
        # for var in vars:
        #     print(var.name)

        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(self.checkpoint_dir, sess.graph)
        config = tf.ConfigProto()

        count_param()
        
        # cc = train_batch
        # aa = sess.run(cc)
        # #
        # print(aa.shape)
        # bb = aa[0]
        # aaa = bb[:,:,1]
        # # # print(aaa.shape)
        # plt.imshow(aaa,cmap='gray')
        # plt.show()

        # bb = sess.run(label_batch)
        # print(bb.shape)

        total_start = time.time()
        n = 1

        for step in range(sess.run(global_step), self.max_batch_num + 1):

            start_time = time.time()

            _, f_loss, acc = sess.run([optimizer, focal_loss, accuracy])

            duration = time.time() - start_time
            print("Step:{}, f_Loss: {:.3f} , acc: {:.3f} , sec:{:.3f}".format(step, f_loss, acc, duration))

            if step % 20 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, global_step=step)

            #if step % 500 == 0 or step == self.max_batch_num:
            if step % (self.step_per_epoch*100) == 0 :
                total_time = time.time() - total_start
                m, s = divmod(total_time, 60)
                h, m = divmod(m, 60)
                print('\033[33m ------------------------------------------', '\033[0m')
                print('Saving %s epoch'%(n*100))
                print("It cost %02d:%02d:%02d" % (h, m, s))
                print('\033[33m ------------------------------------------', '\033[0m')
                self.save(sess, self.checkpoint_dir, step)
                n += 1

            if step == self.max_batch_num:
                ## 儲存模型
                print('\033[33m last ckpt ', '\033[0m')
                self.save(sess, self.checkpoint_dir, step)
                total_time = time.time() - total_start
                m, s = divmod(total_time, 60)
                h, m = divmod(m, 60)
                print("It cost %02d:%02d:%02d" % (h, m, s))

    def save(self, sess, checkpoint_dir, step):
        model_name = self.modelname
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(sess, os.path.join(checkpoint_dir, model_name),
                        global_step=step)  # write_meta_graph 模型結構不變，儲存一次即可

    def load(self, sess, checkpoint_dir, step=None):
        print("Reading checkpoints...")
        model_name = self.modelname
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        if step is not None:
            ckpt_name = model_name + '-' + str(step)
            self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print('\033[33m Reading intermediate checkpoints... Success', '\033[0m')
            return str(step)
        elif ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            ckpt_iter = ckpt_name.split('-')[1]
            self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print('\033[33m Reading updated checkpoints Success', '\033[0m')
            return ckpt_iter
        else:
            print('\033[33m Reading checkpoints... error', '\033[0m')
            return False


    def test3(self):

        ##  input size setting   ##
        #     H = 100
        #     W = 100
        H, W = self.input_size

        Channel = self.band_select
        # Channel = 30            ## RGB
        # Channel = self.band_select
        new_dict = {v: k for k, v in self.dict_labels.items()}  ### Get key by value

        # test_data_dir = self.test_dir
        # test_data_list = get_datalist(test_data_dir,".npy")

        test_list = self.test_list
        with open(test_list, 'r') as f:
            test_data_list = f.readlines()  # readlines --> out 已經是list
        random.shuffle(test_data_list)

        ###########################
        '''   ################# Testing data #############################  '''
        ##### 單張圖片  #####
        #test_img = self.get_one_img()
        # print(test_img.shape)

        '''######################################################################'''

        #test_img = test_img.reshape((1, H, W, Channel))

        img = tf.placeholder(tf.float32, [1, H, W, Channel])
        # test_label = tf.placeholder(tf.uint8, [1, self.NUM_CLASS])
        keep_prob = tf.placeholder(tf.float32)

        ''''######  Model 要記得改 #####'''

        logits = model_v9(img, self.NUM_CLASS, keep_pro=1.0)        ##final

        #### ↓↓↓↓↓↓  baseline   ↓↓↓↓↓↓↓↓
        # logits,_ = vgg_16(img, self.NUM_CLASS, keep_pro=1.0,global_pool=True)               ### 因為我們的Vgg只用八層 故gap一定要開
        # logits= V1_slim(img, is_train = True,num_cls=self.NUM_CLASS ,keep_prob=1.0)         # is_train 應該是要False才對，But 結果都超爛，故用True
        # logits, _ = alexnet_v2(img, self.NUM_CLASS,is_training=False,keep_pro=1.0,global_pool=True)
        #

        # session
        sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        # #self.load(sess, self.test_ckpt_dir, step=self.test_model_step)  # 載入模型
        # self.load(sess, self.test_ckpt_dir, step=9600)

        test_ckpt_dir = self.test_ckpt_dir
        self.load(sess, test_ckpt_dir, step=self.test_model_step)  # 載入模型


        count_param()
        # ##############################################################################################################################
        # total_parameters = 0
        # for variable in tf.trainable_variables():
        #     shape = variable.get_shape()
        #     variable_parameters = 1
        #     for dim in shape:
        #         variable_parameters *= dim.value
        #     print('shape' + str(shape),'variable_parameters: ' + '\033[92m' + str(variable_parameters) + '\033[0m')
        #     # print('variable_parameters: ' + '\033[92m' + str(variable_parameters) + '\033[0m')
        #     total_parameters += variable_parameters
        # print('Trainable parameters: ' + '\033[92m' + str(total_parameters) + '\033[0m')


        #################################################################################################################################
        y_predict = tf.nn.softmax(logits)
        prediction_class_num = tf.argmax(y_predict,axis = 1)

        oAA = []

        pred_class = []
        test_label_all = []
        file_name = []
        res = []

        for test_data_path in tqdm(test_data_list,ncols = 70):
            test_data_path = test_data_path.strip('\n')
            test_data = np.load(test_data_path)

            ### for PCA (PC1~6)
            # test_data = test_data[:,:,:1]


            test_label = get_label(test_data_path,self.dict_labels)
            filename = os.path.splitext(os.path.basename(test_data_path))[0]

            ### resize
            test_data = cv2.resize(test_data, self.input_size, interpolation=cv2.INTER_LINEAR)
            ### random crop
            # test_data = crop_img(test_data,self.input_size[0],self.input_size[1])


            test_data = np.expand_dims(test_data, axis=0)
            prediction = sess.run(prediction_class_num, feed_dict={img: test_data,
                                                                   keep_prob: 1.0})

            result = (test_label == prediction)

            file_name.append(filename)
            test_label_all.append(test_label)
            pred_class.append(prediction)
            res.append(result)

        conMatrix = confusion_matrix(test_label_all, pred_class)  ###test_label --> without one-hot

        kappa = cohen_kappa_score(test_label_all, pred_class)
        oAA.append(accuracy_score(test_label_all, pred_class, normalize=True))
        print('---------------------------------')

        print('confusion Matrix:')
        # print(conMatrix)
        print("overall acc:", oAA)

        print('Macro precision: %.4f '%precision_score(test_label_all, pred_class, average='macro'))
        # print(precision_score(test_label_all, pred_class, average='micro'))

        print('Macro recall: %.4f '%recall_score(test_label_all, pred_class, average='macro'))
        # print(recall_score(test_label_all, pred_class, average='micro'))

        ####  sklearn 的 Macro F1 公式為各類別的F1 score總和再平均 , 不是直接拿上面的precision 和 recall 去計算
        print('Macro F1-score: %.4f '%f1_score(test_label_all, pred_class, average='macro'))
        # print(f1_score(test_label_all, pred_class, average='micro'))

        print("kappa : ", kappa)



        oAA_save = int(round(oAA[0], 3) * 1000)

        ###  save to excel file
        aa = pd.DataFrame(file_name)
        bb = pd.DataFrame(test_label_all)
        cc = pd.DataFrame(pred_class)
        dd = pd.DataFrame(res)

        ppp = pd.concat([aa, bb, cc, dd], axis=1)
        ppp.columns=['file','True_label','Pred_label','result']

        writer = pd.ExcelWriter(test_ckpt_dir+'/result_%s_%s.xlsx'%(self.test_model_step,oAA_save), engine='xlsxwriter')
        ppp.to_excel(writer,sheet_name='result',index=True)
        writer.save()

        ###  save confusion Matrix
        ID = [i for i in range(self.NUM_CLASS)]
        plt.figure(figsize=(16, 12))            ### 單一張的(16,12) 兩張的figsize要再另外找
        # plt.subplot(1, 2, 1)
        plot_confusion_matrix(conMatrix,classes=ID,normalize=True)
        # plt.subplot(1, 2, 2)
        # plot_confusion_matrix(conMatrix, classes=ID, normalize=False)
        plt.savefig(test_ckpt_dir + '/res_coMatrix_%s_%s.png' %(self.test_model_step,oAA_save))
        plt.show()


        """
        ## 單張
        print(' prediction:', prediction)
        max_idx = np.argmax(prediction)  ## prediction 最大值的idx
        predict_label = new_dict[max_idx]
        prediction = np.max(prediction)  ## 預測的機率

        print("This is \033[33m %s \033[0m" % predict_label,"with possibility \033[33m %.2f \033[0m" % (prediction * 100), '%')
        """



