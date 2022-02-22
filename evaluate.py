from ModelSpecPatConv3D import net
from argparse import ArgumentParser
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import tensorflow as tf
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'          #我設的

parser = ArgumentParser()
parser.add_argument('--patch_size', type=int, default=5)
#parser.add_argument('--data', type=str, default='Indian_pines', help='Indian_pines or Salinas or KSC')
#number_of_band = {'Indian_pines': 2, 'Salinas': 2, 'KSC': 2, 'Botswana': 1}
number_of_band = 1
def GroundTruthVisualise(data, original=True):
    from matplotlib.pyplot import imshow, show, colorbar, set_cmap, clim,savefig

    labels = []

    if original:
        #labels = ['Poinsettia', 'Calathea_lancifolia', 'Bromeliads','background' ]
        labels = ['background', 'Bromeliads','Calathea_lancifolia','Poinsettia' ]
        #labels = ['Poinsettia', background', 'Bromeliads',' 'Calathea_lancifolia']
        #labels = ['Bromeliads', 'background']
    else:
        labels = []

    def discrete_matshow(data):
        # get discrete colormap
        cmap = plt.get_cmap('jet', np.max(data) - np.min(data) + 1)
        # set limits .5 outside true range
        mat = plt.matshow(data, cmap=cmap, vmin=np.min(data) - 0.5, vmax=np.max(data) + 0.5)
        # tell the colorbar to tick at integers
        cax = plt.colorbar(mat, ticks=np.arange(np.min(data), np.max(data)+1))

        cax.ax.set_yticklabels(labels)
        savefig('./output/Poinsettia05.png')
    plt.imshow(data)
    discrete_matshow(data)
    plt.show()
    #

def evaluate(opt):

    data = sio.loadmat('./data/Processed_testBro_55.mat')
    #_, _, TEST = maybeExtract(opt.data, opt.patch_size)            #切好的patch檔
    TEST = (data['test_patch'], data['test_labels'])

    test_data, test_label = TEST[0], TEST[1]
    HEIGHT = test_data.shape[1]
    WIDTH = test_data.shape[2]
    CHANNELS = test_data.shape[3]
    N_PARALLEL_BAND = number_of_band
    #NUM_CLASS = test_label.shape[1]
    NUM_CLASS = 4

    graph = tf.Graph()
    with graph.as_default():

        # Define Model entry placeholder
        img_entry = tf.placeholder(tf.float32, shape=[None, WIDTH, HEIGHT, CHANNELS])
        img_label = tf.placeholder(tf.uint8, shape=[None, NUM_CLASS])

        # Get true class from one-hot encoded format
        image_true_class = tf.argmax(img_label, axis=1)

        # Dropout probability for the model
        prob = tf.placeholder(tf.float32)

        # Network model definition
        model = net(img_entry, prob, HEIGHT, WIDTH, CHANNELS, N_PARALLEL_BAND, NUM_CLASS)

        # Cost Function
        final_layer = model['dense3']

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=final_layer,
                                                                       labels=img_label)
            cost = tf.reduce_mean(cross_entropy)

        # Optimisation function
        with tf.name_scope('adam_optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(cost)

        # Model Performance Measure
        with tf.name_scope('accuracy'):
            predict_class = model['predict_class_number']
            correction = tf.equal(predict_class, image_true_class)

        accuracy = tf.reduce_mean(tf.cast(correction, tf.float32))
        saver = tf.train.Saver()
        #saver = tf.train.import_meta_graph('Trained_model3/model250.meta')
        with tf.Session(graph=graph) as session:
            saver.restore(session, tf.train.latest_checkpoint('./Trained_model3_55_dp0.5_new/'))
            def test(t_data, t_label, test_iterations=1, evalate=False):
                assert test_data.shape[0] == test_label.shape[0]
                y_predict_class = model['predict_class_number']
                # OverallAccuracy, averageAccuracy and accuracyPerClass
                overAllAcc, avgAcc, averageAccClass = [], [], []
                for _ in range(test_iterations):

                    pred_class = []
                    #for t in tqdm(t_data):
                    for t in t_data:
                        t = np.expand_dims(t, axis=0)
                        feed_dict_test = {img_entry: t, prob: 1.0}
                        prediction = session.run(y_predict_class, feed_dict=feed_dict_test)
                        pred_class.append(prediction)

                    true_class = np.argmax(t_label, axis=1)
                    conMatrix = confusion_matrix(true_class, pred_class)

                    # Calculate recall score across each class
                    classArray = []
                    for c in range(len(conMatrix)):
                        recallScore = conMatrix[c][c] / sum(conMatrix[c])
                        classArray += [recallScore]
                    averageAccClass.append(classArray)
                    avgAcc.append(sum(classArray) / len(classArray))
                    overAllAcc.append(accuracy_score(true_class, pred_class))

                averageAccClass = np.transpose(averageAccClass)
                meanPerClass = np.mean(averageAccClass, axis=1)

                #showClassTable(meanPerClass, title='Class accuracy')
                print('Average Accuracy: ' + str(np.mean(avgAcc)))
                print('Overall Accuracy: ' + str(np.mean(overAllAcc)))

            # -- Pixel-wise classification --#

            def pixelClassification():
                input_mat = sio.loadmat("./data/Poinsettia05.mat")['Poinsettia05']

                input_height, input_width = input_mat.shape[0], input_mat.shape[1]
                BAND = input_mat.shape[2]
                PATCH_SIZE = opt.patch_size
                #PATCH_SIZE = 5

                MEAN_ARRAY = np.ndarray(shape=(BAND, 1))
                new_input_mat = []
                calib_val_pad = int((PATCH_SIZE-1)/2)
                for i in range(BAND):
                    MEAN_ARRAY[i] = np.mean(input_mat[:,:, i])      #mean[i]:第i個波段的均值 (為一個值)
                    new_input_mat.append(np.pad(input_mat[:,:, i], calib_val_pad, 'constant', constant_values=0))

                new_input_mat = np.transpose(new_input_mat, (1, 2, 0))
                input_mat = new_input_mat

                def Patch(height_index, width_index):

                    transpose_array = input_mat
                    height_slice = slice(height_index, height_index+PATCH_SIZE)
                    width_slice = slice(width_index, width_index+PATCH_SIZE)

                    patch = transpose_array[height_slice, width_slice, :]
                    mean_normalized_patch = []
                    for i in range(BAND):
                        mean_normalized_patch.append(patch[:, :, i] - MEAN_ARRAY[i])

                    mean_normalized_patch = np.array(mean_normalized_patch).astype(np.float16)
                    mean_normalized_patch = np.transpose(mean_normalized_patch, (1, 2, 0))
                    return mean_normalized_patch

                labelled_img = np.ndarray(shape=(input_height, input_width))

                for i in tqdm(range(input_height - 1),ncols=70):
                #for i in range(input_height - 1):
                    for j in range(input_width - 1):
                        current_input = Patch(i, j)
                        current_input = np.expand_dims(current_input, axis=0)
                        feed_dict_test = {img_entry: current_input, prob: 1.0}
                        prediction = session.run(model['predict_class_number'], feed_dict=feed_dict_test)
                        labelled_img[i,j] = prediction[0]

                #labelled_img += 1      #好像是畫圖分類的類別??
                labelled_img = np.pad(labelled_img, [(0,1),(0,0)], 'constant', constant_values=(0, 0))

                print('labelled_img_min',np.min(labelled_img) , 'labelled_img_,max',np.max(labelled_img), 'labelled_img.shape:',labelled_img.shape)
                #GroundTruthVisualise(labelled_img, opt.data, False)
                GroundTruthVisualise(labelled_img, True)

            # Test and plot
            ##test(test_data, test_label, test_iterations=1)
            pixelClassification()

if __name__ == '__main__':
    option = parser.parse_args()
    evaluate(option)
