import os
import numpy as np
import tensorflow as tf
import cv2
import glob
import random
from sklearn.decomposition import PCA
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import itertools
from scipy import misc
import numexpr as ne
from tqdm import tqdm

#################################################################################################
dict_labels = {"Fittonia": 0,                "Carnosa": 1,                "Hoya_kerrii": 2,
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
##################################################################################################
#####################↓↓↓↓以下為主程式有用到的function ########################################


def get_datalist(dir: object, filetype: object) -> object:             ##filetype: 副檔名
    img_list = []
    for folders in glob.glob(dir):  ## os.listdir沒有包含整個路徑不能用，因為下面那個for 會找不到資料夾
        for filename in os.listdir(folders):
            if filename.endswith(filetype):       #".hdr"
                # print(filename)
                img_path = os.path.join(folders, filename)
                img_list.append(img_path)
    """
    ## for 測試
    print('---------------------------------------------------------')
    print('Summary')
    #print(*img_list, sep='\n')
    print('all list:',np.array(img_list).shape)
    print('---------------------------------------------------------')
    """
    return img_list


def doPCA(input_cub,n_components):          ### 同樣的cub每次結果都一樣
    #print("\n Do PCA......")

    H,W,C = input_cub.shape
    input_cub = input_cub.reshape((H*W,C))

    pca = PCA(n_components = n_components)
    new = pca.fit_transform(input_cub)

    out_cub = new.reshape((H,W,n_components))

    np.set_printoptions(precision=4,suppress=True)
    var = pca.explained_variance_ratio_                     #獨立的
    var_sum = pca.explained_variance_ratio_.cumsum()        #累加的

    #print('out_cub shape:',out_cub.shape)
    #print('Finish PCA')
    #print('--------------------------------------------')
    return out_cub,var,var_sum


def get_label(file_path,dict):
    if "Fittonia" in file_path:
        return dict["Fittonia"]
    elif "Carnosa" in file_path:
        return dict["Carnosa"]
    elif "Hoya_kerrii" in file_path:
        return dict["Hoya_kerrii"]
    elif "Peace_Lilies" in file_path:
        return dict["Peace_Lilies"]
    elif "Zamioculcas" in file_path:
        return dict["Zamioculcas"]
    ###############################
    elif "Anyamanee" in file_path:
        return dict["Anyamanee"]
    elif "Aglaonema" in file_path:
        return dict["Aglaonema"]
    elif "Epipremnum" in file_path:
        return dict["Epipremnum"]
    elif "Alocasia" in file_path:
        return dict["Alocasia"]
    elif "Hydrocotyle" in file_path:
        return dict["Hydrocotyle"]
    ###############################
    elif "Guilfoyle" in file_path:
        return dict["Guilfoyle"]
    elif "Scandent" in file_path:
        return dict["Scandent"]
    elif "Sansevieria" in file_path:
        return dict["Sansevieria"]
    elif "Dracaena" in file_path:
        return dict["Dracaena"]
    elif "Chlorophytum" in file_path:
        return dict["Chlorophytum"]
    ###############################
    elif "Begonia" in file_path:
        return dict["Begonia"]
    elif "Bromeliads" in file_path:
        return dict["Bromeliads"]
    elif "Clusia_rosea" in file_path:
        return dict["Clusia_rosea"]
    elif "Davallia" in file_path:
        return dict["Davallia"]
    elif "Poinsettia" in file_path:
        return dict["Poinsettia"]
    ###############################
    elif "Glechoma" in file_path:
        return dict["Glechoma"]
    elif "Plectranthus" in file_path:
        return dict["Plectranthus"]
    elif "Variegata" in file_path:
        return dict["Variegata"]
    elif "Pachira_aquatica" in file_path:
        return dict["Pachira_aquatica"]
    elif "Calathea_lancifolia" in file_path:
        return dict["Calathea_lancifolia"]
    ###############################
    elif "Boston_Swordfern" in file_path:
        return dict["Boston_Swordfern"]
    elif "Spathoglottis" in file_path:
        return dict["Spathoglottis"]
    elif "Peperomia" in file_path:
        return dict["Peperomia"]
    elif "Podocarpaceae" in file_path:
        return dict["Podocarpaceae"]
    elif "Aluminium_plant" in file_path:
        return dict["Aluminium_plant"]
    ###############################
    elif "Ammocallis_rosea" in file_path:
        return dict["Ammocallis_rosea"]
    elif "Myriophyllum" in file_path:
        return dict["Myriophyllum"]

    else:
        print('\033[33m key is not in dictionary', '\033[0m')


def label_abbreviation(file_path):                  ## for 期刊畫圖
    if "Fittonia" in file_path:
        return str('F.a.')
    elif "Carnosa" in file_path:
        return str('H.c.')
    elif "Hoya_kerrii" in file_path:
        return str('H.k.')
    elif "Peace_Lilies" in file_path:
        return str('S.k.')
    elif "Zamioculcas" in file_path:
        return str('Z.z.')
        ###############################
    elif "Anyamanee" in file_path:
        return str('A.an.')
    elif "Aglaonema" in file_path:
        return str('A.c.')
    elif "Alocasia" in file_path:
        return str('A.am.')
    elif "Hydrocotyle" in file_path:
        return str('H.v.')
        ###############################
    elif "Guilfoyle" in file_path:
        return str('P.g.')
    elif "Scandent" in file_path:
        return str('S.a.')
    elif "Sansevieria" in file_path:
        return str('S.t.')
    elif "Dracaena" in file_path:
        return str('D.m.')
    elif "Chlorophytum" in file_path:
        return str('C.c.')
        ###############################
    elif "Begonia" in file_path:
        return str('B.c.')
    elif "Bromeliads" in file_path:
        return str('C.b.')
    elif "Clusia_rosea" in file_path:
        return str('C.r.')
    elif "Davallia" in file_path:
        return str('D.g')
        ###############################
    elif "Glechoma" in file_path:
        return str('G.h')
    elif "Plectranthus" in file_path:
        return str('P.am.')
    elif "Variegata" in file_path:
        return str('P.a.cv.')
    elif "Pachira_aquatica" in file_path:
        return str('P.aq.')
    elif "Calathea_lancifolia" in file_path:
        return str('C.l.')
        ###############################
    elif "Boston_Swordfern" in file_path:
        return str('N.e.')
    elif "Spathoglottis" in file_path:
        return str('S.p.')
    elif "Peperomia" in file_path:
        return str('P.p.')
    elif "Podocarpaceae" in file_path:
        return str('P.m.')
    elif "Aluminium_plant" in file_path:
        return str('P.c.')
        ###############################
    elif "Ammocallis_rosea" in file_path:
        return str('A.r.')
    elif "Myriophyllum" in file_path:
        return str('M.a.')

    else:
        print('\033[33m key is not in dictionary', '\033[0m')

# def Plot_avgSpectrum(dir,save_path,save_name):
def Plot_avgSpectrum(dir):                  ### Only can plot band 147 (raw,snv,msc..) ; PCA..要另外畫
    def avgSpectrum(spec):
        """
        param spec: shape (H,W,C)  ###就是cub
        return: plt
        """

        H, W, C = spec.shape
        temp = []

        for i in range(C):
            aa = np.mean(spec[:, :, i])
            temp.append(aa)

        temp = np.asarray(temp)
        print(temp.shape)
        return temp

    datalist = get_datalist(dir,'.npy')
    x = np.linspace(468, 898, 147)
    plt.figure(figsize=(6, 3.1), dpi=300)

    # datalist = random.sample(datalist,15)

    ###########################
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.Paired(np.linspace(0, 1, 40))))

    max_all_data = 0                ## initial value
    min_all_data = 1

    for i in tqdm(datalist,ncols = 70):
        cub = np.load(i)

        for j in range(cub.shape[2]):
            if np.max(cub[:, :, j]) > max_all_data:
                max_all_data = np.max(cub[:, :, j])
            if np.min(cub[:, :, j]) < min_all_data:
                min_all_data = np.min(cub[:, :, j])

        cub = cub - min_all_data
        cub = cub / (max_all_data - min_all_data)

        temp = avgSpectrum(cub)

        # label = get_label(i,dict_labels)
        # print(np.max(temp))
        label = label_abbreviation(i)

        plt.plot(x, temp, linewidth=0.6,label = '%s'%label)         ### 畫多條用0.6  單條用1.0


        # plt.text(x[-30], temp[-30], 'ID:%s'%label , fontsize=3)                 ### [] 內的值要一樣才會抓到正確的點
        # plt.text(x[-1], temp[-1], 'ID:%s' % label, fontsize=3)

        plt.legend(fontsize=8,bbox_to_anchor=(1, 1),ncol=2)
        # plt.legend(fontsize=8, loc = 4)

    fonts = 8
    plt.xlim(465, 910)                  ### 910
    # plt.ylim(0, 1)          ###
    plt.xlabel('Wavelength (nm)', fontsize=fonts)
    plt.ylabel('Reflectance', fontsize=fonts)
    plt.yticks(fontsize=fonts)
    plt.xticks(fontsize=fonts)
    plt.tight_layout(pad=0.3)
    # plt.grid(True)
    plt.grid(axis='both',linestyle= 'dotted' ,linewidth = 0.3)

    # plt.savefig(save_path+save_name)
    plt.show()
    # plt.savefig(dir+'/wavelength.png')

# dir = './data_new/new0702/train/uni_crop/forplotcurve_forIEEE(test)'
# savepath = './data_new/'
# savename = 'plot_singel'
# Plot_avgSpectrum(dir,savepath,savename)



def crop_img(img,new_height,new_width):
# def crop_img(img, rgb_img, PCA_cub, AA, BB, CC, DD, new_height, new_width):
    x0 = np.random.randint(0, img.shape[1] - new_height + 1)
    y0 = np.random.randint(0, img.shape[0] - new_width + 1)
    #z0 = np.random.randint(0, img.shape[2] - band + 1)
    img = img[ x0:x0 + new_height,y0:y0 + new_width]     #, z0:z0 + band]
    '''
    if rgb_img is not None:
        rgb_img = rgb_img[x0:x0 + new_height, y0:y0 + new_width]
    if PCA_cub is not None:
        PCA_cub = PCA_cub[x0:x0 + new_height, y0:y0 + new_width]
    if AA is not None:
        AA = AA[x0:x0 + new_height, y0:y0 + new_width]

    if BB is not None:
        BB = BB[x0:x0 + new_height, y0:y0 + new_width]

    if CC is not None:
        CC = CC[x0:x0 + new_height, y0:y0 + new_width]

    if DD is not None:
        DD = DD[x0:x0 + new_height, y0:y0 + new_width]
    '''
    return img  #,rgb_img,PCA_cub,AA,BB,CC,DD


def uni_crop(file_path,crop_h,crop_w,save_path):
    """
    :param file_path:
    :param crop_h: 挑可以整除的
    :param crop_w:
    :return:
    """
    def get_rgb(raw_data):
        rgb = [38,18,1]
        rgb_arr = np.zeros((1200, 1200, 3))
        for i, j in zip(range(3), rgb):
            rgb_arr[:,:,i] =raw_data[:,:,j]

        return rgb_arr.astype(np.float32)


    file = np.load(file_path)
    ori_H = file.shape[0]
    ori_W = file.shape[1]

    if (ori_H % crop_h) or (ori_W % crop_w) != 0:
        print('\033[33m Choose a number which is divisible  ...', '\033[0m')
        exit()

    label = os.path.splitext(os.path.basename(file_path))[0]

    index = 0
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    rgb_arr = get_rgb(file)

    # for i in tqdm(range(int(ori_H / crop_h)),ncols = 70):
    for i in range(int(ori_H / crop_h)):
        for j in range(int(ori_W / crop_w)):
            new_img = file[i*crop_h:(i+1)*crop_h,j*crop_w:(j+1)*crop_w]
            new_rgb = rgb_arr[i*crop_h:(i+1)*crop_h,j*crop_w:(j+1)*crop_w]

            misc.imsave(save_path +  '%s_rgb_%s.png'%(label,index) , new_rgb)
            #
            np.save(save_path + '%s_rgb_%s' % (label, index), new_rgb)
            np.save(save_path + '%s_%s'%(label,index) , new_img)
            index += 1

    # print('\033[33m Finish uniform crop...', '\033[0m')


def OnehotTransform(labels):
    from sklearn.preprocessing import OneHotEncoder
    onehot_encoder = OneHotEncoder(sparse=False) #sparse=True 表示編碼的格式，默認為True，即為稀疏的格式，指定False 則就不用toarray() 了
    labels = np.reshape(labels, (len(labels), 1))
    labels = onehot_encoder.fit_transform(labels).astype(np.uint8)
    return labels

def count_param():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print('Trainable parameters: ' + '\033[92m' + str(total_parameters) + '\033[0m')

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    # fmt = '.2f' if  normalize and cm.all() != 0 else 'd'
    # thresh = cm.max() / 2.
    thresh = 0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
       if cm[i,j]==0:
           plt.text(j, i, format(cm[i, j], '.0f'),
                    horizontalalignment="center")
       else:
           plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center")
                 # ,color="white" if cm[i, j] > thresh else "black")
                 # ,color="black" if cm[i, j] > thresh else "red")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def excel2coMatrix(excelfile,num_class,savepath,savename):              ### 從excel畫confusion matrix
    df = pd.read_excel(excelfile, sheetname=0)

    True_label = df['True_label'].tolist()                  ### '%^&' -> 看excel裡面要找的數列的標題
    Pred_label = df['Pred_label'].tolist()

    conMatrix = confusion_matrix(True_label, Pred_label)
    ID = [i for i in range(num_class)]             ### number of classes

    def plot_confusion_matrix2(cm, classes,                    ## ###  改一些圖的label blabla...  for 期刊
                               normalize=False,
                               title='Confusion matrix',
                               cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        prop = {'family': 'Times New Roman', 'size': 14}

        plt.figure(figsize=(12, 10))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title,prop)

        plt.colorbar(pad=0.01)                  ### pad -> colorbar 與fig 的間距
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        # fmt = '.2f' if  normalize and cm.all() != 0 else 'd'
        thresh = cm.max() / 2.
        # thresh = 0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if cm[i, j] == 0:
                plt.text(j, i, format(cm[i, j], '.0f'),
                         horizontalalignment="center")
            elif i == j:          ###對角線
                plt.text(j, i, format(cm[i, j], fmt),
                         horizontalalignment="center",
                         weight="bold",
                         size=8,
                         color="white")  # if cm[i, j] > thresh else "black"):
            else:
                plt.text(j, i, format(cm[i, j], fmt),
                         horizontalalignment="center")

        plt.ylabel('True label', prop)
        plt.xlabel('Predicted label', prop)
        plt.tight_layout()

    plot_confusion_matrix2(conMatrix, classes=ID, normalize=True)
    plt.savefig(savepath + '%s.png' % savename)
    # plt.show()


# file = './data_new/forIEEE/coMatrixforIEEE/ours_RGB.xlsx'
# file = './data_new/forIEEE/coMatrixforIEEE/ours_PCA.xlsx'
# file = './data_new/forIEEE/coMatrixforIEEE/ours_NIR.xlsx'
# file = './data_new/forIEEE/coMatrixforIEEE/ours_RGBNIR.xlsx'
# file = './data_new/forIEEE/coMatrixforIEEE/ours_UBS.xlsx'
# # #
# savepath = './data_new/forIEEE/coMatrixforIEEE/'
# savename = 'ours_UBS'
# excel2coMatrix(file,30,savepath,savename)


def trans2metrix (filepath):            ###2021/06/01 將30種植物各1張 ，組成一個大矩陣  for OMP
    data_list = get_datalist(filepath, ".npy")
    # data_list = random.sample(data_list, 3)

    firstfile = np.load(data_list[0])             ###為了建立第一個開頭
    H, W, C = firstfile.shape
    all = firstfile.reshape((H * W, C))

    for file in tqdm(data_list[1:],ncols = 70):                     ###從第二個開始append

        cub = np.load(file)
        H, W, C = cub.shape
        cub = cub.reshape((H * W, C))
        all = np.append(all,cub,axis=0)

    all = np.transpose(all)

    sio.savemat(filepath + 'all.mat',{'all':all})
    print("Finish~~~")
    # print(type(all))
    # print(all.dtype)
    # print("cub shape : ",all.shape)

# filepath = './data_new/ori1200data/'
# trans2metrix(filepath)