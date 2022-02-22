from func_for_v5 import *
from spectral import *
from tqdm import tqdm
from scipy import misc
import cv2
import concurrent.futures
#################################################
#   從原始檔選RGB:[42,22,5] ;;
#   從147選的話 RGB:[38,18,1]   紅外光(3-bands): 750.7nm[88], 800.0nm[108], 851.0nm[125]
#                               紅外光(6-bands): 725.0nm[80], 750.7nm[88], 774.0nm[95], 800.0[108], 824.7nm[116], 851.0[125]
#                               紅外光(9-bands): 700.3nm[73], 725.0nm[80], 750.7nm[88], 774.0nm[95], 800.0[108], 824.7nm[116], 851.0nm[125], 873.9nm[137], 898.7nm[146]
#
#################################################
def creat_data(Mode=True):  # ## crop後儲存"每一張"，為了手動整理成dataset(替除掉背景)

    dict_labels = {"Fittonia":0,            "Carnosa":1,            "Hoya_kerrii":2,
                   "Peace_Lilies":3,        "Zamioculcas":4,        "Anyamanee":5,
                   "Aglaonema":6,           "Epipremnum":7,         "Alocasia":8,
                   "Hydrocotyle":9,         "Guilfoyle":10,         "Scandent":11,
                   "Sansevieria":12,        "Dracaena":13,          "Chlorophytum":14,
                   "Begonia":15,            "Bromeliads":16,        "Clusia_rosea":17,
                   "Davallia":18,           "Poinsettia":19,        "Glechoma":20,
                   "Plectranthus":21,       "Variegata":22,         "Pachira_aquatica":23,
                   "Calathea_lancifolia":24,"Boston_Swordfern":25,  "Spathoglottis":26,
                   "Peperomia":27,          "Podocarpaceae":28,     "Aluminium_plant":29
                   }

    ori_band_list = [i for i in range(161)]  ## 原始資料161個band
    deleted_band = [0, 1, 2, 3, 46, 47, 48, 49, 50, 156, 157, 158, 159, 160]  ##沒有用的band
    [ori_band_list.remove(cElement) for cElement in [ori_band_list[i] for i in deleted_band]]

    ### All Bands
    precessed_band = ori_band_list  ### 原始161-無用的(14) = 147個band
    band = 147

    ### Uniform select bands
    # band = 10
    # itval = int(len(ori_band_list) // 9)                        ### 10band --> set 9,
    # precessed_band = ori_band_list[0:147:itval]                 ### 10band 可用(0,1,2)

    if Mode is True:
        train_dir = './data_new/train/*'
        ori_data_list = get_datalist(train_dir,".hdr")
        nbofcub = 12
        save_path = './data_new/croped_data/train_fullclass/'  ### 每次新切資料要記得改
    else:
        test_dir = './data_new/test/*'
        ori_data_list = get_datalist(test_dir,".hdr")
        nbofcub = 4
        save_path = './data_new/croped_data/test_fullclass/'  ### 每次新切資料要記得改
    ### Setting ########################################################################################
    rgb = [42, 22, 5]

    crop_size = 350
    input_size = (100, 100)
    n_component = 6

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    ####################################################################################################

    temp = []
    total_nbofdata = 0
    # ori_data_list = random.sample(ori_data_list,4)             ## 測試code時用，不用load全部資料

    print('Total ori list data:',len(ori_data_list))
    print('\n\033[33m Creating dataset...', '\033[0m')

    for i in tqdm(ori_data_list, ncols=70):
    #for i in (ori_data_list):
        cub_raw = open_image(i)
        cub = cub_raw.read_bands(precessed_band)
        rgb_img = cub_raw.read_bands(rgb)

        # cub = cub.astype(np.float32)      ### 好像是錯的 不要用
        # cub = cub - np.min(cub)
        # cub = cub / (np.max(cub) - np.min(cub))

        label_name = os.path.basename(os.path.dirname(i))
        ###  為了存 rgb_img 所用的 ###
        label_2 = os.path.basename(i).rsplit('_', 1)[0]

        # Do PCA
        pca_cub, _, pca_varsum = doPCA(cub, n_components=n_component)
        # temp.append(pca_varsum)

        for num in range(nbofcub):
            data, rgb_patch ,pca_data = crop_img(cub, rgb_img,pca_cub, crop_size, crop_size, band=band)
            #data, rgb_patch = crop_img(pca_cub, rgb_img, crop_size, crop_size, band=n_component)

            misc.imsave(save_path + "%s_%s.png" % (label_2, num), rgb_patch)
            data = cv2.resize(data, input_size, interpolation=cv2.INTER_LINEAR)
            pca_data = cv2.resize(pca_data, input_size, interpolation=cv2.INTER_LINEAR)
            label = dict_labels[label_name]

            if Mode is True:
                np.savez(save_path + "%s_%s" % (label_2, num), train_data=data, train_label=label)
                np.savez(save_path + "%s_%s_PCA" % (label_2, num), train_data=pca_data, train_label=label)
            else:
                np.savez(save_path + "%s_%s" % (label_2, num), test_data=data, test_label=label)
                np.savez(save_path + "%s_%s_PCA" % (label_2, num), test_data=pca_data, test_label=label)

            total_nbofdata += 1

    #print('pca_varsum mean:', np.mean(temp,axis=0))
    print('\ntotal data:' + str(total_nbofdata))
    print('\033[33m Finish creating dataset...', '\033[0m')
    # return train_dataset,train_label


# def data2np(Mode=True):  # ## crop後儲存"每一張"，為了手動整理成dataset(替除掉背景)
#
#     ori_band_list = [i for i in range(161)]  ## 原始資料161個band
#     deleted_band = [0, 1, 2, 3, 46, 47, 48, 49, 50, 156, 157, 158, 159, 160]  ##沒有用的band
#     [ori_band_list.remove(cElement) for cElement in [ori_band_list[i] for i in deleted_band]]
#
#     ### All Bands
#     precessed_band = ori_band_list  ### 原始161-無用的(14) = 147個band
#     band = 147
#
#     if Mode is True:
#         train_dir = './data_new/train/*'
#         ori_data_list = get_datalist(train_dir,".hdr")
#         # nbofcub = 8
#         # '''
#         path_raw = './data_new/new0702/train/ori/raw/'  ### 每次新切資料要記得改
#         path_rgb = './data_new/new0702/train/ori/rgb/'
#         path_pca = './data_new/new0702/train/ori/pca/'
#         path_snv = './data_new/new0702/train/ori/snv/'
#         path_snvpca = './data_new/new0702/train/ori/snv_pca/'
#         save_path = [path_raw,path_rgb,path_pca,path_snv,path_snvpca]
#         '''
#         path_msc = './data_new/new0702/train/ori/msc/'
#         path_mscpca = './data_new/new0702/train/ori/mscpca/'
#         save_path = [path_msc,path_mscpca]
#         '''
#     else:
#         test_dir = './data_new/test3/*'
#         ori_data_list = get_datalist(test_dir,".hdr")
#         # nbofcub = 4
#         '''
#         path_raw = './data_new/new0702/test/ori/raw/'  ### 每次新切資料要記得改
#         path_rgb = './data_new/new0702/test/ori/rgb/'
#         path_pca = './data_new/new0702/test/ori/pca/'
#         path_snv = './data_new/new0702/test/ori/snv/'
#         path_snvpca = './data_new/new0702/test/ori/snv_pca/'
#         '''
#         path_msc = './data_new/new0702/test/ori/msc/'
#         path_mscpca = './data_new/new0702/test/ori/mscpca/'
#         # save_path = [path_raw,path_rgb,path_pca,path_snv,path_snvpca] #,path_msc,path_mscpca]
#         save_path = [path_msc,path_mscpca]
#         # '''
#     ### Setting ########################################################################################
#     rgb = [42, 22, 5]
#
#     n_component = 6
#
#     for path in save_path:
#         if not os.path.exists(path):
#             os.makedirs(path)
#
#     ####################################################################################################
#
#     # ori_data_list = random.sample(ori_data_list,4)             ## 測試code時用，不用load全部資料
#
#     print('Total ori list data:',len(ori_data_list))
#     print('\n\033[33m Converting ori data to np format...', '\033[0m')
#
#     for i in tqdm(ori_data_list, ncols=70):
#     #for i in (ori_data_list):
#         cub_raw = open_image(i)
#         cub = cub_raw.read_bands(precessed_band)
#         rgb_img = cub_raw.read_bands(rgb)
#
#         ###  為了存 rgb_img 所用的 ###
#         label_2 = os.path.basename(i).rsplit('_', 1)[0]
#
#         # Do PCA
#         # pca_cub, _, pca_varsum = doPCA(cub, n_components=n_component)
#         # temp.append(pca_varsum)
#
#
#         if Mode is True:
#
#             np.save(path_raw + "%s" % (label_2), cub)
#             np.save(path_rgb + "%s_rgb" % (label_2), rgb_img)
#             np.save(path_pca + "%s_pca" % (label_2), pca_cub)
#
#         else:
#             np.save(path_raw + "%s" % (label_2), cub)
#             np.save(path_rgb + "%s_rgb" % (label_2), rgb_img)
#             np.save(path_pca + "%s_pca" % (label_2), pca_cub)
#
#
#     # print('pca_varsum mean:', np.mean(temp,axis=0))
#     # print('\ntotal data:' + str(total_nbofdata))
#     print('\033[33m Finish converting data...', '\033[0m')
#     # return train_dataset,train_label

def crop_data(Mode=True):

    if Mode:
        # path_raw = './data_new/new0702/train/ori/raw/'          #[0]
        # path_rgb = './data_new/new0702/train/ori/rgb/'          #[1]
        path_pca = './data_new/new0702/train/ori/pca/'          #[2]

        save_path =  './data_new/new0702/train/cropped/'

        nbofcub = 16
    else:
        path_raw = './data_new/new0702/test/ori/raw/'  # [0]
        path_rgb = './data_new/new0702/test/ori/rgb/'  # [1]
        path_pca = './data_new/new0702/test/ori/pca/'  # [2]

        save_path = './data_new/new0702/test/cropped/'

        nbofcub = 4


    if not os.path.exists(save_path):
        os.makedirs(save_path)
        # all_path = [path_raw,path_rgb,path_pca,path_snv,path_snvpca,path_msc,path_mscpca]

    list_raw = get_datalist(path_raw,'.npy')
    list_rgb = get_datalist(path_rgb, '.npy')
    list_pca = get_datalist(path_pca, '.npy')

    crop_size = 350
    total_nbofdata = 0

    all_list = list(zip(list_raw,list_rgb,list_pca))

    # aa = all_list[:2]
    # print(np.asarray(all_list).shape)
    for a,b,c,d,e,f,g in tqdm(all_list, ncols = 70):
        print('---------------')
        data_raw = np.load(a)
        data_rgb = np.load(b)
        data_pca = np.load(c)

        ########################
        label_raw = os.path.splitext(os.path.basename(a))[0]  ##.rsplit('_', 1)[0]

        for num in range(nbofcub):
            raw,rgb,pca = crop_img(data_raw,data_rgb,data_pca,crop_size,crop_size)

            misc.imsave(save_path + "%s_%s.png" % (label_raw, num), rgb)
            if Mode :
                np.save(save_path + "%s_%s" % (label_raw, num), raw)
                np.save(save_path + "%s_%s_rgb" % (label_raw, num), rgb)
                np.save(save_path + "%s_%s_pca" % (label_raw, num), pca)

            else:
                np.save(save_path + "%s_%s" % (label_raw, num), raw)
                np.save(save_path + "%s_%s_rgb" % (label_raw, num), rgb)
                np.save(save_path + "%s_%s_pca" % (label_raw, num), pca)

            total_nbofdata += 1
            # print(label_name+'_%s'%num)
    print('\n Total num of data of each precess:' + str(total_nbofdata))        ## 各種前處理各切多少張
    print('\033[33m Finish creating dataset...', '\033[0m')

def uni_band_selection(numofband):
    dir = './data_new/new0702/train/uni_crop/raw_ROI/'                ### 換資料要記得改
    if numofband == 51:
        save_dir = './data_new/new0702/train/uni_crop/band_selection/raw_band51/'         ### 要記得改
    elif numofband == 54:
        save_dir = './data_new/new0702/train/uni_crop/band_selection/raw_band54/'
    elif numofband == 57:
        save_dir = './data_new/new0702/train/uni_crop/band_selection/raw_band57/'
    elif numofband == 60:
        save_dir = './data_new/new0702/train/uni_crop/band_selection/raw_band60/'
    elif numofband == 39:
        save_dir = './data_new/new0702/train/uni_crop/band_selection/raw_band39/'
    elif numofband == 42:
        save_dir = './data_new/new0702/train/uni_crop/band_selection/raw_band42/'
    elif numofband == 45:
        save_dir = './data_new/new0702/train/uni_crop/band_selection/raw_band45/'
    elif numofband == 48:
        save_dir = './data_new/new0702/train/uni_crop/band_selection/raw_band48/'

    else:
        print('\033[33m Number of band selecttion cannot choose...', '\033[0m')
        exit()

    datalist = get_datalist(dir, ".npy")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ne.set_num_threads(8)


    for path in tqdm(datalist,ncols = 70):
        savename = os.path.splitext(os.path.basename(path))[0]

        data = np.load(path)
        H, W,_ = data.shape
        bands = np.linspace(0, 146, numofband, dtype=int)
        train_data = np.zeros((H, W, numofband))

        for i,j in zip(range(numofband),bands):
            train_data[:, :, i] = data[:, :, j]            ### ### 之前有錯，已改成zip

        np.save(save_dir + savename,train_data.astype(np.float32))         ### float32要記得!! 不知為啥存的自動變float64

    print('\033[33m Finish band_%s selection...'%numofband, '\033[0m')

#
# uni_band_selection(51)
# uni_band_selection(54)
# uni_band_selection(57)


def band_selection():            ### 10/29  from 147 to do
    #dir = './data_new/new0702/train/uni_crop/UBS9_result/'                ### 換資料要記得改
    dir = 'J:\\myproject\\data_new\\new0702\\train\\uni_crop\\raw_ROI\\'

    #save_dir = './data_new/new0702/train/uni_crop/UBS9_result/'         ### 要記得改
    save_dir = 'J:\\myproject\\data_new\\new0702\\train\\uni_crop\\FNG_BS\\raw_ROI_60bands\\'

    datalist = get_datalist(dir, ".npy")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ne.set_num_threads(8)

    # datalist = random.sample(datalist, 5)


    for path in tqdm(datalist,ncols = 70):
        savename = os.path.splitext(os.path.basename(path))[0]

        data = np.load(path)
        H, W,_ = data.shape
        # rgb_NIR = [38,18,1,88,108,125]
        # nir_bands = [88,108,125]
        # nir_6_bands = [80, 88, 95, 108, 116, 125]
        # nir_9_bands = [73, 80, 88, 95, 108, 116, 125, 137, 146]
        # omp_9_bs = [3, 25, 59, 73, 80, 90, 104, 116, 143]
        #rgb = [38,18,1]
        # FNG_12BS = [14,24,37,46,57,68,83,91,101,117,126,136]
        # FNG_15BS = [12,13,25,39,46,55,68,77,80,91,101,108,117,139,140]
        # FNG_18BS = [10,15,25,30,32,47,54,67,69,76,87,90,101,106,117,126,140,141]
        # FNG_21BS = [0,14,16,24,33,34,48,50,59,67,75,77,89,91,100,106,118,120,126,140,142]
        # FNG_24BS = [1,7,15,18,24,37,38,47,54,55,67,71,78,83,91,100,103,105,112,117,124,133,136,143]
        # FNG_27BS = [1,12,14,19,24,31,38,40,47,52,59,63,67,75,77,85,91,94,103,104,109,118,120,126,132,136,143]
        # FNG_30BS = [0,8,14,17,21,25,33,34,41,47,50,54,61,67,71,77,82,85,87,95,102,103,108,114,117,124,130,134,136,143]
        # FNG_33BS = [1,7,12,15,18,24,29,35,36,42,47,50,57,61,67,70,75,80,84,87,90,95,97,102,107,113,117,123,126,133,137,138,144]
        # FNG_36BS = [0,7,13,15,18,23,24,31,36,38,43,47,51,55,57,63,64,70,77,79,84,90,91,96,101,102,106,112,116,117,124,130,132,137,138,144]
        # FNG_39BS = [0,5,11,12,16,22,25,28,33,34,40,44,47,51,55,57,62,67,70,74,78,79,84,90,91,95,101,102,106,110,114,117,123,126,128,133,139,140,145]
        # FNG_42BS = [1,5,10,11,16,18,23,24,30,33,35,40,43,47,51,54,56,62,64,67,72,76,80,82,85,87,91,95,99,104,105,110,113,116,119,124,127,132,134,139,140,145]
        # FNG_45BS = [1,5,8,10,15,18,21,24,28,31,34,37,41,44,47,51,53,58,60,64,68,71,74,76,79,83,86,88,93,95,99,104,106,109,112,116,117,123,125,130,132,136,138,140,146]
        # FNG_48BS = [0,6,9,11,14,17,22,23,26,29,32,35,38,41,44,47,51,54,56,59,62,64,69,72,75,78,82,84,87,90,93,96,99,102,105,108,111,114,117,120,123,126,130,132,136,138,140,145]
        # FNG_51BS = [1,4,8,10,13,17,18,24,28,29,30,33,36,39,41,44,47,51,53,56,59,61,64,68,71,74,76,80,82,84,87,90,94,95,99,102,105,107,110,113,116,119,123,125,128,131,133,137,139,142,145]
        # FNG_54BS = [16,18,23,25,26,27,28,29,30,31,32,33,34,37,40,42,44,48,50,53,55,59,66,67,68,70,73,75,77,80,82,85,88,91,93,95,99,101,104,107,110,112,116,118,120,123,126,128,131,134,137,139,142,145]
        # FNG_57BS = [67,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,112,114,115,116,117,118,119,120,121,122,123,124,126,132,133,134,135,136,137,138,140,143,145]
        FNG_60BS = [38,47,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,82,84,85,86,87,88,89,90,91,92,93,94,95,98,106,107,108,109,110,111,117,123,124,125,126,127,128,129,130,131,132,136,138,140,143,145]
        train_data = np.zeros((H, W, len(FNG_60BS)))
        for i,j in zip(range(len(FNG_60BS)),FNG_60BS):
            train_data[:, :, i] = data[:, :, j]            ### ### 之前有錯，已改成zip

        # misc.imsave(save_dir + '%s.png'%savename, train_data)       ##存rgb用的
        np.save(save_dir + savename,train_data.astype(np.float32))          ### float32要記得!! 不知為啥存的自動變float64

    print('\033[33m Finish nir band selection...', '\033[0m')

# band_selection()

def cropped2pca(dir,n_component):#),save_path,n_component):         ### crop後獨立的patch 做PCA

    ori_data_list = get_datalist(dir, ".npy")
    save_path = './data_new/new0702/train/uni_crop/pca_ROI/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # ori_data_list = random.sample(ori_data_list, 7)
    temp = []

    for raw_path in tqdm(ori_data_list, ncols = 70):
        raw_data = np.load(raw_path)
        save_name = os.path.splitext(os.path.basename(raw_path))[0]

        pca_data, _, pca_varsum = doPCA(raw_data, n_components=n_component)
        temp.append(pca_varsum)
        # snv_data = snv(raw_data)


        np.save(save_path+save_name+'_pca',pca_data)

    print('\033[33m --------------------------------------', '\033[0m')
    print('pca_varsum mean:', np.mean(temp, axis=0))
    print('\033[33m Finish ...', '\033[0m')


################################################################
# dir = './data_new/new0702/train/uni_crop/raw_ROI/'
# cropped2pca(dir,6)

#################################################################

def data2dataset(Mode=True):            ### 將篩選完的data 整理成dataset

    if Mode is True:
        dir = './data_new/croped_data/train_fullband_10class'  ### 有新切資料要記得改
    else:
        dir = './data_new/croped_data/test_fullband_10class'                    ### 有新切資料要記得改

    save_path = './data_new/'
    precessed_data = None
    temp = []

    for file in tqdm(os.listdir(dir), ncols=70):
        if file.endswith(".npz"):
            cub = np.load(os.path.join(dir, file))
            if Mode is True:
                data = cub['train_data']
                label = cub['train_label']
            else:
                data = cub['test_data']
                label = cub['test_label']

            data = np.expand_dims(data, axis=0)  ### 將(100,100,6) --> (1,100,100,6)

            if precessed_data is None:
                precessed_data = data
                #precessed_label = label
            else:
                precessed_data = np.concatenate((precessed_data, data), axis=0)
            temp.append(label)      ### 0維不能用concat

    total_nbofdata = len(precessed_data)
    ## 做shuffle 打亂
    idx = np.arange(total_nbofdata)
    np.random.shuffle(idx)
    precessed_data = [precessed_data[i] for i in idx]
    temp = [temp[i] for i in idx]

    precessed_dataset = np.asarray(precessed_data)
    precessed_label = np.asarray(temp)

    print('dataset shape:',precessed_dataset.shape)
    print('label shape:', precessed_label.shape)

    if Mode is True:
        np.savez(save_path + 'train_fullband_10class', train_dataset=precessed_dataset, train_labelset=precessed_label)           ## 新切資料檔名記得改
    else:
        np.savez(save_path + 'test_fullband_10class', test_dataset=precessed_dataset, test_labelset=precessed_label)         ## 新切資料檔名記得改

    print('\033[33m Finish creating dataset...', '\033[0m')



def classfy_img():          ### 自動分類rgb到資料夾
    import shutil
    MyPath = './data_new/croped_data/test_rgb'  # 當下目錄

    key = ["Bromeliads","Calathea_lancifolia","Poinsettia","Malus_spectabilis","Podocarpaceae","Sansevieria"]

    # KeyWord = input = '請輸入檔案關鍵字:'

    for root, dirs, files in os.walk(MyPath):
        for i in files:
            FullPath = os.path.join(root, i)  # 獲取檔案完整路徑
            FileName = os.path.join(i)  # 獲取檔案名稱

            for KeyWord in key:
                if KeyWord in FullPath:

                    if not os.path.exists(MyPath + '/' + KeyWord + '/' + FileName):
                        if not os.path.exists(os.path.join(MyPath,KeyWord)):
                            os.makedirs(os.path.join(MyPath,KeyWord))
                        shutil.move(FullPath, MyPath + '/' + KeyWord)
                        #print('成功將', FileName, '移動至', KeyWord, '資料夾')
                    else:
                        print(FileName, '已存在，故不執行動作')


def creat_txt(a):
    dir = 'J:/myproject/data_new/new0702/train/uni_crop/average_RGB/'
    alist = get_datalist(dir, '.npy')

    savepath = 'J:/myproject/data_new/new0702/train/uni_crop/average_RGB_txt/'
    num_class = '30'


    if not os.path.exists(savepath):
        os.makedirs(savepath)

    total_count = len(alist)
    test_count = int(total_count * 0.2)  ## 20 %

    print("It's %s class" %int(total_count/50))

    width = int(total_count / test_count)
    test_list = []
    train_list = []

    for i in range(total_count):
        if (i+int(a)) % width == 0:              ### 從0開始: 0~4
            test_list.append(alist[i])
        else:
            train_list.append(alist[i])


    # '''
    with open(savepath + 'Average_RGB_class_30_train_%s.txt'%(str(a+1)),'w') as f:
        for i in train_list:
            f.write(str(i)+'\n')
    with open(savepath + 'Average_RGB_class_30_test_%s.txt'%(str(a+1)),'w') as f:
        for i in test_list:
            f.write(str(i)+'\n')

# creat_txt(2)

# for i in range(5):
#     creat_txt(i)

# creat_data(Mode=False)
# data2np(Mode=False)
# crop_data(Mode=False)
# data2dataset(Mode=True)
#data2dataset_rgb(Mode=False)
# classfy_img()


def do_ROI(threshold=0.2):
    dir = './data_new/new0702/train/uni_crop/raw/'
    alist = get_datalist(dir, '.npy')
    savepath = './data_new/new0702/train/uni_crop/raw_ROI/'

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    # alist = random.sample(alist, 7)

    for path in tqdm(alist,ncols = 70):
        data = np.load(path)
        ROI = data[:,:,146] - data[:,:,0]
        save_name = os.path.splitext(os.path.basename(path))[0]

        for i in range(400):
            for j in range(400):
                if ROI[i][j] < threshold:
                    ROI[i][j] = 0
                else:
                    ROI[i][j] = 1

        out = np.zeros_like(data, dtype=np.float32)
        for i in range(147):
            out[:, :, i] = data[:, :, i] * ROI

        # print(out.shape)
        np.save(savepath + save_name ,out)

# do_ROI(threshold=0.2)

