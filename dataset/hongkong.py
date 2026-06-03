import torch
import glob
import random
import os
import numpy as np
from skimage import io
from scipy.ndimage import label as label11
from scipy.stats import mode
import albumentations as A



def get_year_type(label):
    arr_processed=label.copy()
    arr_processed[(arr_processed <= 1970) & (arr_processed > 1)] = 1
    arr_processed[(arr_processed >= 1970) & (arr_processed < 1980)] = 2
    arr_processed[(arr_processed >= 1980) & (arr_processed < 1990)] = 3
    arr_processed[(arr_processed >= 1990) & (arr_processed < 2000)] = 4
    arr_processed[(arr_processed >= 2000) & (arr_processed < 2010)] = 5
    arr_processed[(arr_processed >= 2010) & (arr_processed <= 2020)] = 6
    arr_processed[(arr_processed >= 2020)] = 0
    arr_processed[(arr_processed <0)] = 0
    return arr_processed
def get_ufz_type(arr_processed):
    arr_processed[(arr_processed >= 1) & (arr_processed <= 4)] = 0
    arr_processed[(arr_processed >= 5) & (arr_processed <= 9)] = 1
    return arr_processed

class Hongkong_dataset(torch.utils.data.Dataset):
    def __init__(self, mode,cache=False, augmentation=True):
        super(Hongkong_dataset, self).__init__()
        if mode=='test':
            MAIN_FOLDER = './my_dataset/hk_building_age/val/'
        else:
            MAIN_FOLDER = './my_dataset/hk_building_age/'+mode+'/'
        DATA_FOLDER = MAIN_FOLDER + 'image/tdop*.tif'
        self.LABEL_FOLDER = MAIN_FOLDER + 'class/'
        self.BOUNDARY_FOLDER = MAIN_FOLDER + 'mask/'
        self.HEIGHT_FOLDER = MAIN_FOLDER + 'height/'
        self.UFZ_FOLDER = MAIN_FOLDER + 'ufz/'
        self.mode = mode
        self.augmentation = augmentation
        self.cache = cache
        self.max_num=0
        self.data_files = glob.glob(DATA_FOLDER)
        if mode=='test':
            self.data_files = random.sample(self.data_files, 100)
        # if mode == 'train':
        # # List of files
        #     #self.data_files = random.sample(glob.glob(DATA_FOLDER), 2075)
        #     self.data_files = glob.glob(DATA_FOLDER)
        #     with open("train.txt", "r", encoding="utf-8") as f:
        #         self.data_files +=[line.strip() for line in f if line.strip()]
        # else:
        #     with open("test.txt", "r", encoding="utf-8") as f:
        #         self.data_files = [line.strip() for line in f if line.strip()]

        # Sanity check : raise an error if some files do not exist
        # for f in self.data_files + self.label_files:
        #     if not os.path.isfile(f):
        #         raise KeyError('{} is not a file !'.format(f))
        self.csv_data=[]
        with open('./my_dataset/hk_building_age/building_area.csv', 'r', encoding='utf-8') as f:
            # 读取每一行
            lines = f.readlines()
            # 去掉换行符，按逗号分割
            for line in lines[1:]:
                row = line.strip().split(',')
                row_num = [float(val) for val in row[1:]]
                self.csv_data.append(row_num)
        self.csv_np = np.array(self.csv_data, dtype=np.float32)

    def __len__(self):
        # Default epoch size is 10 000 samples
        return len(self.data_files)

    @classmethod
    def data_augmentation(cls, *arrays, flip=True, mirror=True):
        will_flip, will_mirror = False, False
        if flip and random.random() < 0.5:
            will_flip = True
        if mirror and random.random() < 0.5:
            will_mirror = True

        results = []
        for array in arrays:
            if will_flip:
                if len(array.shape) == 2:
                    array = array[::-1, :]
                else:
                    array = array[:, ::-1, :]
            if will_mirror:
                if len(array.shape) == 2:
                    array = array[:, ::-1]
                else:
                    array = array[:, :, ::-1]
            results.append(np.copy(array))

        return tuple(results)
    
    def generate_instance_mask(self,mask,min_count_threshold=200):
        mask_flat = mask.flatten()
        geo_instance=np.zeros((50,4),dtype=np.float32)
        unique_ids, counts = np.unique(mask_flat, return_counts=True)
        

        id_count_dict = dict(zip(unique_ids, counts))
        

        retained_ids = [
            id_ for id_ in unique_ids 
            if id_ != 0 and id_count_dict[id_] >= min_count_threshold
        ]
        
        id_mapping = {0: 0} 
        for new_id, old_id in enumerate(retained_ids, start=1):
            geo_instance[new_id-1] =self.csv_data[old_id-1]
            id_mapping[old_id] = new_id
        

        for old_id in unique_ids:
            if old_id != 0 and old_id not in retained_ids:
                id_mapping[old_id] = 0
        

        vectorized_mapping = np.vectorize(lambda x: id_mapping[x])
        reencoded_mask = vectorized_mapping(mask)
        
        
        return reencoded_mask,geo_instance, len(retained_ids)
    
    def extract_instance_masks(self,instance_id_tensor) -> dict:

        unique_ids = np.unique(instance_id_tensor)
     
        unique_ids = unique_ids[unique_ids != -1]
        
        instance_masks = []
        for ins_id in unique_ids:
            mask = (instance_id_tensor == ins_id)
            instance_masks.append(mask)
        return instance_masks

    def __getitem__(self, i):
        name=self.data_files[i].split('/')[-1].split('.')[0].replace('image', '')
        data = io.imread(self.data_files[i])[:, :, :3].transpose((2, 0, 1))
        data = 1 / 255 * np.asarray(data, dtype='float32')

        height_files = self.HEIGHT_FOLDER+name+'height.tif'
        height = io.imread(height_files)
        height = np.asarray(height, dtype='float32')
        height = height - height.min()
        height = height / 100.0  # normalize to 0-1
        height = height[np.newaxis, :, :]

        ufzs=[]
        for year in ['1990','2000','2010','2020']:
            if os.path.exists(self.UFZ_FOLDER+name+'ufz_'+year+'.tif'):
                ufz = io.imread(self.UFZ_FOLDER+name+'ufz_'+year+'.tif')
                ufz = np.asarray(ufz, dtype=np.float32)
                ufz=get_ufz_type(ufz)
                # unique_values1 = np.unique(ufz)
                # print(unique_values1)
                ufzs.append(ufz)
            else:
                ufzs.append(np.zeros((512, 512)))


        label = np.asarray(io.imread(self.LABEL_FOLDER+name+'class.tif'))
        label = label.astype(np.int64)
        
        label_id = get_year_type(label) #背景为0

        boundary_files = self.BOUNDARY_FOLDER+name+'mask.tif'
        boundary = np.asarray(io.imread(boundary_files))
        boundary = boundary.astype(np.int64)
        zero_mask = (label_id == 0)
        boundary[zero_mask] = 0
        boundary, geo_instance,instance_num = self.generate_instance_mask(boundary)


        boundary = boundary-1 #整张图的instance mask 从-1开始
        instances = self.extract_instance_masks(boundary) #转换为instance mask
        label_id[boundary == -1] = 0
        # Data augmentation
        ufzs = np.stack(ufzs, axis=0).astype(np.float32)
        if self.mode == 'train' and self.augmentation:
            data, boundary, height,label,label_id, ufzs = self.data_augmentation(data,boundary, height, label,label_id, ufzs)
        if self.mode == 'train' or self.mode == 'test':
            return (torch.from_numpy(data),
                    torch.from_numpy(data), #无用之前是instances表示每一个instance 的mask，但train里面没有用到，先放data占位
                    torch.from_numpy(height),
                    torch.from_numpy(ufzs),
                    torch.from_numpy(label_id)-1,
                    torch.from_numpy(boundary),
                    torch.from_numpy(geo_instance),
                    torch.from_numpy(label) #具体的年份，train的时候无用
                    )
        else:
            instances = np.array(instances)  
            return (torch.from_numpy(data),
                    torch.from_numpy(instances),
                    torch.from_numpy(height),
                    torch.from_numpy(ufzs),
                    torch.from_numpy(label_id)-1,
                    torch.from_numpy(boundary),
                    torch.from_numpy(geo_instance),
                    torch.from_numpy(label)
                    )



def get_mask_classes(mask, label) -> np.ndarray:

    mask_classes = np.zeros(len(mask), dtype=label.dtype)
    for i in range(len(mask)):
        mask_i = mask[i]  # (W, H)
        label_masked = label[mask_i]  # (N,)，N为mask覆盖的像素数
        
        mask_classes[i] = mode(label_masked, keepdims=False).mode
    
    return mask_classes