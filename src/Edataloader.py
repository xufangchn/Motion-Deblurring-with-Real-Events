import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image
import os
import numpy as np
import cv2
import math

import argparse

# convert tensor to 'png'
def show_img(img):
    img = img.cpu().detach().numpy()
    img = np.transpose(img, (0,2,3,1))
    img = np.squeeze(img)
    img = (img - np.min(img))/(np.max(img)-np.min(img))
    img = img * 255.
    return img

# dataloader
class EventData(Dataset):
    def __init__(self, opts, load_dataset_name, load_bag_name, is_load_sharp_image):
        self._opts = opts
        self._load_dataset_name = load_dataset_name
        self._load_bag_name = load_bag_name
        self._is_load_sharp_image = is_load_sharp_image
        self.bag_names, self.n_ima = self.read_file_paths(self._opts.data_folder_path, self._load_dataset_name, self._load_bag_name)

    def __getitem__(self, index):
        bag_iter = 0
        for i in self.n_ima:
            if index < i:
                break
            bag_iter += 1
        bag_iter -= 1

        bag_name = self.bag_names[bag_iter]

        if self._load_dataset_name == 'HQF':
            image_iter = (index - self.n_ima[bag_iter])*7        

        ##===================================================##
        ##******************* load events *******************##
        ##===================================================##
        if self._load_dataset_name == 'HQF':
            event01_path = os.path.join(self._opts.data_folder_path, self._load_dataset_name, bag_name, 'npy', 'event'+str(image_iter).rjust(5,'0') + '.npy')
            event12_path = os.path.join(self._opts.data_folder_path, self._load_dataset_name, bag_name, 'npy', 'event'+str(image_iter+1).rjust(5,'0') + '.npy')
            event23_path = os.path.join(self._opts.data_folder_path, self._load_dataset_name, bag_name, 'npy', 'event'+str(image_iter+2).rjust(5,'0') + '.npy')
            event34_path = os.path.join(self._opts.data_folder_path, self._load_dataset_name, bag_name, 'npy', 'event'+str(image_iter+3).rjust(5,'0') + '.npy')
            event45_path = os.path.join(self._opts.data_folder_path, self._load_dataset_name, bag_name, 'npy', 'event'+str(image_iter+4).rjust(5,'0') + '.npy')
            event56_path = os.path.join(self._opts.data_folder_path, self._load_dataset_name, bag_name, 'npy', 'event'+str(image_iter+5).rjust(5,'0') + '.npy')
        
        event_count_images01, event_time_images01 = np.load(event01_path, encoding='bytes', allow_pickle=True)
        event_count_images01 = torch.from_numpy(event_count_images01.astype(np.int16))
        event_time_images01 = torch.from_numpy(event_time_images01.astype(np.float32))

        event_count_images12, event_time_images12 = np.load(event12_path, encoding='bytes', allow_pickle=True)
        event_count_images12 = torch.from_numpy(event_count_images12.astype(np.int16))
        event_time_images12 = torch.from_numpy(event_time_images12.astype(np.float32))

        event_count_images23, event_time_images23 = np.load(event23_path, encoding='bytes', allow_pickle=True)
        event_count_images23 = torch.from_numpy(event_count_images23.astype(np.int16))
        event_time_images23 = torch.from_numpy(event_time_images23.astype(np.float32))

        event_count_images34, event_time_images34 = np.load(event34_path, encoding='bytes', allow_pickle=True)
        event_count_images34 = torch.from_numpy(event_count_images34.astype(np.int16))
        event_time_images34 = torch.from_numpy(event_time_images34.astype(np.float32))

        event_count_images45, event_time_images45 = np.load(event45_path, encoding='bytes', allow_pickle=True)
        event_count_images45 = torch.from_numpy(event_count_images45.astype(np.int16))
        event_time_images45 = torch.from_numpy(event_time_images45.astype(np.float32))

        event_count_images56, event_time_images56 = np.load(event56_path, encoding='bytes', allow_pickle=True)
        event_count_images56 = torch.from_numpy(event_count_images56.astype(np.int16))
        event_time_images56 = torch.from_numpy(event_time_images56.astype(np.float32))
        
        event_count_image01, event_time_image01 = self._read_events(event_count_images01, event_time_images01, 1)
        event_count_image12, event_time_image12 = self._read_events(event_count_images12, event_time_images12, 1)
        event_count_image23, event_time_image23 = self._read_events(event_count_images23, event_time_images23, 1)
        event_count_image34, event_time_image34 = self._read_events(event_count_images34, event_time_images34, 1)
        event_count_image45, event_time_image45 = self._read_events(event_count_images45, event_time_images56, 1)
        event_count_image56, event_time_image56 = self._read_events(event_count_images56, event_time_images56, 1)

        ##===================================================##
        ##********************* load GT *********************##
        ##===================================================##        
        if self._is_load_sharp_image:
            if self._load_dataset_name == 'HQF':
                sharp_img0_path = os.path.join(self._opts.data_folder_path, self._load_dataset_name, bag_name, 'images', str(image_iter).rjust(6,'0') + '.png')
                sharp_img1_path = os.path.join(self._opts.data_folder_path, self._load_dataset_name, bag_name, 'images', str(image_iter+1).rjust(6,'0') + '.png')
                sharp_img2_path = os.path.join(self._opts.data_folder_path, self._load_dataset_name, bag_name, 'images', str(image_iter+2).rjust(6,'0') + '.png')
                sharp_img3_path = os.path.join(self._opts.data_folder_path, self._load_dataset_name, bag_name, 'images', str(image_iter+3).rjust(6,'0') + '.png')
                sharp_img4_path = os.path.join(self._opts.data_folder_path, self._load_dataset_name, bag_name, 'images', str(image_iter+4).rjust(6,'0') + '.png')
                sharp_img5_path = os.path.join(self._opts.data_folder_path, self._load_dataset_name, bag_name, 'images', str(image_iter+5).rjust(6,'0') + '.png')
                sharp_img6_path = os.path.join(self._opts.data_folder_path, self._load_dataset_name, bag_name, 'images', str(image_iter+6).rjust(6,'0') + '.png')
                sharp_img0 = Image.open(sharp_img0_path).convert('L') 
                sharp_img1 = Image.open(sharp_img1_path).convert('L') 
                sharp_img2 = Image.open(sharp_img2_path).convert('L') 
                sharp_img3 = Image.open(sharp_img3_path).convert('L') 
                sharp_img4 = Image.open(sharp_img4_path).convert('L') 
                sharp_img5 = Image.open(sharp_img5_path).convert('L') 
                sharp_img6 = Image.open(sharp_img6_path).convert('L')

        ##===================================================##
        ##**************** load blurry image ****************##
        ##===================================================##  
        if self._load_dataset_name == 'HQF':
            blurry_img_path = os.path.join(self._opts.data_folder_path, self._load_dataset_name, bag_name, 'blur', str(image_iter).rjust(10,'0') + '.png')
            blurry_img = Image.open(blurry_img_path).convert('L') 

        if self._is_load_sharp_image:
            sharp_img0 = F.to_tensor(sharp_img0)
            sharp_img1 = F.to_tensor(sharp_img1)
            sharp_img2 = F.to_tensor(sharp_img2)
            sharp_img3 = F.to_tensor(sharp_img3)
            sharp_img4 = F.to_tensor(sharp_img4)
            sharp_img5 = F.to_tensor(sharp_img5)
            sharp_img6 = F.to_tensor(sharp_img6)

        blurry_img = F.to_tensor(blurry_img)

        event_image01 = torch.cat((event_count_image01, event_time_image01), 0)
        event_image12 = torch.cat((event_count_image12, event_time_image12), 0)
        event_image23 = torch.cat((event_count_image23, event_time_image23), 0)
        event_image34 = torch.cat((event_count_image34, event_time_image34), 0)
        event_image45 = torch.cat((event_count_image45, event_time_image45), 0)
        event_image56 = torch.cat((event_count_image56, event_time_image56), 0)

        # crop 
        if event_image01.shape[1]>self._opts.crop_sz_H and event_image01.shape[2]>self._opts.crop_sz_W:
            if self._opts.is_test:
                y = (event_image01.shape[1] - self._opts.crop_sz_H) // 2
                x = (event_image01.shape[2] - self._opts.crop_sz_W) // 2
            else:
                y = np.random.randint(low=1, high=(event_image01.shape[1]-self._opts.crop_sz_H))
                x = np.random.randint(low=1, high=(event_image01.shape[2]-self._opts.crop_sz_W))

            event_image01 = event_image01[...,y:y+self._opts.crop_sz_H,x:x+self._opts.crop_sz_W]
            event_image12 = event_image12[...,y:y+self._opts.crop_sz_H,x:x+self._opts.crop_sz_W]
            event_image23 = event_image23[...,y:y+self._opts.crop_sz_H,x:x+self._opts.crop_sz_W]
            event_image34 = event_image34[...,y:y+self._opts.crop_sz_H,x:x+self._opts.crop_sz_W]
            event_image45 = event_image45[...,y:y+self._opts.crop_sz_H,x:x+self._opts.crop_sz_W]
            event_image56 = event_image56[...,y:y+self._opts.crop_sz_H,x:x+self._opts.crop_sz_W]
 
            if self._is_load_sharp_image:
                sharp_img0 = sharp_img0[...,y:y+self._opts.crop_sz_H,x:x+self._opts.crop_sz_W]
                sharp_img1 = sharp_img1[...,y:y+self._opts.crop_sz_H,x:x+self._opts.crop_sz_W]
                sharp_img2 = sharp_img2[...,y:y+self._opts.crop_sz_H,x:x+self._opts.crop_sz_W]
                sharp_img3 = sharp_img3[...,y:y+self._opts.crop_sz_H,x:x+self._opts.crop_sz_W]
                sharp_img4 = sharp_img4[...,y:y+self._opts.crop_sz_H,x:x+self._opts.crop_sz_W]
                sharp_img5 = sharp_img5[...,y:y+self._opts.crop_sz_H,x:x+self._opts.crop_sz_W]
                sharp_img6 = sharp_img6[...,y:y+self._opts.crop_sz_H,x:x+self._opts.crop_sz_W]

            blurry_img = blurry_img[...,y:y+self._opts.crop_sz_H,x:x+self._opts.crop_sz_W]

        results = []

        results.append(event_image01)#0
        results.append(event_image12)#1
        results.append(event_image23)#2
        results.append(event_image34)#3
        results.append(event_image45)#4
        results.append(event_image56)#5

        results.append(blurry_img)#6
        
        if self._is_load_sharp_image:
            results.append(sharp_img0)#7
            results.append(sharp_img1)#8
            results.append(sharp_img2)#9
            results.append(sharp_img3)#10
            results.append(sharp_img4)#11
            results.append(sharp_img5)#12
            results.append(sharp_img6)#13

        if self._opts.is_test:
            results.append(bag_name) #14
            results.append(image_iter) #15

        return results

    def __len__(self):
        return self.n_ima[-1]

    def _read_events(self,
                     event_count_images,
                     event_time_images,
                     n_frames):
        event_count_image = event_count_images[:n_frames, :, :, :]
        event_count_image = torch.sum(event_count_image, dim=0).type(torch.float32)
        event_count_image = event_count_image.permute(2,0,1)

        event_time_image = event_time_images[:n_frames, :, :, :]
        event_time_image = torch.max(event_time_image, dim=0)[0]

        event_time_image /= torch.max(event_time_image)
        event_time_image = event_time_image.permute(2,0,1)

        return event_count_image, event_time_image

    def read_file_paths(self, data_folder_path, load_dataset_name, load_bag_name):
        bag_names = []
        
        bag_list_file = open(os.path.join(data_folder_path, load_dataset_name, load_bag_name), 'r')

        lines = bag_list_file.read().splitlines()
        bag_list_file.close()

        n_ima = [0]
        for line in lines:
            bag_name = line

            bag_names.append(bag_name)

            if load_dataset_name == 'HQF': 
                blurry_path = os.path.join(data_folder_path, load_dataset_name, bag_name, 'blur')
                blurry_files = os.listdir(blurry_path)
                blurry_num = len(blurry_files)

            if self._opts.is_test and blurry_num > 50 and load_dataset_name == 'HQF':
                blurry_num = 50

            n_ima.append(blurry_num + n_ima[-1])

        return bag_names, n_ima

if __name__ == "__main__":
    parser=argparse.ArgumentParser()

    parser.add_argument('--crop_sz_H', type=int, default=180, help='cropped image size height')
    parser.add_argument('--crop_sz_W', type=int, default=240, help='cropped image size width')
    parser.add_argument('--data_folder_path', type=str, default='../data')
    parser.add_argument('--load_dataset_name', type=str, default='HQF')
    parser.add_argument('--load_bag_name', type=str, default='test_bags.txt')
    parser.add_argument('--is_load_sharp_image', type=bool, default=True)
    parser.add_argument('--is_test', type=bool, default=True)
    
    opts=parser.parse_args() 

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    data = EventData(opts, opts.load_dataset_name, opts.load_bag_name, opts.is_load_sharp_image)
    dataloader = torch.utils.data.DataLoader(dataset=data, batch_size=1,shuffle=False)

    _iter = 0
    for i in dataloader:
        event_image = i[0]
        print(_iter, event_image.shape)
        
        _iter += 1
        input()


        