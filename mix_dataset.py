from os import listdir
from os.path import join
import random
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import numpy as np
import cv2
import argparse
import glob
from PIL import Image
from torchvision import  transforms as T
from torchvision import  transforms




class datasets(data.Dataset):
    def __init__(self,train = True, transforms = None) :
        out_img_train = []
        gt_img_train = []
        
        if train :
            img = glob.glob(r'C:\Users\admin\Desktop\mix_dataset\train\*.png')          
        else:
            img = glob.glob(r'C:\Users\admin\Desktop\mix_dataset\test\*.png')
   
        #A
        out_img_train = [lab for lab in img if '_out'  in lab]
        #print(out_img_train)
        #B
        gt_img_train = [lab for lab in img if '_out' not in lab]
        #print(gt_img_train)
        self.train = train
        self.gt_img_train = gt_img_train 
        self.out_img_train = out_img_train  
        
        self.transforms = transforms

        
    def __getitem__(self,index):
        
        filename = self.out_img_train[index][-14:-10]
        label_path = self.out_img_train[index][:-10]
        label = Image.open(label_path+'.png').convert('RGB')
        blur = Image.open(self.out_img_train[index]).convert('RGB')
        filename = self.out_img_train[index][-14:]
        

        
        
        if self.transforms:
            blur = self.transforms(blur)
            label = self.transforms(label)

        return blur, label, filename
    
    def __len__(self):

        return len(self.out_img_train)






transforms = T.Compose([T.ToTensor(),T.Normalize([0.5 ,0.5 ,0.5],[0.5 ,0.5, 0.5])])

def get_training_testing_set():
    train_dataset = datasets( True ,transforms)
    test_dataset = datasets( False ,transforms)
    #print('get_mean_and_std',get_mean_and_std(train_dataset),'   ',get_mean_and_std(test_dataset))
    # print('mean_std_v3',mean_std_v3(train_dataset),'   ',mean_std_v3(test_dataset))
    #[len(dataset)-3,3]
    train_dataloader = data.DataLoader(train_dataset, batch_size=2, shuffle=True,num_workers=0,pin_memory=True)
    test_dataloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False,num_workers=0,pin_memory=True)
    return train_dataloader, test_dataloader