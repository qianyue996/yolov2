from torchvision.datasets import VOCDetection
from torch.utils.data import Dataset
import torch
from torchvision.transforms import ToTensor
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
import numpy as np

from src.kmeans_manual import kmeans_manual

class YoloVOCDataset(Dataset):
    def __init__(self, IMG_SIZE=416,S=7,C=20,number_anchors=5,image_set='train'):
        super().__init__()
        # ------------------- Basic parameters -------------------
        self.IMG_SIZE=IMG_SIZE
        self.S=S
        self.C=C
        self.number_anchors=number_anchors
        # ------------------- Basic parameters -------------------
        self.voc_ds=VOCDetection(root='data',year='2012',image_set=image_set,download=False)
        classdict=set()
        self.anchor_boxes=[]
        for _,label in tqdm(self.voc_ds, desc="数据集处理中"):
            for obj in label['annotation']['object']:
                classdict.add(obj['name'])
                xmin,ymin=int(obj['bndbox']['xmin']),int(obj['bndbox']['ymin'])
                xmax,ymax=int(obj['bndbox']['xmax']),int(obj['bndbox']['ymax'])
                w=xmax-xmin
                h=ymax-ymin
                self.anchor_boxes.append([w,h])
                names=sorted(list(classdict))
        self.anchor_boxes=kmeans_manual(np.array(self.anchor_boxes),self.number_anchors).tolist()
        self.id2name={i:c for i,c in enumerate(names)}
        self.name2id={c:i for i,c in self.id2name.items()}
    
    def __getitem__(self,index):
        img,label=self.voc_ds[index]
    
        x_scale=self.IMG_SIZE/img.width
        y_scale=self.IMG_SIZE/img.height
        grid_size=self.IMG_SIZE/self.S
        
        scaled_img=img.resize((self.IMG_SIZE,self.IMG_SIZE))
        x=ToTensor()(scaled_img)
        y=torch.zeros(self.S,self.S,5*(5+self.C))
        
        for obj in label['annotation']['object']:
            box=obj['bndbox']
            classid=self.name2id[obj['name']]
            
            # normal coordinates
            xmin,ymin,xmax,ymax=int(box['xmin'])*x_scale,int(box['ymin'])*y_scale,int(box['xmax'])*x_scale,int(box['ymax'])*y_scale
            xcenter,ycenter=(xmin+xmax)/2,(ymin+ymax)/2
            width,height=xmax-xmin,ymax-ymin
            grid_i,grid_j=int(xcenter//grid_size),int(ycenter//grid_size)
            
            # yolo coordinates
            xcenter,ycenter=xcenter%grid_size/grid_size,ycenter%grid_size/grid_size
            width,height=width*x_scale,height*y_scale
            
            # targets
            for num_anchor in range(self.number_anchors):
                y[grid_i,grid_j,num_anchor*(5+self.C):num_anchor*(5+self.C)+5]=torch.as_tensor([xcenter,ycenter,width,height,1])   # x,y,w,h,c
                y[grid_i,grid_j,num_anchor*(5+self.C)+5+classid]=1
        return x,y # ((3,416,416),(13,13,125))
    
    def __len__(self):
        return len(self.voc_ds)

if __name__=='__main__':
    ds=YoloVOCDataset()
    dataloader=DataLoader(ds)
    for x,y in dataloader:
        a=x
        b=y
