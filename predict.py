import torch
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms

from PIL import Image, ImageDraw, ImageFont

from src.dataset import YoloVOCDataset
from src.model import Yolov2Network

device='cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE=416
S=13
C=20
grid_size=IMG_SIZE/S

ds=YoloVOCDataset(IMG_SIZE,S,C)
anchor_boxes=ds.anchor_boxes

# 1. 加载模型
model=Yolov2Network(S,C,anchor_boxes).to(device)
model.load_state_dict(torch.load('checkpoint.pth',map_location=device)['model'])
model.eval()

def _draw(draw,pred,objname):
    xmin=int(pred[0]-pred[2]/2)
    ymin=int(pred[1]-pred[3]/2)
    xmax=int(pred[2]+pred[2]/2)
    ymax=int(pred[3]+pred[3]/2)

    xmin=min(xmin,xmax)
    ymin=min(ymin,ymax)
    xmax=max(xmin,xmax)
    ymax=max(ymin,ymax)

    draw.circle((pred[0],pred[1]),fill='red',radius=5)
    draw.rectangle((xmin,ymin,xmax,ymax),outline='red')
    draw.text((xmin+5,ymax+5),objname,fill='red')

def decode(output,draw):
    for row in range(S):
        for col in range(S):
            for anchor in range(len(ds.anchor_boxes)):
                pred=output[row,col,anchor]
                conf=pred[4]
                if conf>0.5:
                    obj_id=pred[5:].argmax()
                    objname=ds.id2name[int(obj_id)]
                    pred[0]=(torch.sigmoid(pred[0])+row)*grid_size
                    pred[1]=(torch.sigmoid(pred[1])+col)*grid_size
                    pred[2]=torch.exp(pred[2])*ds.anchor_boxes[anchor][0]
                    pred[3]=torch.exp(pred[3])*ds.anchor_boxes[anchor][1]
                    _draw(draw,pred[:4],objname)

if __name__=='__main__':
    with torch.no_grad():
        img,_=ds.voc_ds[0]
        # for img in imgs:
        img=img.resize((416,416))
        input=ToTensor()(img).unsqueeze(0)
        draw=ImageDraw.Draw(img)
        output = model(input)[0]
        decode(output,draw)
        img.show()