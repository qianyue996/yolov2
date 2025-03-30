from torch.utils.data.dataloader import DataLoader
from torch import optim
import torch
import time
from torch.utils.tensorboard import SummaryWriter
import os
import sys
from tqdm import tqdm

from src.dataset import YoloVOCDataset
from src.model import Yolov2

class Trainer():
    def __init__(self):
        super().__init__()
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.IMG_SIZE=416
        self.S=13
        self.C=20
        self.LAMBDA_COORD=5
        self.LAMBDA_NOOBJ=0.5
        self.lr=3e-5
        self.batch_size=128
        self.start_epoch=0
        self.epochs=300
        self.number_anchors=5

        self.grid_size=self.IMG_SIZE//self.S
        self.losses=[]
        self.checkpoint=None

    def setup(self):
        # 加载数据集
        ds=YoloVOCDataset(self.IMG_SIZE,self.S,self.C,self.number_anchors)
        self.dataloader=DataLoader(ds,batch_size=self.batch_size,shuffle=True)
        self.anchor_boxes=ds.anchor_boxes
        # 模型初始化
        self.model=Yolov2(self.S,self.C,self.anchor_boxes).to(self.device)
        self.optimizer=optim.Adam([param for param in self.model.parameters() if param.requires_grad],lr=self.lr)
        for param in self.model.backbone.parameters():
            param.requires_grad = False

        # 尝试从上次训练结束点开始
        try:
            self.checkpoint=torch.load('checkpoint.pth')
        except Exception as e:
            pass
        if self.checkpoint:
            self.model.load_state_dict(self.checkpoint['model'])
            self.optimizer.load_state_dict(self.checkpoint['optimizer'])
            self.start_epoch = self.checkpoint['epoch'] + 1

        # tensorboard
        self.writer=SummaryWriter(f'runs/{time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime())}')

        self.model.train()

    def train(self):
        for epoch in range(self.start_epoch,self.epochs):
            batch_avg_loss=0
            with tqdm(self.dataloader, disable=False) as bar:
                for batch_x,batch_y in bar:
                    loss=torch.tensor(0)
                    batch_x,batch_y=batch_x.to(self.device),batch_y.to(self.device)
                    batch_output=self.model(batch_x)
                    loss=self.compute_loss(loss,batch_x,batch_y,batch_output)
                    loss=loss/len(batch_x)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    batch_avg_loss+=loss.item()
                    bar.set_postfix({'epoch':epoch,
                                     'loss:':loss.item()})
            batch_avg_loss=batch_avg_loss/len(self.dataloader)
            tqdm.write(f"本epoch平均损失为: {batch_avg_loss}")
            self.writer.add_scalar('epoch_loss',batch_avg_loss,epoch)
            self.losses.append(batch_avg_loss)
            self.save_best_model(epoch=epoch)

    def compute_loss(self,loss,batch_x,batch_y,batch_output):
        for i in range(len(batch_x)):
            x=batch_x[i]
            y=batch_y[i]
            output=batch_output[i]
            # foreach grid
            for row in range(self.S):
                for col in range(self.S):
                    pred_grid=output[row,col]
                    target_grid=y[row,col]
                    if not target_grid[4]>0:  # no object in this grid
                        for num_anchor in range(self.number_anchors):
                            loss_c_noobj=torch.nn.BCELoss()(pred_grid[num_anchor*(5+self.C)+4],target_grid[4])  # no object in grid,so target c is 0
                            loss=loss+loss_c_noobj
                        continue
                    # IOU
                    for num_anchor in range(self.number_anchors):
                        if num_anchor==0:
                            iou_bbox1=self.compute_iou(row,col,pred_grid[:4],target_grid[:4],
                                                       self.anchor_boxes[num_anchor])
                        if num_anchor==1:
                            iou_bbox2=self.compute_iou(row,col,pred_grid[num_anchor*(5+self.C):num_anchor*(5+self.C)+4],
                                                       target_grid[:4],self.anchor_boxes[num_anchor])
                        if num_anchor==2:
                            iou_bbox3=self.compute_iou(row,col,pred_grid[num_anchor*(5+self.C):num_anchor*(5+self.C)+4],
                                                       target_grid[:4],self.anchor_boxes[num_anchor])
                        if num_anchor==3:
                            iou_bbox4=self.compute_iou(row,col,pred_grid[num_anchor*(5+self.C):num_anchor*(5+self.C)+4],
                                                       target_grid[:4],self.anchor_boxes[num_anchor])
                        if num_anchor==4:
                            iou_bbox5=self.compute_iou(row,col,pred_grid[num_anchor*(5+self.C):num_anchor*(5+self.C)+4],
                                                       target_grid[:4],self.anchor_boxes[num_anchor])
                    # 取IOU大的预测框的x,y,w,h,c
                    all_ious=[iou_bbox1,iou_bbox2,iou_bbox3,iou_bbox4,iou_bbox5]
                    max_iou,max_iou_index=torch.max(torch.stack(all_ious),0)
                    # 计算最大iou的loss
                    xywh_pred=pred_grid[max_iou_index*(5+self.C):max_iou_index*(5+self.C)+4]
                    c_obj=pred_grid[max_iou_index*(5+self.C):max_iou_index*(5+self.C)+5][4]
                    iou_obj=max_iou
                    # 计算loss
                    loss_xywh_pred_anchors=[]
                    for num_anchor in range(self.number_anchors):
                        wh_anchor=self.anchor_boxes[num_anchor]
                        loss_xywh_pred_anchor=((xywh_pred[0]-row-target_grid[0])**2+
                                                (xywh_pred[1]-col-target_grid[1])**2+
                                                (torch.sqrt(xywh_pred[2])-torch.sqrt(torch.as_tensor(wh_anchor[0])))**2+
                                                (torch.sqrt(xywh_pred[3])-torch.sqrt(torch.as_tensor(wh_anchor[1])))**2)
                        loss_xywh_pred_anchors.append(loss_xywh_pred_anchor)
                    loss_xywh_pred_targ=((xywh_pred[0]-row-target_grid[0])**2+
                                         (xywh_pred[1]-col-target_grid[1])**2+
                                         (torch.sqrt(xywh_pred[2])-torch.sqrt(target_grid[2]))**2+
                                         (torch.sqrt(xywh_pred[3])-torch.sqrt(target_grid[3]))**2)
                    loss_c_obj=torch.nn.BCELoss()(c_obj,target_grid[4])
                    loss_class=((pred_grid[max_iou_index*(5+self.C)+5:max_iou_index*(5+self.C)+5+self.C]-
                                 target_grid[max_iou_index*(5+self.C)+5:max_iou_index*(5+self.C)+5+self.C])**2).sum()
                    loss=(loss+
                          loss_xywh_pred_targ*self.LAMBDA_COORD+
                          loss_xywh_pred_anchors[0]*self.LAMBDA_COORD+
                          loss_xywh_pred_anchors[1]*self.LAMBDA_COORD+
                          loss_xywh_pred_anchors[2]*self.LAMBDA_COORD+
                          loss_xywh_pred_anchors[3]*self.LAMBDA_COORD+
                          loss_xywh_pred_anchors[4]*self.LAMBDA_COORD+
                          +loss_c_obj+loss_class)
        return loss

    def compute_iou(self,grid_row,grid_col,xywh_pred,xywh_targ,anchor_boxes):
        # yolo coordinates
        # xcenter_a,ycenter_a是加了row或col的值 故其范围大于1    wh正常大小
        # xcenter_b,ycenter_b是没加row或col的值 其范围在0~1之间  wh正常大小
        xcenter_a,ycenter_a,w_a,h_a=xywh_pred
        xcenter_b,ycenter_b,w_b,h_b=xywh_targ
        w_anchor,h_anchor=anchor_boxes
        
        # normal coordinates
        xcenter_a,ycenter_a=xcenter_a*self.grid_size,ycenter_a*self.grid_size
        xcenter_b,ycenter_b=(grid_row+xcenter_b)*self.grid_size,(grid_col+ycenter_b)*self.grid_size
        
        # border
        xmin_a,xmax_a,ymin_a,ymax_a=xcenter_a-w_a/2,xcenter_a+w_a/2,ycenter_a-h_a/2,ycenter_a+h_a/2
        xmin_b,xmax_b,ymin_b,ymax_b=xcenter_b-w_anchor/2,xcenter_b+w_anchor/2,ycenter_b-h_anchor/2,ycenter_b+h_anchor/2
        
        # IOU
        inter_xmin=max(xmin_a,xmin_b)
        inter_xmax=min(xmax_a,xmax_b)
        inter_ymin=max(ymin_a,ymin_b)
        inter_ymax=min(ymax_a,ymax_b)
        if inter_xmax<inter_xmin or inter_ymax<inter_ymin:
            return torch.as_tensor(0).to(self.device)

        inter_area=(inter_xmax-inter_xmin)*(inter_ymax-inter_ymin) # 交集
        union_area=w_a*h_a+w_anchor*h_anchor-inter_area # 并集

        return torch.as_tensor(inter_area/union_area) # IOU

    def save_best_model(self,epoch):
        if len(self.losses)==1 or self.losses[-1]<self.losses[-2]: # 保存更优的model
            checkpoint={
                'model':self.model.state_dict(),
                'optimizer':self.optimizer.state_dict(),
                'epoch':epoch
            }
            torch.save(checkpoint,'.checkpoint.pth')
            os.replace('.checkpoint.pth','checkpoint.pth')

        EARLY_STOP_PATIENCE=5   # 早停忍耐度
        if len(self.losses)>=EARLY_STOP_PATIENCE:
            early_stop=True
            for i in range(1,EARLY_STOP_PATIENCE):
                if self.losses[-i]<self.losses[-i-1]:
                    early_stop=False
                    break
                if early_stop:
                    print(f'early stop, final loss={self.losses[-1]}')
                    sys.exit()
    

if __name__ == '__main__':
    trainer=Trainer()
    trainer.setup()
    trainer.train()