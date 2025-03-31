from torch.utils.data.dataloader import DataLoader
from torch import optim
import torch
import time
from torch.utils.tensorboard import SummaryWriter
import os
import sys
from tqdm import tqdm

from src.dataset import YoloVOCDataset
from src.model import Yolov2Network

class Trainer():
    def __init__(self):
        super().__init__()
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.IMG_SIZE=416
        self.S=7
        self.C=20
        self.LAMBDA_COORD=5
        self.LAMBDA_NOOBJ=0.5
        self.lr=3e-5
        self.batch_size=2
        self.start_epoch=0
        self.epochs=300
        self.number_anchors=5

        self.grid_size=self.IMG_SIZE/self.S
        self.losses=[]
        self.checkpoint=None

    def setup(self):
        # 加载数据集
        ds=YoloVOCDataset(self.IMG_SIZE,self.S,self.C,self.number_anchors)
        self.dataloader=DataLoader(ds,batch_size=self.batch_size,shuffle=True)
        self.anchor_boxes=ds.anchor_boxes
        # 模型初始化
        self.model=Yolov2Network(self.S,self.C,self.anchor_boxes).to(self.device)
        self.optimizer=optim.Adam([param for param in self.model.parameters() if param.requires_grad],lr=self.lr)

        # 尝试从上次训练结束点开始
        # try:
        #     self.checkpoint=torch.load('checkpoint.pth')
        # except Exception as e:
        #     pass
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
                for batch,item in enumerate(bar):
                    batch_x,batch_y=item
                    loss=torch.tensor(0)
                    batch_x,batch_y=batch_x.to(self.device),batch_y.to(self.device)
                    batch_output=self.model(batch_x)
                    loss=self.compute_loss(batch,batch_y,batch_output)
                    loss=loss/len(batch_x)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    batch_avg_loss+=loss.item()
                    bar.set_postfix({'epoch':epoch,
                                     'bs_avg_loss:':loss.item()})
            batch_avg_loss=batch_avg_loss/len(self.dataloader)
            tqdm.write(f"本epoch平均损失为: {batch_avg_loss}")
            self.writer.add_scalar('epoch_loss',batch_avg_loss,epoch)
            self.losses.append(batch_avg_loss)
            self.save_best_model(epoch=epoch)

    def compute_loss(self,batch,batch_y,batch_output):
        batch_y=batch_y.view(self.batch_size,self.S,self.S,self.number_anchors,5+self.C)
        batch_output=batch_output.view(self.batch_size,self.S,self.S,self.number_anchors,5+self.C)  # shape: (bs,S,S,num_a,5+C)
        # no object's grid
        noobj_mask=batch_y[...,4]==0
        noobj_loss=((batch_output[...,4][noobj_mask]-batch_y[...,4][noobj_mask])**2).sum()

        # has object's grid
        # for each grid have five anchor boxes
        # loss=pred and anchor boxes position loss, pred and target position loss, classes loss, confidence loss
        obj_mask=batch_y[...,4]==1
        # batch_output[obj_mask]
        # batch_y[obj_mask]

        # share params
        pred_x=batch_output[obj_mask][...,0]  # pred x
        pred_y=batch_output[obj_mask][...,1]  # pred y
        pred_w=batch_output[obj_mask][...,2]  # pred w
        pred_h=batch_output[obj_mask][...,3]  # pred h
        targ_x=batch_y[obj_mask][...,0]       # targ x
        targ_y=batch_y[obj_mask][...,1]       # targ y
        targ_w=batch_y[obj_mask][...,2]       # targ w
        targ_h=batch_y[obj_mask][...,3]       # targ h

        # pred and anchor
        anchor_boxes=torch.as_tensor(self.anchor_boxes, device=batch_output.device)  # anchor boxes
        loss_x_pred_anchor=((pred_x-torch.trunc(pred_x)-targ_x)**2).sum()
        loss_y_pred_anchor=((pred_y-torch.trunc(pred_y)-targ_y)**2).sum()
        loss_w_pred_anchor=(pred_w.view(-1,1)*anchor_boxes[...,0].view(1,-1)).sum()
        loss_h_pred_anchor=(pred_h.view(-1,1)*anchor_boxes[...,1].view(1,-1)).sum()
        loss_p_a=loss_x_pred_anchor+loss_y_pred_anchor+loss_w_pred_anchor+loss_h_pred_anchor

        # pred and target
        loss_x_pred_targ=loss_x_pred_anchor
        loss_y_pred_targ=loss_y_pred_anchor
        loss_w_pred_targ=(pred_w*targ_w).sum()
        loss_h_pred_targ=(pred_h*targ_h).sum()
        loss_p_t=loss_x_pred_targ+loss_y_pred_targ+loss_w_pred_targ+loss_h_pred_targ

        # confidence loss
        # IOU
        group_size=int(len(batch_y[obj_mask][...,:4])/self.number_anchors)
        targ_group=batch_y[obj_mask][...,:4].view(group_size,self.number_anchors,-1)
        pred_group=batch_output[obj_mask][...,:4].view(group_size,self.number_anchors,-1)
        iou=self.compute_iou(pred_group,targ_group)
        max_iou,index=torch.max(iou,dim=1)
        pred_c=batch_output[obj_mask][...,4].view(group_size,-1)
        pred_best_iou_c=torch.gather(pred_c,dim=1,index=index.view(-1,1)).view(1,-1)
        loss_c=((max_iou-pred_best_iou_c)**2).sum()

        # classes loss
        best_iou_index=index.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,self.C)
        pred_classid=batch_output[obj_mask][...,5:].view(group_size,self.number_anchors,-1)
        targ_classid=batch_y[obj_mask][...,5:].view(group_size,self.number_anchors,-1)
        pred_classid=torch.gather(pred_classid,dim=1,index=best_iou_index).squeeze(1)
        targ_classid=torch.gather(targ_classid,dim=1,index=best_iou_index).squeeze(1)
        classid_loss=((pred_classid-targ_classid)**2).sum()

        loss=(noobj_loss*self.LAMBDA_NOOBJ+
              loss_p_a*2+
              loss_p_t*self.LAMBDA_COORD+
              loss_c+classid_loss
              )
        self.writer.add_scalar('batch_noobj',noobj_loss,batch)
        self.writer.add_scalar('batch_pred_anchor',loss_p_a,batch)
        self.writer.add_scalar('batch_pred_target',loss_p_t,batch)
        self.writer.add_scalar('batch_confidence',loss_c,batch)
        self.writer.add_scalar('batch_classid',classid_loss,batch)
        return loss

    def compute_iou(self,pred_group,targ_group):
        grid_size=self.IMG_SIZE/self.S
        # ----
        grid_row=torch.trunc(pred_group[...,0])
        grid_col=torch.trunc(pred_group[...,1])

        x_pred=pred_group[...,0]*grid_size
        y_pred=pred_group[...,1]*grid_size
        w_pred=pred_group[...,2]
        h_pred=pred_group[...,3]

        x_targ=(targ_group[...,0]+grid_row)*grid_size
        y_targ=(targ_group[...,1]+grid_col)*grid_size
        w_targ=targ_group[...,2]
        h_targ=targ_group[...,3]

        xmin_pred=x_pred-w_pred/2
        ymin_pred=y_pred-h_pred/2
        xmax_pred=x_pred+w_pred/2
        ymax_pred=y_pred+h_pred/2

        xmin_targ=x_targ-w_targ/2
        ymin_targ=y_targ-h_targ/2
        xmax_targ=x_targ+w_targ/2
        ymax_targ=y_targ+h_targ/2

        # IOU
        inter_xmin=torch.max(xmin_pred,xmin_targ)
        inter_xmax=torch.min(xmax_pred,xmax_targ)
        inter_ymin=torch.max(ymin_pred,ymin_targ)
        inter_ymax=torch.min(ymax_pred,ymax_targ)

        inter_area=(inter_xmax-inter_xmin)*(inter_ymax-inter_ymin)  # 交集
        union_area=w_pred*h_pred+w_targ*h_targ-inter_area  # 并集

        iou=inter_area/union_area

        # 无相交
        if (inter_xmax<inter_xmin).any():
            mask=inter_xmax<inter_xmin
            iou[mask]=0
        if (inter_ymax<inter_ymin).any():
            mask=inter_ymax<inter_ymin
            iou[mask]=0

        return  iou

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