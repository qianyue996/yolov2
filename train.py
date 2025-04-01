from torch.utils.data.dataloader import DataLoader
from torch import optim
import torch
import time
from torch.utils.tensorboard import SummaryWriter
import os
import sys
from tqdm import tqdm

from src.dataset import YoloVOCDataset
from src.darknet import YOLOv2

class Trainer():
    def __init__(self):
        super().__init__()
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.IMG_SIZE=416
        self.S=13
        self.C=20
        self.LAMBDA_COORD=1
        self.LAMBDA_OBJ=5
        self.LAMBDA_NOOBJ=1
        self.LAMBDA_ANCB=0.01
        self.lr=3e-5
        self.batch_size=2
        self.start_epoch=0
        self.epochs=300
        self.number_anchors=5

        self.grid_size=self.IMG_SIZE/self.S
        self.step=0
        self.losses=[]
        self.checkpoint=None

    def setup(self):
        # 加载数据集
        ds=YoloVOCDataset()
        self.dataloader=DataLoader(ds,batch_size=self.batch_size,shuffle=True)
        self.anchor_boxes=ds.anchor_boxes
        # 模型初始化
        self.model=YOLOv2().to(self.device)
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
                    batch_x,batch_y=batch_x.to(self.device),batch_y.to(self.device)
                    batch_output=self.model(batch_x)
                    loss=self.compute_loss(batch,batch_output,batch_y)
                    loss=loss/len(batch_x)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    batch_avg_loss+=loss.item()
                    bar.set_postfix({'epoch':epoch,
                                     'bs_avg_loss:':loss.item()})
            epoch_avg_loss=batch_avg_loss/len(self.dataloader)
            tqdm.write(f"本epoch平均损失为: {epoch_avg_loss}")
            self.writer.add_scalar('epoch_loss',epoch_avg_loss,epoch)
            self.losses.append(batch_avg_loss)
            self.save_best_model(epoch=epoch)

    def compute_loss(self,batch,batch_output,batch_y,thresh=0.6):
        loss=torch.tensor(0)
        # share params
        anchor_boxes=torch.as_tensor(self.anchor_boxes,device=batch_output.device)  # anchor boxes
        batch_output[...,0]=torch.sigmoid(batch_output[...,0])
        batch_output[...,1]=torch.sigmoid(batch_output[...,1])
        batch_output[...,2]=torch.exp(batch_output[...,2])*anchor_boxes[:,0].view(1,1,1,-1)
        batch_output[...,3]=torch.exp(batch_output[...,3])*anchor_boxes[:,1].view(1,1,1,-1)
        batch_output[...,4]=torch.sigmoid(batch_output[...,4])

        # IOU
        iou=self.compute_iou(batch_y,anchor_boxes)
        noobj_mask=iou<=thresh
        obj_mask=iou>thresh

        # no object's grid
        # noobj_mask=batch_y[...,4]==0
        noobj_loss=((batch_output[noobj_mask][...,4]-batch_y[noobj_mask][...,4])**2).sum()

        # has object's grid
        # for each grid have five anchor boxes
        # loss=pred and anchor boxes position loss, pred and target position loss, classes loss, confidence loss
        # obj_mask=batch_y[...,4]==1
        # batch_output[obj_mask]
        # batch_y[obj_mask]

        # share params
        pred_x=batch_output[obj_mask][...,0] # pred x
        pred_y=batch_output[obj_mask][...,1] # pred y
        pred_w=batch_output[obj_mask][...,2] / self.IMG_SIZE # pred w
        pred_h=batch_output[obj_mask][...,3] / self.IMG_SIZE # pred h
        targ_x=batch_y[obj_mask][...,0]       # targ x
        targ_y=batch_y[obj_mask][...,1]       # targ y
        targ_w=batch_y[obj_mask][...,2]       # targ w
        targ_h=batch_y[obj_mask][...,3]       # targ h

        self.step+=batch
        if self.step<=12800:
            # pred and anchor
            loss_pred_anchor_x=((pred_x-targ_x)**2).sum()
            loss_pred_anchor_y=((pred_y-targ_y)**2).sum()
            loss_pred_anchor_w=((pred_w-anchor_boxes[...,0].view(-1,1) / self.IMG_SIZE)**2).sum()
            loss_pred_anchor_h=((pred_h-anchor_boxes[...,1].view(-1,1) / self.IMG_SIZE)**2).sum()
            loss=loss+(loss_pred_anchor_x+loss_pred_anchor_y+loss_pred_anchor_w+loss_pred_anchor_h)*self.LAMBDA_ANCB

        # pred and target
        loss_pred_targ_x=((pred_x-targ_x)**2).sum()
        loss_pred_targ_y=((pred_y-targ_y)**2).sum()
        loss_pred_targ_w=((pred_w-targ_w)**2).sum()
        loss_pred_targ_h=((pred_h-targ_h)**2).sum()
        loss_p_t=loss_pred_targ_x+loss_pred_targ_y+loss_pred_targ_w+loss_pred_targ_h

        # confidence loss
        pred_c=batch_output[obj_mask][...,4]
        loss_c=(iou[obj_mask]-pred_c).sum()

        # pred and target

        # classes loss
        pred_classid=batch_output[obj_mask][...,5:]
        targ_classid=batch_y[obj_mask][...,5:]
        classid_loss=((pred_classid-targ_classid)**2).sum()

        loss=(noobj_loss*self.LAMBDA_NOOBJ+
              loss_p_t*self.LAMBDA_COORD+
              loss_c*self.LAMBDA_OBJ+
              classid_loss*self.LAMBDA_OBJ)
        self.writer.add_scalar('batch_noobj',noobj_loss*self.LAMBDA_NOOBJ/self.batch_size,batch)
        self.writer.add_scalar('batch_xywh_loss',loss_p_t*self.LAMBDA_COORD/self.batch_size,batch)
        self.writer.add_scalar('batch_confidence',loss_c*self.LAMBDA_COORD/self.batch_size,batch)
        self.writer.add_scalar('batch_classid',classid_loss*self.LAMBDA_OBJ/self.batch_size,batch)
        return loss

    def compute_iou(self,batch_y,anchor_boxes):
        iou = torch.zeros_like(batch_y[..., 4])
        
        mask = batch_y[...,4] > 0
        
        batch_indices, row_indices, col_indices, anc_indices = torch.where(mask)
        
        x_targ = (batch_y[batch_indices, row_indices, col_indices, :, 0] + row_indices[:, None]) * self.grid_size
        y_targ = (batch_y[batch_indices, row_indices, col_indices, :, 1] + col_indices[:, None]) * self.grid_size
        w_targ = batch_y[batch_indices, row_indices, col_indices, :, 2] * self.IMG_SIZE
        h_targ = batch_y[batch_indices, row_indices, col_indices, :, 3] * self.IMG_SIZE
        
        anchor_boxes = anchor_boxes.to(batch_y.device)
        w_anc, h_anc = anchor_boxes[:, 0], anchor_boxes[:, 1]
        
        # 计算目标框的 xmin, ymin, xmax, ymax
        xmin_targ, ymin_targ = x_targ - w_targ / 2, y_targ - h_targ / 2
        xmax_targ, ymax_targ = x_targ + w_targ / 2, y_targ + h_targ / 2
        
        # 计算 anchor 框的 xmin, ymin, xmax, ymax（anchor 框中心与目标框相同）
        xmin_anc, ymin_anc = x_targ - w_anc / 2, y_targ - h_anc / 2
        xmax_anc, ymax_anc = x_targ + w_anc / 2, y_targ + h_anc / 2
        
        # 计算交集区域 (B, S, S, 5)
        inter_xmin = torch.max(xmin_targ, xmin_anc)
        inter_ymin = torch.max(ymin_targ, ymin_anc)
        inter_xmax = torch.min(xmax_targ, xmax_anc)
        inter_ymax = torch.min(ymax_targ, ymax_anc)
        
        inter_w = (inter_xmax - inter_xmin).clamp(0)
        inter_h = (inter_ymax - inter_ymin).clamp(0)
        inter_area = inter_w * inter_h
        
        # 计算并集区域
        union_area = (w_anc * h_anc) + (w_targ * h_targ) - inter_area
        
        # 计算 IoU
        iou_values = inter_area / union_area
        
        # 选择最大 IoU 对应的 anchor 索引
        best_anchor_indices = iou_values.argmax(dim=-1)
        best_ious = iou_values.max(dim=-1).values
        
        # 将计算出的 IoU 填充回结果张量
        iou[batch_indices, row_indices, col_indices, best_anchor_indices] = best_ious
        
        return iou

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