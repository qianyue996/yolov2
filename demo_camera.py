import cv2 as cv
import torch
from torchvision.transforms import ToTensor

from model import Yolov2
from dataset import YoloVOCDataset

device='cuda' if torch.cuda.is_available() else 'cpu'
cap = cv.VideoCapture(0)
IMG_SIZE=416
S=13
C=20

ds=YoloVOCDataset()
anchors=ds.anchor_boxes

model=Yolov2(S,C,anchors).to(device)
model.load_state_dict(torch.load('checkpoint.pth',map_location=device)['model'])
model.eval()

if __name__=='__main__':
    def predict(img,output,anchors,IMG_SIZE,S,C,conf_thresh=0.8,nms_thresh=0.9):
        boxes,conf,class_ids=decode(output,anchors,img,IMG_SIZE,S,C)
        mask=conf>=conf_thresh
        boxes=boxes[mask]
        conf=conf[mask]
        class_ids=class_ids[mask]

        # 按类别执行NMS（跨grid和anchor去重）
        final_detections = []
        for cls_id in torch.unique(class_ids):
            cls_mask = (class_ids == cls_id)
            cls_boxes = boxes[cls_mask]
            cls_scores = conf[cls_mask]
            
            keep_indices = nms(cls_boxes, cls_scores, nms_thresh)
            for idx in keep_indices:
                final_detections.append([
                    *cls_boxes[idx].tolist(),
                    cls_scores[idx].item(),
                    cls_id.item()
                ])

        _final_detections=[]
        for i in final_detections:
            if i[0]>0 and i[1]>0 and i[2]>0 and i[3]>0:
                _final_detections.append(i)

        for i in _final_detections:
            objname=ds.id2name[i[-1]]
            cv.rectangle(img,(int(i[0]),int(i[1])),(int(i[2]),int(i[3])),color=(0,0,255))
            # cv.addText(img,objname,(int(i[0])+5,int(i[3])),color=(0,0,255))
    def decode(output,anchors,img,IMG_SIZE,S,C):    
        grid_size=IMG_SIZE//S
        output=output.view(S,S,len(anchors),5+C)

        # 坐标还原 -> 416x416
        x,y=output[...,0]*grid_size,output[...,1]*grid_size
        w,h=output[...,2],output[...,3]
        boxes=torch.cat([x-w/2,y-h/2,x+w/2,y+h/2],dim=-1)

        # 解码置信度和类别
        conf=output[..., 4]
        class_probs=output[...,5:]
        class_ids=torch.argmax(class_probs,dim=-1)

        return boxes.view(-1,4),conf.view(-1),class_ids.view(-1)

    def nms(boxes, scores, iou_threshold=0.5):
        """
        Non-Maximum Suppression (NMS) for YOLOv2 output boxes
        :param boxes: Tensor[N,4] (x1,y1,x2,y2格式)
        :param scores: Tensor[N] (置信度分数)
        :param iou_threshold: IoU阈值
        :return: 保留的框的索引列表
        """
        # 按置信度降序排序
        keep = []
        order = torch.argsort(scores, descending=True)
        
        while order.numel() > 0:
            i = order[0]
            keep.append(i)
            
            if order.numel() == 1:
                break
            
            # 计算当前框与其他框的IoU
            xx1 = torch.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = torch.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = torch.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = torch.minimum(boxes[i, 3], boxes[order[1:], 3])
            
            inter = torch.clamp(xx2 - xx1, min=0) * torch.clamp(yy2 - yy1, min=0)
            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_other = (boxes[order[1:], 2] - boxes[order[1:], 0]) * \
                        (boxes[order[1:], 3] - boxes[order[1:], 1])
            iou = inter / (area_i + area_other - inter)
            
            # 保留IoU小于阈值的框
            mask = iou <= iou_threshold
            order = order[1:][mask]
        
        return torch.tensor(keep, dtype=torch.long)

    while True:
        # 读取摄像头帧
        ret, frame = cap.read()
        if not ret:
            print("无法获取帧！")
            break
        frame=cv.resize(frame,(IMG_SIZE,IMG_SIZE))
        input=ToTensor()(frame)
        output=model(input.unsqueeze(0)).squeeze(0)

        predict(frame,output,anchors,IMG_SIZE,S,C)
        frame=cv.resize(frame,(512,512))
        cv.namedWindow('Camera', cv.WINDOW_NORMAL)
        cv.imshow('Camera', frame)
        if cv.waitKey(1) == ord('q'):
            break
    # 释放资源
    cap.release()
    cv.destroyAllWindows()