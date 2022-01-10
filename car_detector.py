import time
import copy
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import alexnet
import torchvision.transforms as transforms
import selectivesearch

import utils.util as util

def get_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_transform():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform

def get_model(device=None):
    model = alexnet()
    num_classes = 2
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, num_classes)
    model.load_state_dict(torch.load('./models/best_linear_svm_alexnet_car.pth', map_location=torch.device('cpu')))
    model.eval()

    for param in model.parameters():
        param.requires_grad = False
    if device:
        model = model.to(device)

    return model

def draw_box_with_text(img, rect_list, score_list):
    """
    :param img:
    :param rect_list:
    :param score_list:
    :return:
    """
    for i in range(len(rect_list)):
        xmin, ymin, xmax, ymax = rect_list[i]
        score = score_list[i]

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=(0, 0, 255), thickness=1)
        cv2.putText(img, "CAR:{:.3f}".format(score), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def nms(rect_list, score_list):
    """
    """
    nms_rects = list()
    nms_scores = list()

    rect_array = np.array(rect_list)
    score_array = np.array(score_list)

    idxs = np.argsort(score_array)[::-1]
    rect_array = rect_array[idxs]
    score_array = score_array[idxs]

    thresh = 0.3
    while len(score_array) > 0:
        nms_rects.append(rect_array[0])
        nms_scores.append(score_array[0])
        rect_array = rect_array[1:]
        score_array = score_array[1:]

        length = len(score_array)
        if length <= 0:
            break

        iou_scores = util.iou(np.array(nms_rects[len(nms_rects) - 1]), rect_array)
        idxs = np.where(iou_scores < thresh)[0]
        rect_array = rect_array[idxs]
        score_array = score_array[idxs]

    return nms_rects, nms_scores

def preds(img, svm_thresh, rects, callback=None):
    score_list = list()
    positive_list = list()
    device = get_device()
    transform = get_transform()
    model = get_model(device = device)
    dst = copy.deepcopy(img)
    nRects = len(rects)

    start = time.time()
    percent = 0
    for i,rect in enumerate(rects):
        
        if (percent<i*10//nRects):
            percent = i/nRects
            if (callback is not None):
                callback(percent)
        
        xmin, ymin, xmax, ymax = rect
        rect_img = img[ymin:ymax, xmin:xmax]

        rect_transform = transform(rect_img).to(device)
        output = model(rect_transform.unsqueeze(0))[0]

        if torch.argmax(output).item() == 1:
            probs = torch.softmax(output, dim=0).cpu().numpy()

            if probs[1] >= svm_thresh:
                score_list.append(probs[1])
                positive_list.append(rect)
                print(rect, output, probs)
    end = time.time()
    print('detect time: %d s' % (end - start))

    nms_rects, nms_scores = nms(positive_list, score_list)
    print(nms_rects)
    print(nms_scores)
    draw_box_with_text(dst, nms_rects, nms_scores)
    
    return dst

def get_ss(img,ss_mode):
    gs = selectivesearch.get_selective_search()
    
    if (ss_mode=="Single"):
        selectivesearch.config(gs, img, strategy='s')
    elif (ss_mode=="Fast"):
        selectivesearch.config(gs, img, strategy='f')
    elif (ss_mode=="Quality"):
        selectivesearch.config(gs, img, strategy='q')
    
    rects = selectivesearch.get_rects(gs)
    return rects