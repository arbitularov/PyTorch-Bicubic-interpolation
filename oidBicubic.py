#This code is very slow, please use the bicubic.py file.
#This file is used for evaluation.

import sys, time
import torch
import numpy as np
from torch.autograd import Variable


# Interpolation kernel
def u(s,a):
    if (abs(s) >=0) & (abs(s) <=1):
        return (a+2)*(abs(s)**3)-(a+3)*(abs(s)**2)+1
    elif (abs(s) > 1) & (abs(s) <= 2):
        return a*(abs(s)**3)-(5*a)*(abs(s)**2)+(8*a)*abs(s)-4*a
    return 0

#Paddnig
def padding(img,B,C,H,W):
    zimg = np.zeros([B,C,H+4,W+4])
    zimg[:,:C,2:H+2,2:W+2] = img
    #Pad the first/last two col and row
    zimg[:,:C,2:H+2,0:2]=img[:,:C,:,0:1]
    zimg[:,:,H+2:H+4,2:W+2]=img[:,:,H-1:H,:]
    zimg[:,:,2:H+2,W+2:W+4]=img[:,:,:,W-1:W]
    zimg[:,:C,0:2,2:W+2]=img[:,:C,0:1,:]
    #Pad the missing eight points
    zimg[0,:C,0:2,0:2]=img[0,:C,0,0]
    zimg[0,:C,H+2:H+4,0:2]=img[0,:C,H-1,0]
    zimg[0,:C,H+2:H+4,W+2:W+4]=img[0,:C,H-1,W-1]
    zimg[0,:C,0:2,W+2:W+4]=img[0,:C,0,W-1]
    return zimg

# Bicubic operation
def bicubic(img, newSize, a=-0.75):

    #Get image size
    img = img.cpu().detach().numpy()
    B, C, H, W = img.shape
    img = padding(img,B,C,H,W)

    #Create new image
    dH = newSize
    dW = newSize
    dst = np.zeros((B, C, dH, dW))


    h = 1/(newSize/H)

    for b in range(B):
        for c in range(C):
            for j in range(dH):
                for i in range(dW):
                    x, y = i * h + 2 , j * h + 2

                    x1 = 1 + x - math.floor(x)
                    x2 = x - math.floor(x)
                    x3 = math.floor(x) + 1 - x
                    x4 = math.floor(x) + 2 - x

                    y1 = 1 + y - math.floor(y)
                    y2 = y - math.floor(y)
                    y3 = math.floor(y) + 1 - y
                    y4 = math.floor(y) + 2 - y

                    mat_l = np.matrix([[u(x1,a),u(x2,a),u(x3,a),u(x4,a)]])
                    mat_m = np.matrix([[img[0,c,int(y-y1),int(x-x1)],img[0,c,int(y-y2),int(x-x1)],img[0,c,int(y+y3),int(x-x1)],img[0,c,int(y+y4),int(x-x1)]],
                                       [img[0,c,int(y-y1),int(x-x2)],img[0,c,int(y-y2),int(x-x2)],img[0,c,int(y+y3),int(x-x2)],img[0,c,int(y+y4),int(x-x2)]],
                                       [img[0,c,int(y-y1),int(x+x3)],img[0,c,int(y-y2),int(x+x3)],img[0,c,int(y+y3),int(x+x3)],img[0,c,int(y+y4),int(x+x3)]],
                                       [img[0,c,int(y-y1),int(x+x4)],img[0,c,int(y-y2),int(x+x4)],img[0,c,int(y+y3),int(x+x4)],img[0,c,int(y+y4),int(x+x4)]]])
                    mat_r = np.matrix([[u(y1,a)],[u(y2,a)],[u(y3,a)],[u(y4,a)]])
                    dst[b, c, j, i] = np.asarray( np.dot( np.dot(mat_l, mat_m), mat_r) )

    dst = torch.Tensor(dst).cuda()
    dst = dst.type(torch.cuda.FloatTensor)
    dst = Variable(dst, requires_grad=True).cuda()
    return dst
