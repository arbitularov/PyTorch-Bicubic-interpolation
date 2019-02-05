#This method works very poorly and very slowly. See the bicubic.py file.

import tensorflow as tf
import torch
from torch.autograd import Variable
import numba


def bicubic(tensor, upsc_size = 6, interp = 'bicubic', align_corners=False, name=None):

    tensor = tensor.permute(0, 2, 3, 1)

    tfTensor = tf.convert_to_tensor(tensor)

    bicubic = tf.image.resize_bicubic(
                                            tfTensor,
                                            (upsc_size, upsc_size ),
                                            align_corners=False,
                                            name=None
                                            )
    a = tf.InteractiveSession()
    torchTensor = torch.from_numpy(bicubic.eval())
    a.close()
    torchTensor  = torchTensor.permute(0, 3, 1, 2)

    torchTensor = torchTensor.type(torch.cuda.FloatTensor)
    torchTensor = Variable(torchTensor, requires_grad=True)

    return torchTensor

torchImg = torch.FloatTensor([[[[1,1,1],[2,2,2],[3,3,3]]], [[[1,1,1],[2,2,2],[3,3,3]]]])

d = bicubic(torchImg)
