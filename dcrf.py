import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils

def crf(img, prob, iter_max):
    img = np.array(img)
    # Exercise - 1 
    # prob.shape : H, W, C
    prob_tr =
    img = np.array(img)
    C, H, W = prob_tr.shape

    # 1 

    U = np.ascontiguousarray(U)
    image = np.ascontiguousarray(img)

    d = dcrf.DenseCRF2D(W, H, C)

    # 2 
    # 3 
    # 4 

    Q = d.inference(iter_max)

    Q = np.array(Q).reshape((C,H,W))
    Q = Q.transpose((2,1,0))


    return Q
