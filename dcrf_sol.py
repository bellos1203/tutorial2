import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils

def crf(img, prob, iter_max):
    img = np.array(img)
    ## Exercise - 1 
    # prob.shape : H, W, C
    prob_tr = prob.transpose((2,0,1))  # H W C to C H W
    img = np.array(img)
    C, H, W = prob_tr.shape

    U = utils.unary_from_softmax(prob_tr)
    U = np.ascontiguousarray(U)

    image = np.ascontiguousarray(img)

    d = dcrf.DenseCRF2D(W, H, C)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=1, compat=3)
    d.addPairwiseBilateral(sxy=67, srgb=3, rgbim=image, compat=4)

    Q = d.inference(iter_max)
    Q = np.array(Q).reshape((C,H,W))
    Q = Q.transpose((2,1,0))


    return Q
