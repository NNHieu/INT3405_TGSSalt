from .unet import UNet
from .planx import Res34Unetv3, Res34Unetv4, Res34Unetv5
from .resunet import Res34Unet

def get_model(model):
    if model == 'baseline':
        return UNet(1, 1, 16, 4)
    if model == 'baseline64':
        return UNet(1, 1, 64, 4)
    elif model == 'res34':
        return Res34Unet()
    elif model == 'phalanx_res34v3':
        return Res34Unetv3()

    elif model == 'phalanx_res34v4':
        return Res34Unetv4()
    
    elif model == 'phalanx_res34v5':
        return Res34Unetv5()
    
    else:
        print('Error: ', model, ' is not defined.')
        return