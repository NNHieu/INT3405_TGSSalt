from .unet import UNet
from .planx import Res34Unetv3, Res34Unetv4, Res34Unetv5
from .resunet import Res34Unet
from .effunet import EffUnet

def get_model(model):
    if model == 'baseline':
        return UNet(1, 1, 16, 4)
    elif model == 'baseline_mish':
        return UNet(1, 1, 16, 4, activation='mish')
    elif model == 'baseline64':
        return UNet(1, 1, 64, 4)
    elif model == 'baseline_3c':
        return UNet(3, 1, 16, 4)
    elif model == 'res34':
        return Res34Unet()
    elif model == 'phalanx_res34v3':
        return Res34Unetv3()

    elif model == 'phalanx_res34v4':
        return Res34Unetv4()
    
    elif model == 'phalanx_res34v5':
        return Res34Unetv5()
    
    elif model == 'effunet_b0':
        return EffUnet('tf_efficientnet_b0_ns', True)
    elif model == 'effunet_b3':
        return EffUnet('tf_efficientnet_b3_ns', True)
    elif model == 'effunet_b4':
        return EffUnet('tf_efficientnet_b4_ns', True)
    elif model == 'effunet_b5':
        return EffUnet('tf_efficientnet_b5_ns', True)
    
    else:
        print('Error: ', model, ' is not defined.')
        return