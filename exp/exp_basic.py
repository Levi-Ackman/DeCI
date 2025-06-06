import os
import torch
from models import iTransformer,SOFTS,TimesNet,\
            PatchTST,TSMixer,TimeMixer,ModernTCN,Leddam,DeCI,Medformer,BrainOOD,BrainNetTF

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model_dict = {
            'DeCI':DeCI,
            'Leddam':Leddam,
            'Medformer':Medformer,
            'iTransformer':iTransformer,
            'PatchTST':PatchTST,
            'TSMixer':TSMixer,
            'SOFTS':SOFTS,
            'TimeMixer':TimeMixer,
            'ModernTCN':ModernTCN,
            'TimesNet':TimesNet,
            'BrainOOD':BrainOOD,
            'BrainNetTF':BrainNetTF,
        }
        self.model, self.initial_model= self._build_model()
    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
            
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass