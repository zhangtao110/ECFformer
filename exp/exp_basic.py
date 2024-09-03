import os
import torch
from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, Crossformer3, LSTNet, Linear,\
    Crossformer1_1, Crossformer1_2, Crossformer1_3, Crossformer1_4, Crossformer1_5, Crossformer1_6, Crossformer2_1, Crossformer2_2,\
    Crossformer2_3, Crossformer2_4, Crossformer2_5, Crossformer2_6, Crossformer2_7, Crossformer2_8, Crossformer2_9,Crossformer2_10,\
    Crossformer2_11, Crossformer2_12, Crossformer2_13, Crossformer2_14, Crossformer2_15, Crossformer3_1, Crossformer3Random, Crossformer3Random2, Crossformer3Random3, Crossformer32, DLinear_Transformer,\
    CrossformerZT, CrossformerN, Crossformer3Bias, Crossformer3Test, Crossformer3Random_linear, Crossformer3Random_max, \
    Crossformer3Random_min, Crossformer3Random_mean, CrossformerNRandom3, Crossformer4_1, Crossformer4_2, Crossformer4_3, iTransformer, TSMixer, NTformer, LSTM, Crossformer3_2, Crossformer3_3, Crossformer3_4,\
    Crossformer3_5, Crossformer3_Multi_scale1, Crossformer3_Multi_scale2, Crossformer3_Multi_scale_add, Crossformer3_time, Crossformer3_public, Crossformer3_feature, Crossformer3_attention, \
    Crossformer3_s5, Crossformer3_s6, Crossformer3_s10, Crossformer3_s15, Crossformer3_s30, Crossformer3_nomergy, Anomaly_ECformer

from models import  FlourierSplit
# from models.train_model import PCformer, PCformer01, PCformer02, PCformer03

from models import Crossformer3PatchNorm

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'LightTS': LightTS,
            'Reformer': Reformer,
            'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            'Pyraformer': Pyraformer,
            'MICN': MICN,
            'Crossformer': Crossformer,
            'FiLM': FiLM,
            'Crossformer3': Crossformer3,
            'Crossformer3_time': Crossformer3_time,
            'Crossformer3_feature': Crossformer3_feature,
            'Crossformer3_attention': Crossformer3_attention,
            'Crossformer3_public': Crossformer3_public,
            'Crossformer32': Crossformer32,
            'LSTNet':LSTNet,
            'Linear':Linear,
            'Crossformer1_1':Crossformer1_1,
            'Crossformer1_2':Crossformer1_2,
            'Crossformer1_3':Crossformer1_3,
            'Crossformer1_4':Crossformer1_4,
            'Crossformer1_5':Crossformer1_5,
            'Crossformer1_6':Crossformer1_6,
            'Crossformer2_1':Crossformer2_1,
            'Crossformer2_2':Crossformer2_2,
            'Crossformer2_3':Crossformer2_3,
            'Crossformer2_4':Crossformer2_4,
            'Crossformer2_5':Crossformer2_5,
            'Crossformer2_6':Crossformer2_6,
            'Crossformer2_7':Crossformer2_7,
            'Crossformer2_8':Crossformer2_8,
            'Crossformer2_9':Crossformer2_9,
            'Crossformer2_10':Crossformer2_10,
            'Crossformer2_11':Crossformer2_11,
            'Crossformer2_12':Crossformer2_12,
            'Crossformer2_13': Crossformer2_13,
            'Crossformer2_14': Crossformer2_14,
            'Crossformer2_15': Crossformer2_15,
            'Crossformer3_1':Crossformer3_1,
            'Crossformer3_2': Crossformer3_2,
            'Crossformer3_3': Crossformer3_3,
            'Crossformer3_4': Crossformer3_4,
            'Crossformer3_5': Crossformer3_5,
            'Crossformer3_Multi_scale1': Crossformer3_Multi_scale1,
            'Crossformer3_Multi_scale2': Crossformer3_Multi_scale2,
            'Crossformer3_Multi_scale_add': Crossformer3_Multi_scale_add,
            'Crossformer3_s5': Crossformer3_s5,
            'Crossformer3_s6': Crossformer3_s6,
            'Crossformer3_s10': Crossformer3_s10,
            'Crossformer3_s15': Crossformer3_s15,
            'Crossformer3_s30': Crossformer3_s30,
            'Crossformer3_nomergy': Crossformer3_nomergy,
            'Crossformer4_1': Crossformer4_1,
            'Crossformer4_2': Crossformer4_2,
            'Crossformer4_3': Crossformer4_3,
            'Crossformer3Random': Crossformer3Random,
            'Crossformer3Random2': Crossformer3Random2,
            'Crossformer3Random3': Crossformer3Random3,
            'DLinear_Transformer':DLinear_Transformer,
            'CrossformerN':CrossformerN,
            'CrossformerZT': CrossformerZT,
            'Crossformer3Bias':Crossformer3Bias,
            'FlourierSplit':FlourierSplit,
            'Crossformer3Test':Crossformer3Test,
            'Crossformer3Random_linear': Crossformer3Random_linear,
            'Crossformer3Random_max': Crossformer3Random_max,
            'Crossformer3Random_min': Crossformer3Random_min,
            'Crossformer3Random_mean': Crossformer3Random_mean,
            'CrossformerNRandom3': CrossformerNRandom3,
            'Crossformer3PatchNorm':Crossformer3PatchNorm,
            'iTransformer': iTransformer,
            'TSMixer': TSMixer,
            'NTformer': NTformer,
            'LSTM': LSTM,
            'Anomaly_ECformer': Anomaly_ECformer
            # 'PCformer': PCformer,
            # 'PCformer01': PCformer01,
            # 'PCformer02': PCformer02,
            # 'PCformer03': PCformer03
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        # for param in self.model.parameters():
        #     param.requires_grad = False
        # self.model.final.weight.requires_grad = True
        # self.model.final.bias.requires_grad = True
        # print(self.model)

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
