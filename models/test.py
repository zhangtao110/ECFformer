import torch



mols = ['Crossformer','FEDformer','Autoformer','Crossformer3Test','InformerTest']


ins = 96
outs = 96
fe = 7


class config():
    def __init__(self):
        self.task_name = 'long_term_forecast'
        self.seq_len = ins
        self.label_len = 48
        self.pred_len = outs
        self.e_layers = 2
        self.d_layers = 1
        self.factor = 3
        self.enc_in =fe
        self.dec_in = fe
        self.c_out = fe
        self.moving_avg=25
        self.d_model = 512
        self.n_heads = 8
        self.d_ff = 2048
        self.dropout = 0.5
        self.embed = 'timeF'
        self.freq = 'h'
        self.output_attention = True
        self.activation = 'gelu'
        self.distil = False
        self.seg_len = ins//2

config = config()
from thop import profile

for i in mols:
    if i == 'Crossformer':
        from CrossformerTestM import Model
    elif i == 'FEDformer':
        from FEDformerTest import Model
    elif i == 'Autoformer':
        from AutoformerTest import Model
    elif i =='Crossformer3Test':
        from Crossformer3TestM import Model
    elif i=='InformerTest':
        from InformerTest import Model

    model = Model(config).cpu()
    a = torch.randn(32, ins, fe)
    flops, params = profile(model, inputs=(a,))
    from thop import clever_format
    macs, params = clever_format([flops, params], "%.3f")
    print(i+'------------------')
    print(macs)
    print(params)