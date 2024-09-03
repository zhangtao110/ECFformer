import os

names = ['../pumping/cc8-244_5.csv',
         '../pumping/ps12.csv',
         '../pumping/ps104.csv',
         '../pumping/ps_7_282_58.csv',
         '../pumping/pt104.csv',
         '../pumping/pt107.csv',
         '../pumping/xt1-177.8.csv',
         '../pumping/z201H62-1-139.7.csv',
         '../pumping/daan1H1_3_undelete_flow.csv',
         '../pumping/tc_244_5.csv',
         ]

for i in names:
    t = i.split('/')[-1].split('.csv')[0]
    print(t)
    if os.path.exists(t):
        pass
    else:
        os.mkdir(t)