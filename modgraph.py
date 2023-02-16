import numpy as np
import pickle
import os

map_dir = '/remote-home/liuwenbo/pycproj/tsdata/maps/smd'
for root,dir_list,file_list in os.walk(map_dir):
    for file in file_list:
        print(os.path.join(root,file))
        f=open(os.path.join(root,file),'rb')
        data=pickle.load(f)
        f.close()
        print(type(data['G'].graph))
        f=open(os.path.join(root[:-3],'npmap',file[:-10]+'.npmap.pkl'),'wb')
        pickle.dump(data['G'].graph,f)
        # print('save file:',os.path.join(root,file[:-10]+'.npmap.pkl'))
        f.close()