import os
from os import listdir
from PIL import Image

count = 0
parent = 'web256'
for dirs in os.listdir(parent):
    for path in os.listdir(os.path.join(parent,dirs)):
        path = os.path.join(parent,dirs,path)
        try:
            img=Image.open(path)
            img.verify()
        except(IOError,SyntaxError)as e:
            print('Bad file  :  '+path)
            count=count+1
            print(count)

