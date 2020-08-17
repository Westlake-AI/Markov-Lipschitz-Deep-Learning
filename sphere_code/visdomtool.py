
import visdom
import numpy as np
from PIL import Image
import cv2

viz = visdom.Visdom(port=5501, env='MAE')



def ShowImg(path, name):
    img = cv2.imread(path)
    # img.show()
    
    # print(img)
    # print(np.array(img))
    # print(img.shape)
    viz.images(
        img.transpose(2, 0, 1)[::-1,...],
        win=name,
        opts=dict(
            title=name,
        )
        )