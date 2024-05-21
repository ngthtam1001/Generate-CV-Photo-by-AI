import datetime
import numpy as np
import os
import os.path as osp
import glob
import cv2
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import matplotlib.pyplot as plt
from shape_determine import determine_face_shape as dfs
from shape_detect import calculate



assert float('.'.join(insightface.__version__.split('.')[:2]))>=float('0.7')

app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

swapper = insightface.model_zoo.get_model('inswapper_128.onnx',
                                download=False,
                                download_zip=False)

def swap_n_show(img1_fn, img2_fn, app, swapper, plot_after=True):
    
    img1 = cv2.imread(img1_fn)
    img2 = cv2.imread(img2_fn)
    
    face1 = app.get(img1)[0]
    face2 = app.get(img2)[0]
    
    img1_ = img1.copy()
    img2_ = img2.copy()
    if plot_after:
        img1_ = swapper.get(img1_, face1, face2, paste_back=True)
        plt.imshow(img1_[:,:,::-1])
        plt.axis('off') 
        plt.show()

    return img1_
img = 'face1.jpg' #Path to your image
gender = str(input())


#Process for male
if gender == "male" and dfs(calculate(img)[0],calculate(img)[1],calculate(img)[2], calculate(img)[3]) == "Round":
    _ = swap_n_show('round_man.webp', img, app, swapper)
    
if gender == "male" and dfs(calculate(img)[0],calculate(img)[1],calculate(img)[2], calculate(img)[3]) == 'Rectangular':
    _ = swap_n_show('rectangular_man.webp', img, app, swapper)
    
if gender == "male" and dfs(calculate(img)[0],calculate(img)[1],calculate(img)[2], calculate(img)[3]) == 'Oval':
    _ = swap_n_show('oval_man.webp', img, app, swapper)
    
if gender == "male" and dfs(calculate(img)[0],calculate(img)[1],calculate(img)[2], calculate(img)[3]) == 'Oblong':
    _ = swap_n_show('oblong_man.webp', img, app, swapper)
    
if gender == "male" and dfs(calculate(img)[0],calculate(img)[1],calculate(img)[2], calculate(img)[3]) == 'Square':
    _ = swap_n_show('square_man.webp', img, app, swapper)
    
if gender == "male" and dfs(calculate(img)[0],calculate(img)[1],calculate(img)[2], calculate(img)[3]) == 'Diamond':
    _ = swap_n_show('diamond_man.webp', img, app, swapper)
    
if gender == "male" and dfs(calculate(img)[0],calculate(img)[1],calculate(img)[2], calculate(img)[3]) == 'Heart':
    _ = swap_n_show('heart_man.webp', img, app, swapper)

#Process for female
if gender == "female" and dfs(calculate(img)[0],calculate(img)[1],calculate(img)[2], calculate(img)[3]) == "Round":
    _ = swap_n_show('round_woman.png', img, app, swapper)
    
if gender == "female" and dfs(calculate(img)[0],calculate(img)[1],calculate(img)[2], calculate(img)[3]) == 'Rectangular':
    _ = swap_n_show('rectangular_woman.png', img, app, swapper)
    
if gender == "female" and dfs(calculate(img)[0],calculate(img)[1],calculate(img)[2], calculate(img)[3]) == 'Oval':
    _ = swap_n_show('oval_woman.png', img, app, swapper)
    
if gender == "female" and dfs(calculate(img)[0],calculate(img)[1],calculate(img)[2], calculate(img)[3]) == 'Oblong':
    _ = swap_n_show('oblong_woman.jpg', img, app, swapper)
    
if gender == "female" and dfs(calculate(img)[0],calculate(img)[1],calculate(img)[2], calculate(img)[3]) == 'Square':
    _ = swap_n_show('square_woman.png', img, app, swapper)
    
if gender == "female" and dfs(calculate(img)[0],calculate(img)[1],calculate(img)[2], calculate(img)[3]) == 'Diamond':
    _ = swap_n_show('diamond_woman.png', img, app, swapper)
    
if gender == "female" and dfs(calculate(img)[0],calculate(img)[1],calculate(img)[2], calculate(img)[3]) == 'Heart':
    _ = swap_n_show('heart_woman.png', img, app, swapper)
