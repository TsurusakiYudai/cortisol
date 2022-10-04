from gcmstools.filetypes import AiaFile
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import os
import glob
from natsort import natsorted
import numpy as np
from PIL import Image
import skimage
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import cv2
from numpy import zeros
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import RMSprop
import glob
import numpy as np
from PIL import Image, ImageChops
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.models import model_from_json
from scipy.optimize import curve_fit
import matplotlib.cm as cm
from multiprocessing import Process
import csv
import pprint
import scipy.optimize as optimize
import sklearn.metrics as metrics

#LinearRegression
LR_file = []
LR_score = []
files = glob.glob('/Volumes/USB DISK/cortisol/Raw data(.CDF) testing files/Cortisol/Image data with noise/Noise15%_Step0.25_0.15/110/*_110.png')
for file in files:
    file_new = np.ravel(np.array(Image.open(f'{file}').convert('L')))
    LR_file.append(file_new)
    LR_score.append(110)
files = glob.glob('/Volumes/USB DISK/cortisol/Raw data(.CDF) testing files/Cortisol/Image data with noise/Noise15%_Step0.25_0.15/247/*_247.png')
for file in files:
    file_new = np.ravel(np.array(Image.open(f'{file}').convert('L')))
    LR_file.append(file_new)
    LR_score.append(247)
files = glob.glob('/Volumes/USB DISK/cortisol/Raw data(.CDF) testing files/Cortisol/Image data with noise/Noise15%_Step0.25_0.15/335/*_335.png')
for file in files:
    file_new = np.ravel(np.array(Image.open(f'{file}').convert('L')))
    LR_file.append(file_new)
    LR_score.append(335)
files = glob.glob('/Volumes/USB DISK/cortisol/Raw data(.CDF) testing files/Cortisol/Image data with noise/Noise15%_Step0.25_0.15/365/*_365.png')
for file in files:
    file_new = np.ravel(np.array(Image.open(f'{file}').convert('L')))
    LR_file.append(file_new)
    LR_score.append(365)
files = glob.glob('/Volumes/USB DISK/cortisol/Raw data(.CDF) testing files/Cortisol/Image data with noise/Noise15%_Step0.25_0.15/394/*_394.png')
for file in files:
    file_new = np.ravel(np.array(Image.open(f'{file}').convert('L')))
    LR_file.append(file_new)
    LR_score.append(394)
files = glob.glob('/Volumes/USB DISK/cortisol/Raw data(.CDF) testing files/Cortisol/Image data with noise/Noise15%_Step0.25_0.15/446/*_446.png')
for file in files:
    file_new = np.ravel(np.array(Image.open(f'{file}').convert('L')))
    LR_file.append(file_new)
    LR_score.append(446)
files = glob.glob('/Volumes/USB DISK/cortisol/Raw data(.CDF) testing files/Cortisol/Image data with noise/Noise15%_Step0.25_0.15/631/*_631.png')
for file in files:
    file_new = np.ravel(np.array(Image.open(f'{file}').convert('L')))
    LR_file.append(file_new)
    LR_score.append(631)
files = glob.glob('/Volumes/USB DISK/cortisol/Raw data(.CDF) testing files/Cortisol/Image data with noise/Noise15%_Step0.25_0.15/818/*_818.png')
for file in files:
    file_new = np.ravel(np.array(Image.open(f'{file}').convert('L')))
    LR_file.append(file_new)
    LR_score.append(818)
files = glob.glob('/Volumes/USB DISK/cortisol/Raw data(.CDF) testing files/Cortisol/Image data with noise/Noise15%_Step0.25_0.15/1095/*_1095.png')
for file in files:
    file_new = np.ravel(np.array(Image.open(f'{file}').convert('L')))
    LR_file.append(file_new)
    LR_score.append(1095)
    
file_train, file_test, score_train, score_test = train_test_split(LR_file, LR_score, test_size=0.2, random_state=44)
model = LinearRegression()
model.fit(file_train, score_train)
print('train score : ', model.score(file_train, score_train))
print('test score : ', model.score(file_test, score_test))
importance = model.coef_
print(importance.size)
importance = importance.reshape(787, 1050)
plt.imshow(importance, cmap='magma', vmax=np.max(importance), vmin=0.00000)
ax=plt.gca()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.axis('off')
plt.savefig(f'/Volumes/USB DISK/cortisol/Raw data(.CDF) testing files/Cortisol/importance.png', bbox_inches='tight',pad_inches = 0, dpi=360)

def CountPeak(imgpath, threshold, numplot):
    img = np.array(Image.open(imgpath).convert('L'))
    peak_position = []
    peak_information_1 = []
    peak_information = []
    for x in range(img.shape[0]-2):
        for y in range(img.shape[1]-2):
            target_x = x+1
            target_y = y+1
            target_score = img[target_x][target_y]
            if target_score<threshold:
                continue
            if img[target_x-1][target_y-1]>target_score:
                continue
            if img[target_x][target_y-1]>target_score:
                continue
            if img[target_x+1][target_y-1]>target_score:
                continue
            if img[target_x-1][target_y]>target_score:
                continue
            if img[target_x+1][target_y]>target_score:
                continue
            if img[target_x-1][target_y+1]>target_score:
                continue
            if img[target_x][target_y+1]>target_score:
                continue
            if img[target_x+1][target_y+1]>target_score:
                continue
            peak_position.append((target_x, target_y))
            peak_information_1.append([target_x, target_y, target_score])
            peak_information.append([target_x, target_y, target_score])
            
    i = 0
    remove_nums =[]
    for position in peak_position:
        if position == peak_position[0]:
            i += 1
            continue
        else:
            for j in range(i):
                if (((position[0]-1, position[1]-1) == peak_position[j] and img[(position[0]-1, position[1]-1)] == img[position])
                    or ((position[0], position[1]-1) == peak_position[j] and img[(position[0], position[1]-1)] == img[position])
                    or ((position[0]+1, position[1]-1) == peak_position[j] and img[(position[0]+1, position[1]-1)] == img[position])
                    or ((position[0]-1, position[1]) == peak_position[j] and img[(position[0]-1, position[1])] == img[position])
                    or ((position[0]+1, position[1]-1) == peak_position[j] and img[(position[0]+1, position[1])] == img[position])
                    or ((position[0]-1, position[1]+1) == peak_position[j] and img[(position[0]-1, position[1]+1)] == img[position])
                    or ((position[0], position[1]+1) == peak_position[j] and img[(position[0], position[1]+1)] == img[position])
                    or ((position[0]+1, position[1]+1) == peak_position[j] and img[(position[0]+1, position[1]+1)] == img[position])):
                        remove_nums.append(i)
            i += 1
    remove_nums =  list(set(remove_nums))
    for num in remove_nums:
        peak_information.remove(peak_information_1[num])
    peak_information.sort(key=lambda x:x[2])
    peak_information.reverse()
    img = cv2.imread(imgpath)
    for k in range(len(peak_information)):  
        print(f'NO.{k+1} position:({peak_information[k][0]},{peak_information[k][1]}), intensity:{peak_information[k][2]}')
        cv2.circle(img,
                   center=(peak_information[k][1], peak_information[k][0]),
                   radius=4,
                   color=(0, 255, 0),
                   thickness=2)
        if (k%2 == 0 and numplot == True):
            cv2.putText(img,
                        text=f'{k+1}',
                        org=(peak_information[k][1]-5, peak_information[k][0]-10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(0, 255, 0),
                        thickness=1)
        if (k%2 ==1 and numplot == True):
            cv2.putText(img,
                        text=f'{k+1}',
                        org=(peak_information[k][1]-5, peak_information[k][0]+18),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(0, 255, 0),
                        thickness=1)
    cv2.putText(img,
                text=f'Peak:{len(peak_information)}',
                org=(10,100),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=3.0,
                color=(0, 255, 0),
                thickness=2)
    Nstr = len(imgpath)
    filename = imgpath[0:Nstr-4]
    cv2.imwrite(f'{filename}_peak_{threshold}.png', img)

CountPeak('/Volumes/USB DISK/cortisol/Raw data(.CDF) testing files/Cortisol/importance.png',  50, False)
CountPeak('/Volumes/USB DISK/cortisol/Raw data(.CDF) testing files/Cortisol/Image data with noise/Noise15%_Step0.25_0.15/110/1_110_0.png', 100, True)

#Detect Peak range
x_start = []
y_start = []
x_goal = []
y_goal = []

def onMouse(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start.append(x), y_start.append(y)
    if event == cv2.EVENT_LBUTTONUP:
        x_goal.append(x), y_goal.append(y)
        print(x_start, y_start, x_goal, y_goal)

img = cv2.imread('/Volumes/USB DISK/cortisol/Raw data(.CDF) testing files/Cortisol/importance.png')
cv2.imshow('sample', img)
cv2.setMouseCallback('sample', onMouse)
while True:
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cv2.waitKey(1)
cv2.destroyAllWindows()
cv2.waitKey(1)


x_start = [1012]
y_start = [115]
x_goal = [1047]
y_goal = [142]
x_start = np.array(x_start)
y_start = np.array(y_start)
x_goal = np.array(x_goal)
y_goal = np.array(y_goal)
for x_start_0, y_start_0, x_goal_1, y_goal_1, i in zip(x_start, y_start, x_goal, y_goal, range(x_start.size)):
    image = np.array(Image.open('/Volumes/USB DISK/cortisol/Raw data(.CDF) testing files/Cortisol/Image data with noise/Noise15%_Step0.25_0.15/1095/1_1095.png'))
    image = image[y_start_0:y_goal_1, x_start_0:x_goal_1]
    print(image.shape)
    plt.imshow(image)
    ax=plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.axis('off')
    plt.savefig(f'/Volumes/USB DISK/cortisol/Raw data(.CDF) testing files/Cortisol/Image data with noise/Noise15%_Step0.25_0.15/110/1_1095_{i}.png', bbox_inches='tight',pad_inches = 0)

[1012] [115] [1047] [142]

#Detect peak number    
def peakplot(file_path):
    img = np.array(Image.open(file_path).convert('L'))
    peak_axis_y = np.amax(img, axis=0)
    peak_axis_y = peak_axis_y.tolist()
    peak_axis_x = list(range(len(peak_axis_y)))
    plt.plot(peak_axis_x, peak_axis_y)

def func(x, *params):
    num_func = int(len(params)/3)
    y_list = []
    for i in range(num_func):
        y = np.zeros_like(x)
        param_range = list(range(3*i, 3*(i+1),1))
        amp = params[int(param_range[0])]
        ctr = params[int(param_range[1])]
        wid = params[int(param_range[2])]
        y = y + amp * np.exp( -((x - ctr)/wid)**2)
        y_list.append(y)
    y_sum = np.zeros_like(x)
    for i in y_list:
        y_sum = y_sum + i
    y_sum = y_sum + params[-1]
    return y_sum

def fit_plot(x, *params):
    num_func = int(len(params)/3)
    y_list = []
    for i in range(num_func):
        y = np.zeros_like(x)
        param_range = list(range(3*i,3*(i+1),1))
        amp = params[int(param_range[0])]
        ctr = params[int(param_range[1])]
        wid = params[int(param_range[2])]
        y = y + amp * np.exp( -((x - ctr)/wid)**2) + params[-1]
        y_list.append(y)
    return y_list

def peak_number(file_path):
    img = np.array(Image.open(file_path).convert('L'))
    peak_axis_y = np.amax(img, axis=0)
    peak_axis_y = peak_axis_y.tolist()
    peak_axis_x = list(range(len(peak_axis_y)))
    
    guess = []
    guess_average = [70, 100, 50]#amp,pos,wid
    guess.append([60, 150, 50])
    guess.append([20, 200, 50])
    #guess.append([40, 10, 25])
    #guess.append([20, 115, 10])
    #guess.append(guess_average)
    #guess.append(guess_average)
    #guess.append(guess_average)
    background = 70

    guess_total = []
    for i in guess:
        guess_total.extend(i)
    guess_total.append(background)
    x = np.linspace(0, len(peak_axis_x), 10000)
    popt, pcov = curve_fit(func, peak_axis_x, peak_axis_y, p0=guess_total)
    fit = func(x, *popt)
    plt.scatter(peak_axis_x, peak_axis_y, s=20)
    plt.plot(x, fit , ls='-', c='black', lw=1)

    y_list = fit_plot(peak_axis_x, *popt)
    baseline = np.zeros_like(peak_axis_x) + popt[-1]
    for n,i in enumerate(y_list):
        plt.fill_between(peak_axis_x, i, baseline, facecolor=cm.rainbow(n/len(y_list)), alpha=0.6)

peakplot('/Volumes/USB DISK/cortisol/Raw data(.CDF) testing files/Cortisol/Image data with noise/Noise15%_Step0.25_0.15/110/1_110_0.png')
peak_number('/Volumes/USB DISK/cortisol/Raw data(.CDF) testing files/Cortisol/Image data with noise/Noise15%_Step0.25_0.15/110/1_110_0.png')
peakplot('/Volumes/USB DISK/cortisol/Raw data(.CDF) testing files/Cortisol/Image data with noise/Noise15%_Step0.25_0.15/110/1_1095_0.png')
peak_number('/Volumes/USB DISK/cortisol/Raw data(.CDF) testing files/Cortisol/Image data with noise/Noise15%_Step0.25_0.15/110/1_110_0.png')

#Peak intensity
def Peakint_to_csv(pos_x, pos_y):
    scores = [110, 247, 335, 365, 394, 446, 631, 818, 1095]
    i = 0
    peakint = []
    for i in range(9):
        score = scores[i]
        files = glob.glob(f'/Volumes/USB DISK/cortisol/Raw data(.CDF) testing files/Cortisol/Image data with noise/Noise15%_Step0.25_0.15/{score}/*_{score}.png')
        for file in files:
            file_array = np.array(Image.open(f'{file}').convert('L'))
            peak = file_array[pos_x][pos_y]
            peakint.append(peak)
    peakint = np.array(peakint)
    peakint = peakint.reshape(9,180)
    peakint = peakint.T
    np.savetxt(f'/Volumes/USB DISK/cortisol/Raw data(.CDF) testing files/Cortisol/outcsv_({pos_x},{pos_y}).csv', peakint, delimiter = ',') 

Peakint_to_csv(436, 139)
Peakint_to_csv(122, 1023)
Peakint_to_csv(128, 625)
Peakint_to_csv(122, 769)
Peakint_to_csv(121,784)
Peakint_to_csv(126,1029)

img = np.array(Image.open('/Volumes/USB DISK/cortisol/Raw data(.CDF) testing files/Cortisol/Image data with noise/Noise15%_Step0.25_0.15/110/1_110.png').convert('L'))
img.shape
for i in range(787):
    for j in range(1050):
        if img[i][j] == 132 and i >= 115 and i <=130:
            print(f'({i},{j})')
            continue

def func(x, a, b):
    return a + b*x

def Peak_R2(x, y):
    scores = [110, 247, 335, 365, 394, 446, 631, 818, 1095]
    peak_ave = []
    peak_pred = []
    for i in range(9):
        score = scores[i]
        files = glob.glob(f'/Volumes/USB DISK/cortisol/Raw data(.CDF) testing files/Cortisol/Image data with noise/Noise15%_Step0.25_0.15/{score}/*_{score}.png')
        peak_sum = 0
        i = 0
        for file in files:
            file_array = np.array(Image.open(f'{file}').convert('L'))
            peak_sum += file_array[x][y]
            i += 1
        peak_ave.append(peak_sum/i)
    popt, _ = optimize.curve_fit(func, scores, peak_ave)
    for i in range(9):
        score = scores[i]
        peak_pred.append(popt[0]+popt[1]*score)
    peak_ave = np.array(peak_ave)
    peak_pred = np.array(peak_pred)
    rss = np.sum((peak_ave-peak_pred)**2)
    tss= np.sum((peak_ave-np.mean(peak_ave))**2)
    r2 = 1-(rss/tss)
    return r2, popt[1]

def Max_r2_pos(imgpath, threshold):
    img = np.array(Image.open(imgpath).convert('L'))
    peak_position = []
    peak_information_1 = []
    peak_information = []
    for x in range(img.shape[0]-2):
        for y in range(img.shape[1]-2):
            target_x = x+1
            target_y = y+1
            target_score = img[target_x][target_y]
            if target_score<threshold:
                continue
            if img[target_x-1][target_y-1]>target_score:
                continue
            if img[target_x][target_y-1]>target_score:
                continue
            if img[target_x+1][target_y-1]>target_score:
                continue
            if img[target_x-1][target_y]>target_score:
                continue
            if img[target_x+1][target_y]>target_score:
                continue
            if img[target_x-1][target_y+1]>target_score:
                continue
            if img[target_x][target_y+1]>target_score:
                continue
            if img[target_x+1][target_y+1]>target_score:
                continue
            peak_position.append((target_x, target_y))
            peak_information_1.append([target_x, target_y, target_score])
            peak_information.append([target_x, target_y, target_score])
            
    i = 0
    remove_nums =[]
    for position in peak_position:
        if position == peak_position[0]:
            i += 1
            continue
        else:
            for j in range(i):
                if (((position[0]-1, position[1]-1) == peak_position[j] and img[(position[0]-1, position[1]-1)] == img[position])
                    or ((position[0], position[1]-1) == peak_position[j] and img[(position[0], position[1]-1)] == img[position])
                    or ((position[0]+1, position[1]-1) == peak_position[j] and img[(position[0]+1, position[1]-1)] == img[position])
                    or ((position[0]-1, position[1]) == peak_position[j] and img[(position[0]-1, position[1])] == img[position])
                    or ((position[0]+1, position[1]-1) == peak_position[j] and img[(position[0]+1, position[1])] == img[position])
                    or ((position[0]-1, position[1]+1) == peak_position[j] and img[(position[0]-1, position[1]+1)] == img[position])
                    or ((position[0], position[1]+1) == peak_position[j] and img[(position[0], position[1]+1)] == img[position])
                    or ((position[0]+1, position[1]+1) == peak_position[j] and img[(position[0]+1, position[1]+1)] == img[position])):
                        remove_nums.append(i)
            i += 1
    remove_nums =  list(set(remove_nums))
    for num in remove_nums:
        peak_information.remove(peak_information_1[num])
    peak_information.sort(key=lambda x:x[2])
    peak_information.reverse()
    
    Max_r2 = 0
    for k in range(len(peak_information)): 
        r2 = Peak_R2(peak_information[k][0], peak_information[k][1])[0]
        a = Peak_R2(peak_information[k][0], peak_information[k][1])[1]
        if r2 >= Max_r2 and a>0:
            Max_r2 = r2
            Max_r2_pos = (peak_information[k][0], peak_information[k][1])
            continue
    print(f'Max_r2: {Max_r2_pos} r2={Max_r2}')

Max_r2_pos('/Volumes/USB DISK/cortisol/Raw data(.CDF) testing files/Cortisol/importance.png', 150)





















