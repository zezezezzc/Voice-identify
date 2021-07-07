from __future__ import absolute_import
from librosa import resample
import numpy as np
from scipy.io import wavfile
from os import listdir
from os.path import join
from sklearn.model_selection import StratifiedKFold
import random
from transform import *
from tqdm import tqdm

def is_wav_file(filename):
    return any(filename.endswith(extension) for extension in [".wav"])

def load_wav_file(filename):
    """Loads an audio file and returns a float PCN-encoded array of samples
    加载音频文件并返回浮点PCN编码的样本数组"""
    sample_rate, data = wavfile.read(filename)    #.wavfile.read(somefile)来读取.wav音频文件。它会返回一个元组,第一项为音频的采样率,第二项为音频数据的numpy数组
    numpy.save("./cache/data.npy",data)
    return sample_rate, data

def load_data(wave_dir):        #从文件夹中读取文件  wave_dir = opt.data_dir
    print(wave_dir)             #结果： ./data
    labels = listdir(wave_dir)  #列出给定目录的文件名--人名作为标签

    ulabel = -1
    print(labels)     #label应该是个文件夹：['calm', 'noisy', 'traffic', 'tv', '暂存']   #将文件名显示  ['201.npy', 'beijing.npy', 'calm.npy', 'example1.npy', 'example2.npy']
    dataset = {'data':[], 'target':[]}
    '''dataset的值为
    {'data': [array([ 13.24215923, -12.3754675 , -26.89302811, ...,   0.        ,
         0.        ,   0.        ]), array([  2.30414637, -36.17490024, -32.47214503, ...,  34.10792057,
        55.08576757,  80.92231544])], 'target': [0, 0]}
    '''
    duration = []
    count = 0
    
    for label in labels:    # 由于内存问题，限制了样本数量
        #print(label)  #可以循环，依次出现201...
        data_dir = join(wave_dir, label) # join() 方法用于将序列中的元素以指定的字符连接生成一个新的字符串
        print(data_dir)    #结果为 ./data\calm  ./data\201.npy  ./data\beijing.npy  ./data\example1.npy  ./data\example2.npy
       
        #后面的语句提前进行测试
        #wav_filenames = listdir(data_dir)  #  listdir是列出文件夹里的内容
        #print(wav_filenames)  #['calm.npy','example1.npy']

        if label == 'yangguang':
            ulabel = 0
           # print("q") #可以循环
        elif label == 'chaiwenjiao':
            ulabel = 1
            #print("w") #不能进入
        elif label == 'bumaliya':
            ulabel = 2
        # elif label == 'tv':
        #     ulabel = 3
        else:
            pass

        if ulabel == -1:
            print("No Such Label In LABELS ERROR")
            continue

        wav_filenames = listdir(data_dir)
        print(wav_filenames)
        for filename in tqdm(wav_filenames):   #tqdm：用来显示进度条     这里wav_filenames应该是wav文件？？
            #限制最大数量
            '''
            if count == 6000:
                count = 0
                break
            '''
            filepath = join(data_dir, filename)
            #print(filepath)         #./data\calm\calm.npy 表示内嵌循环没有走完
            sample_rate, data = load_wav_file(filepath)       #sample_rate 为音频的采样率
            print(sample_rate)
            # SPECIFIED SAMPLING RATE (ALL IDENTICAL)
            if sample_rate == 16000:
                duration.append(data.shape[0])
                dataset['data'].append(data)
                dataset['target'].append(ulabel)
                count += 1
            
    #min_duration = min(duration)
    max_duration = max(duration)

    for i in tqdm(range (0, len(dataset['data']))):
        tmp = np.zeros(max_duration)
        tmp[:dataset['data'][i].shape[0]] = dataset['data'][i]
        dataset['data'][i] = tmp
        #dataset['data'][i] = dataset['data'][i][:min_duration]
        dataset['data'][i] = mfcc(dataset['data'][i], 16000, normalization=1, logscale=1, delta=0)
        dataset['data'][i] = dataset['data'][i].flatten()
    return dataset

def train_and_test(dataset):
    ## 选择训练和测试数据
    #分层K折 交叉验证，对于训练集，它做了5等分，注意是按顺序分的,最后一份做验证
    skf = StratifiedKFold(n_splits=2)   #原来n_splits=5，生成索引以将数据分为训练和测试集。
    train_index, test_index = next(iter(skf.split(dataset['data'], dataset['target'])))
    
    train_set = list(zip(np.asarray(dataset['data'])[train_index], np.asarray(dataset['target'])[list(train_index)]))
    random.shuffle(train_set)
    X_train = np.asarray([i for i, j in train_set])
    Y_train = np.asarray([j for i, j in train_set])
    X_test = np.asarray(dataset['data'])[test_index]
    Y_test = np.asarray(dataset['target'])[test_index]
    return X_train, Y_train, X_test, Y_test

def load_data2(filepath):
    '''
    Load wav file and pre-process it to correct shape and type
    param filepath: file path of wav file
    return: mfcc data
    加载wav文件并对其进行预处理以更正形状和类型
    param filepath：WAV文件的文件路径
    返回：MFCC数据
    '''
    sample_rate, data = load_wav_file(filepath)
    if data.dtype != np.floating:
        data = data.astype(np.floating)

    if sample_rate != 16000:
        data = resample(data, sample_rate, 16000)

    # scaling needed -> not flatten yet
    data = mfcc(data, sample_rate, normalization=1, logscale=1, delta=0)
    return data
    #data = load_data("./example/beijing.wav")
