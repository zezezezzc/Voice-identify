# -*- coding: utf-8 -*-
"""
@function:语音说话人识别主程序。接收用户的语音wav文件，调用语音降噪、提取语音特征、建立GMM模型、识别说话人
"""
from recognize import Recognize     #从降噪脚本中引入语音识别类(Recognize)
from scipy.io import wavfile
from sklearn.mixture import GaussianMixture
from librosa import resample
from transform import *     #导入MFCC
import util
import numpy as np
import argparse     # argparse是python标准库里面用来处理命令行参数的库
import os


def is_wav_file(filename):
    '''
    Check if the file is '.wav' file or not
    :param filename: file name to check out
    :return: True if the file is '.wav' file
    '''
    return any(filename.endswith(extension) for extension in [".wav"])

def load_wav_file(filename):
    '''
    Load wav file using filename
    :param filename: file name to open
    :return: sample rate and data of the wav file
    '''
    sample_rate, data = wavfile.read(filename)
    return sample_rate, data

def load_data(filepath):
    '''
    加载wav文件并对其进行预处理以更正形状和类型
    param:
    filepath: file path of wav file
    return: mfcc data
    '''
    sample_rate, data = load_wav_file(filepath)
    if data.dtype != np.floating:
        data = data.astype(np.floating)

    if sample_rate != 16000:
        data = resample(data, sample_rate, 16000)
    
    # scaling needed -> not flatten yet
    data = mfcc(data, sample_rate, normalization=1, logscale=1, delta=0)
    return data

def paste_slices(tup):
    pos, w, max_w = tup
    dest_min = max(pos, 0)
    dest_max = min(pos+w, max_w)
    src_min = -min(pos,0)
    src_max = max_w - max(pos+w, max_w)
    src_max = src_max if src_max != 0 else None
    return slice(dest_min, dest_max), slice(src_min, src_max)

def paste(dest, src, loc):
    # map方法，创建一个迭代器，该迭代器使用每个可迭代对象的参数来计算函数。当最短的迭代次数用尽时停止
    # zip方法 返回一个zip对象，其.next（）方法返回一个元组，其中第i个元素来自第i个可迭代参数。 .next（）方法继续，直到参数序列中最短的
    # 可迭代耗尽为止，然后引发StopIteration。
    loc_zip = zip(loc, src.shape, dest.shape)
    dest_slices, src_slices = zip(*map(paste_slices,loc_zip))
    dest[dest_slices] += src[src_slices]

if __name__ == '__main__':
    rec_obj = Recognize('语音预处理及降噪')
    # 接收语音
    print("请输入您的文件路径：")
    filepath = ''
    os.listdir(filepath)
    # 接收用户的文件路径输入
    print("您的指令已接收！请稍等片刻~:")

    #预处理
    outpath1='./cache/out.wav'
    rec_obj.preprocess(filepath,outpath1)

    # 分片 1-2s
    sspath='./cache/out.wav'
    rec_obj.slice(sspath)
    spath='./cache/output[0]'

    # 进行降噪处理，在这个函数在会识别用户输入的路径是否存在，如果不存在则请用户重新输入
    rec_obj.denoise(spath)
    print("成功降噪")

    # 进行说话人识别
    parser = argparse.ArgumentParser(description="Acoustic Scene Classification Using Gaussian Mixture Model")  #创建一个解析对象
    parser.add_argument('--input_file', type=str, default="./example/wangzhe _neutral_206.wav", help="Input file path")# 向该对象中添加你要关注的命令行参数和选项
    parser.add_argument('--model', type=str, default="./model/gmm.json", help="Model file(json) to load")

    opt = parser.parse_args()       #进行解析

    # Labels of your Machine
    labels = ['liuchanhg', 'wangzhe', 'zhaoquanyin', 'ZhaoZuoxiang']

    filepath = opt.input_file
    estimator = util.load_model(opt.model)  #导入训练好的高斯混合模型

    n_classes, input_shape = estimator.means_.shape
    print(n_classes)
    print(input_shape)
    # 13 is the number of mfcc feature
    input = np.zeros([input_shape//13, 13])
    wav = load_data(filepath)

    paste(input, wav, (0,0))            #调用paste
    input = np.asarray([input.flatten()])

    output = estimator.predict(input)

    for item in enumerate(labels):
        if item[0] == output:
            print(item[1])