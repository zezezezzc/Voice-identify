# -*- coding: utf-8 -*-
"""
Created on 2019-12-7-21.00
"""

from pydub import AudioSegment
from ffmpy import FFmpeg
from os import path

# .wav的音频和本脚本应放在同一目录下
import numpy as np
import os
import wave
import nextpow2
import math
import librosa

# 语音识别类
class Recognize:

    def __init__(self, name):
        self.name = name

    def preprocess(self, file_name, out_path1):
        song = AudioSegment.from_mp3(file_name)
        song.export(out_path1, format="wav")

        # newFilename ='./cache/out_ 16k.wav'     # 局部变量
        # y,sr = librosa.load(out_path1, sr=None)
        # y_16 = librosa.resample(y,sr,16000)
        # librosa.output.write_wav(newFilename, y_16, 16000)
        # ff = FFmpeg(inputs={out_path1: None},outputs={out_path2: '-i out_path1 -acodec pcm_s16le -y out_path2'})
        print('success')
        # ff.run()
        '''
        #将格式不正确的wav文件转码为ffmpeg格式的wav文件：
        ffmpeg -i "sourceFile" -y "targetFile"

        #将mp3文件转码为ffmpeg格式的wav文件（编码格式为16PCM、小端模式）：
        ffmpeg -i "sourceFile" -acodec pcm_s16le -y "targetFile"

        # 音频切割：
        ffmpeg -i "sourceFile" -ss startTime -to endTime -y "targetFile"（按起点和终点切割）
        ffpmeg -i "sourceFile" -ss startTime -t duration -y "targetFile"（按起点和持续时间切割）
        ''' 
        # 文件处理为Wav，采样率16k的文件，返回文件名
        
    def slice(self,spath):
        """
		@description:音频切分
		@input_file:input.wav
		@output_file:output[i].wav
		"""
        path = spath
        if os.listdir(path):
            for i in os.listdir(path):
                path_file = os.path.join(path, i)
                os.remove(path_file)
        sound_new = AudioSegment.from_file(path, format="wav")  # 路径错误
        # 打开目标音频文件
        len_sound = len(sound_new)
        # 计算音频时长
        for i in range(0, 1000):
            filename = "./cache/output" + str(i) + ".wav"
            # 命名输出文件名
            n = i + 1
            sound_new[4286 * i:4286 * n].export(filename, format="wav")
            # 每2秒一切分，并按顺序输出
            if len_sound < 4286:
                break
            # 当音频文件小于4秒时停止切分
            len_sound = len_sound - 4286

    def berouti1(self,SNR):
        # @description: 幅度谱函数
        if -5.0 <= SNR <= 20.0:
            a = 3 - SNR * 2 / 20
        else:
            if SNR < -5.0:
                a = 4
            if SNR > 20:
                a = 1
        return a

    def berouti(self,SNR):
        # @description:功率谱函数
        if -5.0 <= SNR <= 20.0:
            a = 4 - SNR * 3 / 20
        else:
            if SNR < -5.0:
                a = 5
            if SNR > 20:
                a = 1
        return a

    def find_index(self,x_list):
        index_list = []
        for i in range(len(x_list)):
            if x_list[i] < 0:
                index_list.append(i)
        return index_list

    
    def denoise(self, audio):
        """
        @description:音频降噪
        @input_file:语音路径
        @output_file:input.wav
        @语音文件路径差错处理体系：如用户输入的文件地址无效，则告知用户并重新接收用户输入
        """
        try:
            fpath = audio
            wave.open(fpath)
        except Exception as e:
            print(e)
            print("输入的文件地址不存在,或文件格式出错 请重新输入~")
            fpath = input()
            while True:
                if path.exists(fpath):
                    print("文件地址已收到，请稍候片刻...")
                    break
                else:
                    print("出错了！请重新输入~注意大小写哦！")
                    fpath = input()
            # 若文件路径不存在，则一直提示用户，直到正确
        finally:
            # 读取格式信息
            # (nchannels, sampwidth, framerate, nframes, comptype, compname)
            f = wave.open(fpath)
            params = f.getparams()
            nchannels, sampwidth, framerate, nframes = params[:4]
            fs = framerate
            # 读取波形数据
            str_data = f.readframes(nframes)
            f.close()
            # 将波形数据转换为数组
            x = np.fromstring(str_data, dtype=np.short)
            # 计算参数
            len_ = 20 * fs // 1000
            PERC = 50
            len1 = len_ * PERC // 100
            len2 = len_ - len1
            # 设置默认参数
            Thres = 3
            Expnt = 2.0
            beta = 0.002
            G = 0.9
            # 初始化汉明窗
            win = np.hamming(len_)
            # 重叠的归一化增益+ 增加50％重叠
            winGain = len2 / sum(win)

            # 噪声幅度计算-假设前5帧是噪声/静音
            nFFT = 2 * 2 ** (nextpow2.nextpow2(len_))
            noise_mean = np.zeros(nFFT)

            j = 0
            for k in range(1, 6):
                noise_mean = noise_mean + abs(np.fft.fft(win * x[j:j + len_], nFFT))
                j = j + len_
                noise_mu = noise_mean / 5
            k = 1
            img = 1j
            x_old = np.zeros(len1)
            Nframes = len(x) // len2 - 1
            xfinal = np.zeros(Nframes * len2)

            for n in range(0, Nframes):
                # Windowing
                insign = win * x[k - 1:k + len_ - 1]
                # 计算帧的傅里叶变换
                spec = np.fft.fft(insign, nFFT)
                # 计算幅度
                sig = abs(spec)

                # 保存嘈杂的相位信息
                theta = np.angle(spec)
                SNRseg = 10 * np.log10(np.linalg.norm(sig, 2) ** 2 / np.linalg.norm(noise_mu, 2) ** 2)

                if Expnt == 1.0:  # 幅度谱
                    alpha = self.berouti1(SNRseg)

                else:  # 功率谱
                    alpha = self.berouti(SNRseg)
                    #############
                sub_speech = sig ** Expnt - alpha * noise_mu ** Expnt
                # 当纯净信号小于噪声信号的功率时
                diffw = sub_speech - beta * noise_mu ** Expnt
                # beta负面组件

                z = self.find_index(diffw)  # z = Recognize.find_index(diffw)
                if len(z) > 0:
                    # 用估计出来的噪声信号表示下限值
                    sub_speech[z] = beta * noise_mu[z] ** Expnt
                    if SNRseg < Thres:  # 更新噪声频谱
                        noise_temp = G * noise_mu ** Expnt + (1 - G) * sig ** Expnt  # 平滑处理噪声功率谱
                        noise_mu = noise_temp ** (1 / Expnt)  # 新的噪声幅度谱
                    # flipud函数实现矩阵的上下翻转，是以矩阵的“水平中线”为对称轴
                    # 交换上下对称元素
                    sub_speech[nFFT // 2 + 1:nFFT] = np.flipud(sub_speech[1:nFFT // 2])
                    x_phase = (sub_speech ** (1 / Expnt)) * (
                                np.array([math.cos(x) for x in theta]) + img * (np.array([math.sin(x) for x in theta])))

                    xi = np.fft.ifft(x_phase).real
                    # --- Overlap and add ---------------
                    xfinal[k - 1:k + len2 - 1] = x_old + xi[0:len1]
                    x_old = xi[0 + len1:len_]
                    k = k + len2
            # 保存文件
            wf = wave.open('example/denoiseout.wav', 'wb')
            # 设置参数
            wf.setparams(params)
            # 设置波形文件 .tostring()将array转换为data
            wave_data = (winGain * xfinal).astype(np.short)
            wf.writeframes(wave_data.tostring())
            wf.close()
