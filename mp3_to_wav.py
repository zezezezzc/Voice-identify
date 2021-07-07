import os
import wavTools
import wave
# import subprocess
from pydub import AudioSegment
from recognize import Recognize
import librosa
from scipy.io import wavfile

# def wma2mp3(wma_path,mp3_path=None):
#     path, name = os.path.split(wma_path)
#     if name.split('.')[-1]!='wma':
#         print('not a wma file')
#         return 0
#     if mp3_path is None or mp3_path.split('.')[-1]!='mp3':
#         mp3_path = os.path.join(path, name.split('.')[0] +'.mp3')
#         error = subprocess.call(['ffmpeg','-i',wma_path,mp3_path])
#         print(wma_path,' was changed to .MP3',)
#         if error:
#             print('NOT sucess')
#             return 0
#         print('success')
#         #return mp3_path


## 将mp3文件转为wav文件
if __name__ == '__main__':
    rec_obj = Recognize('语音预处理')
    # 接收语音
    # print("请输入您的文件路径：")
    # filepath1 = './cache/柴文娇.mp3'
    # filepath2 = './cache/阳光.mp3'
    # filepath3 = './cache/布玛利亚木.mp3'
    # filepath=[filepath1,filepath2,filepath3]
    # print(filepath)

    # outpath1='./cache/chai.wav'
    # outpath2='./cache/yang.wav'
    # outpath3='./cache/bu.wav'
    # outpath_1=[outpath1,outpath2,outpath3]
    
    # ## mp3_to_wav
    # for i in range(3):
    #     rec_obj.preprocess(filepath[i],outpath_1[i])

    # ## 切片 1s
    # idx_str = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19']
    # outfile1 = './data/chaiwenjiao/chai_'
    # for i in range(20):
    #     out_1s = outfile1 + idx_str[i] +'.wav'
    #     wavTools.get_ms_part_wav(outpath1,1000+i*1000,2000+i*1000,out_1s)

    # outfile2 = './data/yangguang/yang_'
    # for i in range(20):
    #     out_1s = outfile2 + idx_str[i] +'.wav'
    #     wavTools.get_ms_part_wav(outpath2,1000+i*1000,2000+i*1000,out_1s)

    # outfile3 = './data/bumaliya/bu_'
    # for i in range(20):
    #     out_1s = outfile3 + idx_str[i] +'.wav'
    #     wavTools.get_ms_part_wav(outpath3,1000+i*1000,2000+i*1000,out_1s)
    

    ## 重采样
    idx_str_1 = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19']
    outfile1_1 = './data/chaiwenjiao/chai_'
    outfile1_2 = './data/chaiwenjiao/chai_16k_'
    for i in range(20):
        input_44k=outfile1_1 + idx_str_1[i] + '.wav'
        out_16k=outfile1_2 + idx_str_1[i] + '.wav'
        y, sr = librosa.load(input_44k, sr=44100)
        y_16k = librosa.resample(y,sr,16000)
        librosa.output.write_wav(out_16k, y_16k, 16000)

    outfile2_1 = './data/yangguang/yang_'
    outfile2_2 = './data/yangguang/yang_16k_'
    for i in range(20):
        input_44k=outfile2_1 + idx_str_1[i] + '.wav'
        out_16k=outfile2_2 + idx_str_1[i] + '.wav'
        y, sr = librosa.load(input_44k, sr=44100)
        y_16k = librosa.resample(y,sr,16000)
        librosa.output.write_wav(out_16k, y_16k, 16000)

    outfile3_1 = './data/bumaliya/bu_'
    outfile3_2 = './data/bumaliya/bu_16k_'
    for i in range(20):
        input_44k=outfile3_1 + idx_str_1[i] + '.wav'
        out_16k=outfile3_2 + idx_str_1[i] + '.wav'
        y, sr = librosa.load(input_44k, sr=44100)
        y_16k = librosa.resample(y,sr,16000)
        librosa.output.write_wav(out_16k, y_16k, 16000)


    ## 查看wav文件参数
    # ff=wave.open(outpath3)
    # params = ff.getparams()
    # nchannels, sampwidth, framerate, nframes = params[:4]                     # sample_rate, data = wavfile.read(outpath3)
    # print(nchannels, sampwidth, framerate, nframes)                           # print(sample_rate)
        

    # sample_rate, data = wavfile.read(outpath2)
    # print(sample_rate)
    # if sample_rate != 16000:
    #     data = resample(data, sample_rate, 16000)   #对数据压缩格式有要求
    #     print('16k')


    # change *.wma files to *.mp3 at current folder
    
    ## 粘贴中的 if __main__ 函数
    # for d,sd,files in os.walk('.'):
    #         for f in files:
    #             wma_path = os.path.join(d,f)
    #             (prefix,sep,suffix)=wma_path.rpartition('.')
    #             if suffix !='wma':
    #                 continue
    #             mp3_path = prefix+'.mp3'
    #             wma2mp3(wma_path)
   

    # file_name='./cache'
    # labels=os.listdir(file_name)
    # my_labels=['bumali','chaiwenjiao','yangguang']
    # print(labels)
    # for i in range(3):
    #     file_mp3 = file_name +'/'+ my_labels[i] +'.mp3'      #os.path.join(file_name +'/'+ lab)
    #     out_path=file_name +'/'+ my_labels[i] + '.wav'
    #     reg_obj.preprocess(file_mp3,out_path)
        # main_wav_path, start_time, end_time, part_wav_path
        # main_wav_path, start_time, end_time, part_wav_path
        # song = AudioSegment.from_mp3(file_mp3)
        # song.export(out_path, format="wav")
        # wavTools.get_ms_part_wav(file_mp3,3000,7000,out_path)
    