import decimal      #定点数和浮点数的数学运算，解决python中浮点数运算精度有误差的问题
import numpy
import math

# 预处理模块

def round_half_up(number):
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))

def framesig(sig, frame_len, frame_step, winfunc=lambda x:numpy.ones((x,))):
    ##加窗，分帧
    slen = len(sig)
    frame_len = int(round_half_up(frame_len))
    frame_step = int(round_half_up(frame_step))
    if slen <= frame_len:
        numframes = 1
    else:
        numframes = 1 + int(math.ceil((1.0 * slen - frame_len) / frame_step))

    padlen = int((numframes-1) * frame_step + frame_len)

    zeros = numpy.zeros((padlen - slen,))
    padsignal = numpy.concatenate((sig, zeros))

    ## np.title 周期性重复
    indices = numpy.tile(numpy.arange(0, frame_len), (numframes, 1)) + numpy.tile(numpy.arange(0, numframes * frame_step, frame_step), (frame_len,1)).T
    indices = numpy.array(indices, dtype=numpy.int32)
    frames = padsignal[indices]
    win = numpy.tile(winfunc(frame_len), (numframes, 1))
    return frames * win

def powspec(frames, NFFT):
    return 1.0 / NFFT * numpy.square(magspec(frames, NFFT))

def preemphasis(signal, coeff=0.95):
    return numpy.append(signal[0], signal[1:]-coeff*signal[:-1])

def magspec(frames, NFFT):
    complex_spec = numpy.fft.rfft(frames, NFFT)
    return numpy.absolute(complex_spec)



'''
def enframe(signal, nw, inc):
    ##将音频信号转化为帧
    # 参数含义
    # signal:原始音频型号
    # nw:每一帧的长度(这里指采样点的长度，即采样频率乘以时间间隔)
    # inc:相邻帧的间隔（同上定义）
    
    signal_length=len(signal) #信号总长度
    if signal_length<=nw: #若信号长度小于一个帧的长度，则帧数定义为1
        nf=1
    else: #否则，计算帧的总长度
        nf=int(np.ceil((1.0*signal_length-nw+inc)/inc))
    pad_length=int((nf-1)*inc+nw) #所有帧加起来总的铺平后的长度
    zeros=np.zeros((pad_length-signal_length,)) #不够的长度使用0填补，类似于FFT中的扩充数组操作
    pad_signal=np.concatenate((signal,zeros)) #填补后的信号记为pad_signal
    indices=np.tile(np.arange(0,nw),(nf,1))+np.tile(np.arange(0,nf*inc,inc),(nw,1)).T  #相当于对所有帧的时间点进行抽取，得到nf*nw长度的矩阵
    indices=np.array(indices,dtype=np.int32) #将indices转化为矩阵
    frames=pad_signal[indices] #得到帧信号
    # win=np.tile(winfunc(nw),(nf,1))  #window窗函数，这里默认取1
    # return frames*win   #返回帧信号矩阵
    return frames


def wavread(filename):
    f = wave.open(filename,'rb')
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    strData = f.readframes(nframes)#读取音频，字符串格式
    waveData = np.fromstring(strData,dtype=np.int16)#将字符串转化为int
    f.close()
    waveData = waveData*1.0/(max(abs(waveData)))#wave幅值归一化
    waveData = np.reshape(waveData,[nframes,nchannels]).T
    return waveData

filepath = "./data/" #添加路径
dirname= os.listdir(filepath) #得到文件夹下的所有文件名称 
filename = filepath+dirname[0]
data = wavread(filename)
nw = 512
inc = 128
Frame = enframe(data[0], nw, inc) 


##如果需要加窗，只需要将函数修改为：

def enframe(signal, nw, inc, winfunc):
    ## 将音频信号转化为帧。
    #参数含义：
    # signal:原始音频型号
    # nw:每一帧的长度(这里指采样点的长度，即采样频率乘以时间间隔)
    # inc:相邻帧的间隔（同上定义）
    
    signal_length=len(signal) #信号总长度
    if signal_length<=nw: #若信号长度小于一个帧的长度，则帧数定义为1
        nf=1
    else: #否则，计算帧的总长度
        nf=int(np.ceil((1.0*signal_length-nw+inc)/inc))
    pad_length=int((nf-1)*inc+nw) #所有帧加起来总的铺平后的长度
    zeros=np.zeros((pad_length-signal_length,)) #不够的长度使用0填补，类似于FFT中的扩充数组操作
    pad_signal=np.concatenate((signal,zeros)) #填补后的信号记为pad_signal
    indices=np.tile(np.arange(0,nw),(nf,1))+np.tile(np.arange(0,nf*inc,inc),(nw,1)).T  #相当于对所有帧的时间点进行抽取，得到nf*nw长度的矩阵
    indices=np.array(indices,dtype=np.int32) #将indices转化为矩阵
    frames=pad_signal[indices] #得到帧信号
    win=np.tile(winfunc,(nf,1))  #window窗函数，这里默认取1
    return frames*win   #返回帧信号矩阵
　　其中窗函数，以hamming窗为例：

winfunc = signal.hamming(nw)
Frame = enframe(data[0], nw, inc, winfunc)
'''
