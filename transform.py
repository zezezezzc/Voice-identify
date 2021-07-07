import numpy
from scipy.fftpack import dct
import math
from init_setting import *      #导入初始化脚本
import sigproc
try:
    xrange(1)       # 产生一个较大的集合,出现异常运行excep代码
except:
    xrange = range
# 提取MFCC算法
def mfcc(signal, sample_rate=SAMPLING_RATE, winlen=FRAME_LEN_MS, winstep=INPUT_LEN_MS, preemph=PRE_EMPHASIS_COEF,
numcep=DIM_OF_MFCC_FEAT, lowfreq=0, highfreq=None, nfilt=26, nfft=NFFT, ceplifter=22, appendEnergy=True, 
winfunc=lambda x:numpy.ones((x,)), normalization=0, logscale=0, delta=0):

    feat, energy = fbank(signal,sample_rate, winlen, winstep, nfilt, nfft, lowfreq, highfreq, preemph, winfunc)

    #normalization & log scale
    if normalization == 1:
        # non zero value for log scale
        energy = numpy.where(energy == 0, numpy.finfo(float).eps, energy)
        featt = numpy.transpose(feat)
        featt = featt / energy
        feat = numpy.transpose(featt)

    feat = numpy.log(feat)
    feat = dct(feat, type=2, axis=1, norm='ortho')[:, :numcep]
    feat = lifter(feat, ceplifter)

    if appendEnergy:
        feat[:, 0] = numpy.log(energy)

    if delta > 0:
        feat_tmp = numpy.zeros(shape=(feat.shape[0]+4, feat.shape[1]))
        feat_dev = numpy.zeros(shape=(feat.shape))
        feat_ret = numpy.zeros(shape=(feat.shape[0], feat.shape[1] + delta))
        feat_tmp[range(2, len(feat)+2)] = feat
        for i in range(0, len(feat)):
            feat_dev[i] = (2*feat_tmp[i+2] + feat_tmp[i+1] - (2*feat_tmp[i-2]+feat_tmp[i-1])) / 10
        feat_ret[:, range(0, feat.shape[1])] = feat
        feat_ret[:, range(feat.shape[1], feat_ret.shape[1])] = feat_dev[:, range(0, delta)]
        feat = feat_ret
        del feat_ret
        del feat_tmp
        del feat_dev

    return feat

def fbank(signal, sample_rate=16000, winlen=0.025, winstep=0.01, nfilt=26, nfft=512, lowfreq=0, highfreq=None,
        preemph=0.97, winfunc=lambda x:numpy.ones((x,))):
    ##预处理
    highfreq = highfreq or sample_rate/2
    signal = sigproc.preemphasis(signal, preemph)
    frames = sigproc.framesig(signal, winlen*sample_rate, winstep*sample_rate, winfunc)
    pspec = sigproc.powspec(frames, nfft)
    energy = numpy.sum(pspec, 1)
    energy = numpy.where(energy == 0, numpy.finfo(float).eps, energy)

    fb = get_filterbanks(nfilt, nfft, sample_rate, lowfreq, highfreq)
    feat = numpy.dot(pspec, fb.T)
    feat = numpy.where(feat == 0, numpy.finfo(float).eps, feat)

    return feat, energy

def get_filterbanks(nfilt=20, nfft=512, sample_rate=16000, lowfreq=0, highfreq=None):
    highfreq = highfreq or sample_rate/2
    assert highfreq <= sample_rate/2

    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = numpy.linspace(lowmel, highmel, nfilt+2)
    bin = numpy.floor((nfft + 1) * mel2hz(melpoints) / sample_rate)

    fbank = numpy.zeros([nfilt, nfft // 2 + 1])
    for j in xrange(0, nfilt):
        for i in xrange(int(bin[j]), int(bin[j+1])):
            fbank[j, i] = (i - bin[j]) / (bin[j+1] - bin[j])
        for i in xrange(int(bin[j+1]), int(bin[j+2])):
            fbank[j, i] = (bin[j+2] - i) / (bin[j+2] - bin[j+1])
    return fbank

def hz2mel(hz):
    return 2595 * numpy.log10(1+hz/700.0)

def mel2hz(mel):
    return 700*(10**(mel/2595.0)-1)

def lifter(cepstra, L=22):
    if L > 0:
        nframes, ncoeff = numpy.shape(cepstra)
        n = numpy.arange(ncoeff)
        lift = 1 + (L/2) * numpy.sin(numpy.pi * n / L)
        return lift * cepstra
    else:
        return cepstra
