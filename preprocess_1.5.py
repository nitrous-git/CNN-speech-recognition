import numpy as np 
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt

########################################################
# Preprocess needed to build spectogram (.jpg) data set 
# in order to do the speech analysis.
# process of 3 separate speech command : no, yes, down 
########################################################


######## convert no_datas to specgrams .jpg
for i in range(1,201):
    audio_dir = r'C:\Users\Of Corrupted Vision\Documents\Source Python\CNN speech recognition\train_data\no\no ('+str(i)+').wav'
    rate_data, audio_data = wavfile.read(audio_dir)

    # remove borders
    fig, ax = plt.subplots(1)
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)

    # plt the specgrams and save files
    Pxx, freqs, bins, im = ax.specgram(audio_data, NFFT=1024, Fs=44100, noverlap=900)
    ax.axis('off')
    ax.axis('tight')
    fig.savefig(r'C:\Users\Of Corrupted Vision\Documents\Source Python\CNN speech recognition\train_specgram\no\sg_no'+str(i)+'.jpg')
    fig.savefig(r'C:\Users\Of Corrupted Vision\Documents\Source Python\CNN speech recognition\validation_specgram\no\sg_no'+str(i)+'.jpg')

######## convert yes_datas to specgrams .jpg
for i in range(1,201):
    audio_dir = r'C:\Users\Of Corrupted Vision\Documents\Source Python\CNN speech recognition\train_data\yes\yes ('+str(i)+').wav'
    rate_data, audio_data = wavfile.read(audio_dir)

    # remove borders
    fig, ax = plt.subplots(1)
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)

    # plt the specgrams and save files
    Pxx, freqs, bins, im = ax.specgram(audio_data, NFFT=1024, Fs=44100, noverlap=900)
    ax.axis('off')
    ax.axis('tight')
    fig.savefig(r'C:\Users\Of Corrupted Vision\Documents\Source Python\CNN speech recognition\train_specgram\yes\sg_yes'+str(i)+'.jpg')
    fig.savefig(r'C:\Users\Of Corrupted Vision\Documents\Source Python\CNN speech recognition\validation_specgram\yes\sg_yes'+str(i)+'.jpg')

######## convert down_datas to specgrams .jpg
for i in range(1,201):
    audio_dir = r'C:\Users\Of Corrupted Vision\Documents\Source Python\CNN speech recognition\train_data\down\down ('+str(i)+').wav'
    rate_data, audio_data = wavfile.read(audio_dir)

    # remove borders
    fig, ax = plt.subplots(1)
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)

    # plt the specgrams and save files
    Pxx, freqs, bins, im = ax.specgram(audio_data, NFFT=1024, Fs=44100, noverlap=900)
    ax.axis('off')
    ax.axis('tight')
    fig.savefig(r'C:\Users\Of Corrupted Vision\Documents\Source Python\CNN speech recognition\train_specgram\down\sg_down'+str(i)+'.jpg')
    fig.savefig(r'C:\Users\Of Corrupted Vision\Documents\Source Python\CNN speech recognition\validation_specgram\down\sg_down'+str(i)+'.jpg')