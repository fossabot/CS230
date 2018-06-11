import numpy as np
import matplotlib.pyplot as plt

model_1_stats = np.load('model_1_stats.npy').item()
model_1_fft_stats = np.load('model_1_fft_stats.npy').item()
model_2_stats = np.load('model_2_stats.npy').item()
model_2_fft_stats = np.load('model_2_fft_stats.npy').item()

print model_1_stats['train_accs']
print model_1_stats['dev_accs']
print model_1_stats['dev_aurocs']

print model_1_fft_stats['train_accs']
print model_1_fft_stats['dev_accs']
print model_1_fft_stats['dev_aurocs']

print model_2_stats['train_accs']
print model_2_stats['dev_accs']
print model_2_stats['dev_aurocs']

print model_2_fft_stats['train_accs']
print model_2_fft_stats['dev_accs']
print model_2_fft_stats['dev_aurocs']
