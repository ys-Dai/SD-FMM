import numpy as np
import pywt
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import copy


def visualize_wavelet_coefficients(signal, wavelet):
    max_level = pywt.dwt_max_level(len(signal), wavelet)
    print(max_level)
    coeffs = pywt.wavedec(signal, wavelet, level=max_level)
    plt.figure(figsize=(12, 8))
    for i in range(0, len(coeffs)):
        if i == 0:
            plt.subplot(max_level+1, 1, i+1)
            plt.plot(signal)
            print('len signal: ', len(signal))
            print(f'len level {i}: ', len(coeffs[i]))
        else:
            plt.subplot(max_level+1, 1, i+1)
            plt.plot(coeffs[i])
            plt.title(f'Detail coefficients at level {i}')
            print(f'len level {i}: ', len(coeffs[i]))
    plt.tight_layout()
    plt.show()


def visualize_wavelet(signal, wavelet):
    max_level = pywt.dwt_max_level(len(signal), wavelet)
    coeffs = pywt.wavedec(signal, wavelet, level=max_level)
    n = len(signal)
    extended_coeffs = np.zeros((max_level + 1, len(signal)))
    for i, coeff in enumerate(coeffs):
        length = len(coeff)
        repeats = len(signal) // length + (len(signal) % length > 0)
        repeated_coeff = np.repeat(coeff, repeats)[:len(signal)]
        extended_coeffs[i] = repeated_coeff
    heatmap_data = np.array(extended_coeffs)
    reduced_matrix = np.delete(heatmap_data, 0, axis=0)
    squared_matrix = reduced_matrix ** 2
    plt.figure(figsize=(10, 6))
    x = np.arange(n+1)
    y = np.arange(max_level+1)
    cax = plt.pcolormesh(x, y, squared_matrix, cmap='Blues', shading='flat')
    plt.colorbar(cax, label='Energy')
    plt.xlabel('Time')
    plt.ylabel('Decomposition Level')
    plt.title('Wavelet Decomposition Energy Distribution')
    plt.show()


def per_flat_signal(flat_signal):
    f_signal = copy.deepcopy(flat_signal)
    for i in range(len(f_signal)):
        signal_sub = copy.deepcopy(f_signal[i])
        sign = np.random.choice([-1, 1], size=(len(signal_sub),))
        if i == 0:
            per = np.random.uniform(-2, 2, size=(len(signal_sub),))
            signal_sub = signal_sub + per
            signal_sub = signal_sub * sign
            f_signal[i] = signal_sub
        else:
            per = sign * np.random.uniform(-1, 1, size=(len(signal_sub),))
            signal_sub = signal_sub + per
            signal_sub = signal_sub * sign
            f_signal[i] = signal_sub
    return f_signal


def perturb_arrays_entire(coeffs_ori, per_max, end_bias, pmin, pmax):
    array_list = copy.deepcopy(coeffs_ori)
    coeffs_out = [0, 0, 0, 0, 0, 0]
    n = len(array_list)
    starts = [None] * n
    ends = [None] * n
    for i in range(n-1, -1, -1):
        a = copy.deepcopy(array_list[i])
        if i == n-1:
            start = np.random.randint(0, len(a))
            per = np.random.randint(0, per_max+1)
            end = min(start + per, len(a)-1)
            starts[i] = start
            ends[i] = end
        elif i == 0:
            starts[i] = starts[i+1]
            ends[i] = ends[i+1]
        else:
            start = (1 + starts[i+1]) // 2
            end = (1 + ends[i+1]) // 2
            bias = np.random.randint(-end_bias, end_bias + 1)
            end = min(max(end + bias, start), len(a) - 1)
            starts[i] = start
            ends[i] = end
        sign = np.random.choice([-1, 1], size=((ends[i] + 1) - starts[i],))
        new_values = sign * np.random.uniform(pmin[i], pmax[i], size=((ends[i] + 1) - starts[i],))
        a[starts[i]:ends[i]+1] = new_values
        coeffs_out[i] = a
    return coeffs_out, starts[-1], ends[-1]


def v_newsignal_shadow_save(y_value, shadow=None, shadow_color='orange', shadow_alpha=0.3,
                            save_fig=True, save_path=None, fig_name=None, y_max=None,
                            x_last=None, xname=None, title_name=None,
                            yvalue_size=14, xname_size=16, title_size=18):
    y = y_value
    plt.figure(figsize=(9.26, 5))
    new_time = np.arange(0, len(y))
    plt.plot(new_time, y, color='#e50000', linewidth=2, alpha=0.9)
    if x_last != None:
        plt.xticks([0, len(new_time)-1], ['0', x_last], fontsize=yvalue_size)
    else:
        plt.xticks([])
    plt.yticks([])
    plt.xlabel(xname, fontsize=xname_size)
    plt.title(title_name, fontsize=title_size)
    if shadow != None:
        for i in range(len(shadow)):
            start = shadow[i][0]
            end = shadow[i][1]
            shadow_start = start
            shadow_end = end
            plt.axvspan(shadow_start, shadow_end, color=shadow_color, alpha=shadow_alpha)
    if save_fig:
        plt.savefig(save_path+fig_name+'.pdf', bbox_inches='tight')
    plt.close()


def generate_new_signal(f_signal, output_path, f_name):
    ran_coeff_f = per_flat_signal(f_signal)
    new_coeff_per, ano_start, ano_end = perturb_arrays_entire(ran_coeff_f, per_max, end_bias, pmin, pmax)
    n_signal = pywt.waverec(new_coeff_per, wavelet)
    n_signal[n_signal < 0] *= -1
    shadow_start = min(max(0, ano_start * 2 - 1), len(n_signal)-1)
    shadow_end = min(ano_end * 2 + 1, len(n_signal)-1)
    np.save(output_path+'signals/' + f_name + '_newsignal.npy', n_signal)
    ano = np.array([shadow_start, shadow_end])
    np.save(output_path+'anotations/' + f_name + '_ano.npy', ano)


if __name__ == "__main__":
    per_max = 5
    end_bias = 1
    pmin = [12, 10, 9, 3, 5, 16]
    pmax = [20, 15, 12, 9, 9, 20]
    wavelet = 'db2'
    generate_new_signal(f_signal, output_path, f_name)
