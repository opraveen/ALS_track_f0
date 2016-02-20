import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.io.wavfile as wav
import scipy.signal as signal
from scipy.signal import cheby1, cheb1ord, lfilter, filtfilt
from scipy.interpolate import interp1d

# Matlab ref: http://cseweb.ucsd.edu/~saul/matlab/track_f0.m
# Paper ref: http://cseweb.ucsd.edu/~saul/papers/voice_nips02.pdf
# Estimate the fundamental freq (f0) of a voice sample, using Adaptive Least Squares (ALS) algorithm
# This is a python adaptation of above matlab code
# TODO: Test using voice samples
# TODO: Check VOLUME filter implementation against matlab code


def track_f0(sampling_freq, dec_rate, wave, window_size, voicing_thresh, silence_thresh):

    '''

    :param sampling_freq: 44100 Hz, typically
    :param dec_rate: 6 or 8, as we are interested only in human voice (50 - 400 Hz)
    :param wave: the sound wave data (assuming a sampling frequency of 44100Hz)
    :param window_size: analysis window size (in seconds). A good choice is 0.04s
    :param voicing_thresh: cost must be below this threshold to be considered as
            a good sinusoidal fit. For Edinburgh data, it was chosen at 0.0825
    :param silence_thresh: energy of the signal must be above this threshold to
            be considered as voiced region. For Edinbugh data, it was
            chosen at 1e2. The silenceThresh is more or less determined by
            the energy level, therefore should be scaled accordingly to the
            data being examined.
    :return:
    f0: the estimated fundamental frequency
    tt: the time stamp (in seconds) when the fundamental frequency is estimated

    ALS algorithm:
    http://www.cis.upenn.edu/~lsaul/papers/voice_nips02.pdf
    '''

    fs = sampling_freq
    sr = fs / dec_rate

    print(dec_rate, len(wave))

    # BAND-PASS FILTERS
    if fs % dec_rate:
        print ("The sampling rate is not a multiple of the decimation rate")
        exit()

    fmin = [50, 71, 100, 141, 200, 283, 400, 533]
    fmax = [75, 107, 150, 212, 300, 425, 600, 800]

    nBand = len(fmin)
    order = 4
    ripple = 0.5
    bf = np.zeros((nBand, 2*order+1))
    af = np.zeros((nBand, 2*order+1))

    for band in range(nBand):
        pass_band = [1.0*fmin[band]/sr, 1.0*fmax[band]/(sr/2)]
        bf[band, :], af[band, :] = cheby1(order, ripple, pass_band, btype='band')
        # plot_filter_response(bf[band, :], af[band, :])

    blp, alp = cheby1(order, ripple, 1.0*1000/(fs/2), btype='low')  # 1KHZ LOW PASS FOR DECIMATION
    plot_filter_response(blp, alp)

    # TRACK
    envA = filtfilt(blp, alp, wave)                         # 1KHZ low-pass filter
    envA = signal.resample(envA, len(envA)/dec_rate)        # Down-sample by dec_rate
    envB = np.maximum(0, envA)                              # Non-linearity, half-wave rectification

    nFrame = len(envB)
    freqs = np.zeros((nBand, nFrame))
    costs = np.zeros((nBand, nFrame))

    for band in range(nBand):
        sine_wave = filtfilt(bf[band, :], af[band, :], envB)

        # plt.plot(range(len(sine_wave)), sine_wave)
        # plt.show()
        # exit()

        freqs[band, :], costs[band, :] = track_band(sine_wave, sr, window_size, fmin[band])

    print(costs.shape, freqs.shape, costs)

    min_idx = np.argmin(costs, axis=0)
    cost = np.min(costs, axis=0)

    # PITCH
    pitch = np.diag(freqs[min_idx])

    # VOLUME: TODO Check against matlab implementation
    volume = lfilter(1.0*np.ones(dec_rate)/dec_rate, 1, wave*wave)
    volume = volume[0::dec_rate]

    # VOICED/UNVOICED determination
    voiced = np.where(cost < voicing_thresh) and np.where(volume > silence_thresh)
    f0 = np.zeros(len(pitch))
    f0[voiced] = pitch[voiced]

    print(f0, f0.shape)

    # ALS fitting uses preceding wave data to compose an analysis window
    # therefore, we shift half of the window and interpolate
    tt = 1.0*np.arange(nFrame)/sr - 0.5*window_size
    uu = 1.0*np.arange(nFrame)/sr

    print(tt, uu, f0.shape, tt.shape, uu.shape)
    f0 = interp1d(tt, f0, bounds_error=False, fill_value=0., kind='nearest')(uu)
    tt = uu

    return f0, tt


def plot_filter_response(b, a):
    w, h = signal.freqs(b, a)
    plt.plot(w, 20 * np.log10(abs(h)))
    plt.xscale('log')
    plt.title('Chebyl1 filter frequency response')
    plt.xlabel('Frequency [radians / second]')
    plt.ylabel('Amplitude [dB]')
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    plt.axvline(100, color='green')   # cutoff frequency
    plt.show()


def track_band(xx, sampling_rate, window_size, min_f0):
    '''
    :param xx: Audio sample
    :param sampling_rate: Sampling rate of the audio sample
    :param window_size: Analysis window size
    :param min_f0: Minimum frequency of interest
    :return:

        f0: Estimated frequency at each instant (sample point)
        cost: Estimated heuristic fitting cost

    Description: Fit wave data to a sinusoid, returning estimated frequencies and
    heuristic fitting costs
    '''

    # Sinusoidal Fit
    nn = math.ceil(window_size*sampling_rate)  # ANALYSIS WINDOW SIZE
    mm = xx[0:-2] + xx[2:]
    mm = np.insert(mm, 0, 2*xx[0])
    mm = np.append(mm, 2*xx[-1])

    xm = np.cumsum(xx*mm)
    m2 = np.cumsum(mm*mm)
    x2 = np.cumsum(xx*xx)

    # To account for the sliding window size, decrement the 'excess' for all indices > analysis_window_size
    xm[nn:] -= xm[0:-nn]
    m2[nn:] -= m2[0:-nn]
    x2[nn:] -= x2[0:-nn]

    # Equation 4 in NIPS paper
    aa = 2.0*xm/(m2 + approx_min)

    # so that corresponding freq is 0, since freq is proportional to arccos(1.0/aa)
    aa[np.where(m2 == 0)] = 1.0

    # Below Minimum Frequency?
    minP = 2 * np.pi * min_f0 / sampling_rate
    aa[np.where(abs(aa) < 1.0/np.cos(minP))] = 1.0

    # Pitch
    f0 = np.arccos(1.0/aa) * sampling_rate / (2 * np.pi)
    print(f0[-20:])
    f0 = np.minimum(f0, sampling_rate-f0)
    pp = 2 * np.pi * f0 / sampling_rate
    sp = np.sin(pp)
    cp = np.cos(pp)

    # Cost: Equation 6 in NIPS paper
    cost = np.sqrt(abs(x2 + (aa/2)*(aa/2)*m2 - aa*xm)/abs(m2 + approx_min))
    cost = cost * cp * cp * sampling_rate / (np.pi*sp + approx_min)
    cost = cost/(f0 + approx_min)
    cost[np.where(f0 == 0)] = np.Inf
    cost[np.where(m2 == 0)] = np.Inf
    cost[np.where(x2 == 0)] = np.Inf

    return f0, cost


SAMPLING_RATE = 8000
WINDOW_SIZE = 0.04  # In seconds SAMPLING_RATE*50/1000  # 400 samples, equivalent to 50 ms
WINDOW_STRIDE = SAMPLING_RATE*10/1000   # 80 samples, equivalent to 10 ms
VOICING_THRESH = 10000*0.0825  # As recommended in matlab code
SILENCE_THRESH = -0.000001  # 1e-100
approx_min = 1e-40

(rate, sig) = wav.read("./audacity_samples/sine_220Hz_44100SR.wav")
# (rate, sig) = wav.read("./audacity_samples/sine_440Hz_44100SR.wav")

f0, tt = track_f0(rate, rate/SAMPLING_RATE, sig[0:0.2*rate], WINDOW_SIZE, VOICING_THRESH, SILENCE_THRESH)

plt.plot(tt, f0, 'r+')
plt.show()

print(np.mean(f0))
