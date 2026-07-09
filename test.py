import matplotlib.pyplot as plt
from Installation.GiantFFTResample import Resampler, resample
import numpy.typing as npt
import numpy as np
from dataclasses import dataclass
import scipy.signal as sig
import os

os.makedirs("./Figures", exist_ok=True)


def dB20(x, eps: float = 1e-10):
    return 20 * np.log(np.maximum(np.abs(x), eps))


def dB10(x, eps: float = 1e-10):
    return 10 * np.log(np.maximum(np.abs(x), eps))


@dataclass
class AudioSignal:
    signal: npt.ArrayLike
    samplerate: int

    def get_time(self):
        return np.arange(self.signal.shape[1]) / self.samplerate

    def get_rfft(self, n=None):
        out = np.fft.rfft(self.signal, n=n)
        return out

    def get_rfftfreq(self, n=None):
        if n is None:
            n = self.signal.shape[1]
        out = np.fft.rfftfreq(n=n, d=1 / self.samplerate)
        return out


in_sr = 48000
out_sr = 32000

# x = track.audio.T
# x = np.mean(x, axis=0, keepdims=True)
# x = x[:, :in_sr*15]
length_s = 4
t = np.arange(int(length_s * in_sr)) / in_sr
x = sig.chirp(t=t, t1=length_s, f0=0, f1=in_sr // 2, method="linear").reshape(1, -1)

x_pad = np.zeros((1, int((length_s + 1) * in_sr)))
nstart = int(0.5 * in_sr)
nend = int(nstart + length_s * in_sr)
x_pad[:, nstart:nend] = x
x = x_pad

x = AudioSignal(signal=x, samplerate=in_sr)
# x = np.mean(x, axis=0)

RS = Resampler(in_samplerate=x.samplerate, out_samplerate=out_sr, lowpass_ratio=0.05)

# y = hpf(x)
y = AudioSignal(signal=RS(x.signal), samplerate=out_sr)
y = AudioSignal(
    signal=resample(
        x.signal, in_samplerate=x.samplerate, out_samplerate=out_sr, lowpass_ratio=0.05
    ),
    samplerate=out_sr,
)

n_fft = 1024

cm = 1 / 2.54
fig, axes = plt.subplots(
    2, 2, sharey=False, width_ratios=(10, 1), dpi=300, figsize=(14 * cm, 8 * cm)
)


_, _, _, img = axes[0, 0].specgram(
    x=x.signal[0],
    Fs=x.samplerate,
    vmin=-260,
    vmax=-20,
    NFFT=n_fft,
    scale="dB",
    rasterized=True,
)
plt.colorbar(img, cax=axes[0, 1], label="Power (dB)")

_, _, _, img = axes[1, 0].specgram(
    x=y.signal[0],
    Fs=y.samplerate,
    vmin=-260,
    vmax=-20,
    NFFT=n_fft,
    scale="dB",
    rasterized=True,
)
axes[0, 0].set_ylabel("Frequency (Hz)")
axes[1, 0].set_xlabel("Time (s)")
axes[1, 0].set_ylabel("Frequency (Hz)")
plt.colorbar(img, cax=axes[1, 1], label="Power (dB)")

# plt.colorbar()
plt.tight_layout()
plt.savefig("Figures/test_sweep_spectrogram.pdf")
# plt.show()
plt.close()


fig, axes = plt.subplots(2, 1, sharex=True, figsize=(14 * cm, 8 * cm), dpi=300)


axes[0].plot(x.get_time(), x.signal[0], linewidth=0.1, color="k", rasterized=True)
axes[1].plot(y.get_time(), y.signal[0], linewidth=0.1, color="k", rasterized=True)

axes[0].set_xlim(0, length_s + 1)

axes[0].set_xlabel("Time (s)")

# plt.colorbar()
plt.tight_layout()
plt.savefig("Figures/test_sweep_waveform.pdf")
# plt.show()
plt.close()
