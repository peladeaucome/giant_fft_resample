import matplotlib.pyplot as plt
from Installation.GiantFFTResample.torch import TorchResampler, resample
import numpy as np
import scipy.signal as sig
import torch
import os

os.makedirs("./Figures/torch", exist_ok=True)

device = torch.device("cuda:0")


def dB20(x, eps: float = 1e-10):
    return 20 * np.log(np.maximum(np.abs(x), eps))


def dB10(x, eps: float = 1e-10):
    return 10 * np.log(np.maximum(np.abs(x), eps))


class AudioSignal:
    def __init__(self, signal: torch.Tensor, samplerate: int):
        self.signal = signal
        self.samplerate = samplerate

    def get_time(self):
        return torch.arange(self.signal.shape[-1], device=self.signal.device) / self.samplerate

    def signal_numpy(self):
        return self.signal.cpu().numpy()

    def time_numpy(self):
        return self.get_time().cpu().numpy()


in_sr = 48000
out_sr = 32000

length_s = 4
t = np.arange(int(length_s * in_sr)) / in_sr
x_np = sig.chirp(t=t, t1=length_s, f0=0, f1=in_sr // 2, method="linear").reshape(1, -1)

x_pad = np.zeros((int((length_s + 1) * in_sr)))
nstart = int(0.5 * in_sr)
nend = int(nstart + length_s * in_sr)
x_pad[nstart:nend] = x_np
x_np = x_pad

x = AudioSignal(
    signal=torch.from_numpy(x_np).to(device),
    samplerate=in_sr,
)

RS = TorchResampler(in_samplerate=x.samplerate, out_samplerate=out_sr, lowpass_ratio=0.05)

y = AudioSignal(signal=RS(x.signal), samplerate=out_sr)
y = AudioSignal(
    signal=resample(x.signal, in_samplerate=x.samplerate, out_samplerate=out_sr, lowpass_ratio=0.05),
    samplerate=out_sr,
)

n_fft = 1024

cm = 1 / 2.54
fig, axes = plt.subplots(
    2, 2, sharey=False, width_ratios=(10, 1), dpi=300, figsize=(14 * cm, 8 * cm)
)

_, _, _, img = axes[0, 0].specgram(
    x=x.signal_numpy(),
    Fs=x.samplerate,
    vmin=-260,
    vmax=-20,
    NFFT=n_fft,
    scale="dB",
    rasterized=True,
)
plt.colorbar(img, cax=axes[0, 1], label="Power (dB)")

_, _, _, img = axes[1, 0].specgram(
    x=y.signal_numpy(),
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

plt.tight_layout()
plt.savefig("Figures/torch/test_sweep_spectrogram.pdf")
plt.close()


fig, axes = plt.subplots(2, 1, sharex=True, figsize=(14 * cm, 8 * cm), dpi=300)

axes[0].plot(x.time_numpy(), x.signal_numpy(), linewidth=0.1, color="k", rasterized=True)
axes[1].plot(y.time_numpy(), y.signal_numpy(), linewidth=0.1, color="k", rasterized=True)

axes[0].set_xlim(0, length_s + 1)
axes[0].set_xlabel("Time (s)")

plt.tight_layout()
plt.savefig("Figures/torch/test_sweep_waveform.pdf")
plt.close()
