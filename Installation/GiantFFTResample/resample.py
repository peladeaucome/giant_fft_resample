import numpy as np
from dataclasses import dataclass


def get_taper(length_sp):
    n = np.arange(length_sp)

    out = np.square(np.cos(n * np.pi * 0.5 / length_sp))
    return out


def get_P_and_Q(in_sr, out_sr):
    up = out_sr > in_sr

    G = max(in_sr, out_sr)
    L = min(in_sr, out_sr)

    if L == 16000:
        if G == 22050:
            p, q = 441, 320
        elif G == 32000:
            p, q = 2, 1
        elif G == 44100:
            p, q = 441, 160
        elif G == 48000:
            p, q = 3, 1
        elif G == 96000:
            p, q = 6, 1
        elif G == 192000:
            p, q = 12, 1
    if L == 22050:
        if G == 32000:
            p, q = 160, 147
        elif G == 44100:
            p, q = 2, 1
        elif G == 48000:
            p, q = 320, 147
        elif G == 96000:
            p, q = 640, 147
        elif G == 192000:
            p, q = 1280, 147
    if L == 32000:
        if G == 44100:
            p, q = 441, 320
        elif G == 48000:
            p, q = 3, 2
        elif G == 96000:
            p, q = 3, 1
        elif G == 192000:
            p, q = 6, 1
    elif L == 44100:
        if G == 48000:
            p, q = 160, 147
        elif G == 96000:
            p, q = 320, 147
        elif G == 192000:
            p, q = 640, 147
    elif L == 48000:
        if G == 96000:
            p, q = 2, 1
        elif G == 192000:
            p, q = 4, 1
    elif L == 96000:
        if G == 192000:
            p, q = 2, 1

    if up:
        return p, q
    else:
        return q, p


@dataclass
class Resampler:
    in_samplerate: int
    out_samplerate: int
    lowpass_ratio: int = 0.05

    def __post_init__(self):
        accepted_samplerates = [16000, 22050, 32000, 44100, 48000, 96000, 192000]
        if (
            self.in_samplerate not in accepted_samplerates
            or self.out_samplerate not in accepted_samplerates
        ):
            raise NotImplementedError(
                f"Samplerates are {self.in_samplerate} and {self.out_samplerate}. Please select a valid samplerate"
            )
        if self.in_samplerate == self.out_samplerate:
            raise ValueError("Input samplerate is equal to output samplerate")
        else:
            self.P, self.Q = get_P_and_Q(self.in_samplerate, self.out_samplerate)

    def resample(self, x: np.ndarray):
        numChannels, Nin = x.shape

        M = int(2 * np.ceil((Nin + self.in_samplerate * 0.5) / (self.Q * 2)))

        Nx = M * self.Q
        Ny = M * self.P

        nbins_in = Nx // 2 + 1
        nbins_out = Ny // 2 + 1

        x_fft = np.fft.rfft(x, axis=1, n=Nx)
        ## TAPERING
        if self.out_samplerate > self.in_samplerate:
            Ntaper = int(self.lowpass_ratio * nbins_in)
            taper = get_taper(Ntaper).reshape(1, Ntaper)
            x_fft[:, -Ntaper:] *= taper
            ## Zero-padding
            y_fft = np.zeros((numChannels, nbins_out), dtype=np.complex128)
            y_fft[:, :nbins_in] = x_fft
        else:
            y_fft = np.zeros((numChannels, nbins_out), dtype=np.complex128)
            y_fft[:, :nbins_out] = x_fft[:, :nbins_out]
            y_fft[:, -1] = np.real(y_fft[:, -1])

            Ntaper = int(self.lowpass_ratio * nbins_out)
            taper = get_taper(Ntaper).reshape(1, Ntaper)
            y_fft[:, -Ntaper:] *= taper

        # IFFT
        y = np.fft.irfft(y_fft, axis=1, n=Ny)

        # Scaling
        y = y * self.P / self.Q

        # Remove additional zeros
        Nout = int(Nin * self.P / self.Q)
        y = y[:, :Nout]
        return y

    def __call__(self, x):
        return self.resample(x)


def resample(
    x: np.ndarray, in_samplerate: int, out_samplerate: int, lowpass_ratio: float = 0.05
):
    """
    Function to resample

    args:
    -----
     - `x` np.ndarray, shape (Nchannels, Nsamples): signal to resample
     - `in_samplerate` input samplerate
     - `out_samplerate` output samplerate
     - `lowpass_ratio` ratio of frequencies lowpassed. The lower the value, the higher the lowpass frequency
    """
    RS = Resampler(in_samplerate, out_samplerate, lowpass_ratio)
    y = RS(x)
    return y
