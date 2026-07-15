import numpy as np
from dataclasses import dataclass
from math import gcd


def get_taper(length_sp):
    n = np.arange(length_sp)
    out = np.square(np.cos(n * np.pi * 0.5 / length_sp))
    return out


def get_P_and_Q(in_sr: int, out_sr: int) -> tuple:
    """Compute the reduced rational resampling ratio P/Q = out_sr/in_sr."""
    g = gcd(in_sr, out_sr)
    return out_sr // g, in_sr // g


@dataclass
class Resampler:
    in_samplerate: int
    out_samplerate: int
    lowpass_ratio: float = 0.05

    def __post_init__(self):
        if self.in_samplerate <= 0 or self.out_samplerate <= 0:
            raise ValueError("Sample rates must be positive integers")
        if self.in_samplerate == self.out_samplerate:
            raise ValueError("Input samplerate is equal to output samplerate")
        self.P, self.Q = get_P_and_Q(self.in_samplerate, self.out_samplerate)

    def resample(self, x: np.ndarray):
        original_shape = x.shape
        Nin = original_shape[-1]

        x_2d = x.reshape(-1, Nin)
        num_rows = x_2d.shape[0]

        M = int(2 * np.ceil((Nin + self.in_samplerate * 0.5) / (self.Q * 2)))

        Nx = M * self.Q
        Ny = M * self.P

        nbins_in = Nx // 2 + 1
        nbins_out = Ny // 2 + 1

        x_fft = np.fft.rfft(x_2d, axis=1, n=Nx)
        ## TAPERING
        if self.out_samplerate > self.in_samplerate:
            Ntaper = int(self.lowpass_ratio * nbins_in)
            taper = get_taper(Ntaper).reshape(1, Ntaper)
            x_fft[:, -Ntaper:] *= taper
            ## Zero-padding
            y_fft = np.zeros((num_rows, nbins_out), dtype=np.complex128)
            y_fft[:, :nbins_in] = x_fft
        else:
            y_fft = np.zeros((num_rows, nbins_out), dtype=np.complex128)
            y_fft[:, :nbins_out] = x_fft[:, :nbins_out]
            y_fft[:, -1] = np.real(y_fft[:, -1])

            Ntaper = int(self.lowpass_ratio * nbins_out)
            taper = get_taper(Ntaper).reshape(1, Ntaper)
            y_fft[:, -Ntaper:] *= taper

        # IFFT
        y = np.fft.irfft(y_fft, axis=1, n=Ny)

        # Scaling
        y = y * self.P / self.Q

        # Remove additional zeros and restore original leading dimensions
        Nout = int(Nin * self.P / self.Q)
        y = y[:, :Nout]
        return y.reshape(original_shape[:-1] + (Nout,))

    def __call__(self, x):
        return self.resample(x)


def resample(
    x: np.ndarray, in_samplerate: int, out_samplerate: int, lowpass_ratio: float = 0.05
):
    """
    Function to resample

    args:
    -----
     - `x` np.ndarray, shape (..., Nsamples): signal to resample; resampling is applied along the last axis
     - `in_samplerate` input samplerate
     - `out_samplerate` output samplerate
     - `lowpass_ratio` ratio of frequencies lowpassed. The lower the value, the higher the lowpass frequency
    """
    RS = Resampler(in_samplerate, out_samplerate, lowpass_ratio)
    y = RS(x)
    return y
