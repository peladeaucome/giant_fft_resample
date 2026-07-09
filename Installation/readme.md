# PyEvalAudio

This repository is a Python (Numpy) implementation of the paper "Giant FFTs for sample-rate conversion" by Vesa Välimäki and Stefan Bilbao.

## Installation

To install, just use
```
pip install git+https://github.com/peladeaucome/giant_fft_resample.git#subdirectory=Installation
```

## How to use

Import `GiantFFTResample`.
The sample rate converter can be found both in object and function forms.

The arguments are input signal (array of size (channels, Nsamples)), input samplerate, output samplerate and lowpass ratio (the amount of frequency information lost due to the lowpass filter).


```Python
from GiantFFTResample import Resampler, resample

# Object form:
rs = Resampler(in_samplerate=in_sr, out_samplerate=out_sr, lowpass_ratio=0.05)
y = rs(x)

# Functional form:
y = resample(x, in_samplerate=in_sr, out_samplerate=out_sr, lowpass_ratio=0.05)
```


# Cite this work:

This work is an implementation of the following article:

```LaTeX
@article{valimakiGiantFftsSamplerate2023,
  title = {Giant Ffts for Sample-Rate Conversion},
  author = {Välimäki, Vesa and Bilbao, Stefan},
  date = {2023-03-10},
  journaltitle = {JAES},
  volume = {71},
  number = {3},
  pages = {88--99},
  issn = {15494950},
  doi = {10.17743/jaes.2022.0061},
  url = {https://www.aes.org/e-lib/browse.cfm?elib=22033},
}

```
