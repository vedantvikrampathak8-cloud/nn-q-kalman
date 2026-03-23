# \# NN-Q Adaptive Kalman Filter for ADC0804

# 

# 

# \\

# 

# \*\*Neural network–adaptive process noise (Q) for a 1-state Kalman filter applied to low-resolution ADC signals.\*\*

# Pure NumPy · 288 weights · 264 MACs/sample · No external ML frameworks

# 

# \---

# 

# \## Overview

# 

# A standard 8-bit ADC (ADC0804, VREF = 5 V) has a quantization noise floor of:

# 

# ```

# √(LSB² / 12) ≈ 5.64 mV

# ```

# 

# This project demonstrates how a combination of \*\*oversampling + adaptive Kalman filtering\*\* can reduce effective noise and improve ENOB.

# 

# \### Pipeline

# 

# ```

# 10 kSPS raw ADC

# &#x20;     │

# &#x20;     ▼  4× Moving Average (Decimation)

# 2.5 kSPS  →  noise: 5.64 → 2.82 mV  (+1.00 bit)

# &#x20;     │

# &#x20;     ▼  NN-Q Adaptive Kalman (1-state)

# 2.5 kSPS  →  noise: \~2.82 → \~1.93 mV  (\~+0.5 bit typical)

# &#x20;     │

# &#x20;     ▼  Output

# ```

# 

# \*\*Measured improvement (real ADC data): \~+1.5 bits ENOB\*\*

# \*\*Theoretical ceiling: \~+2.7 bits (signal-dependent)\*\*

# 

# \---

# 

# \## Key Idea

# 

# Instead of using a fixed process noise (Q), this work:

# 

# \* learns Q dynamically using a \*\*tiny neural network\*\*

# \* uses \*\*innovation history (last 8 samples)\*\* as input

# \* adapts filtering strength in real time

# 

# Result:

# 

# \* Low noise during steady regions

# \* Fast response during signal changes

# 

# \---

# 

# \## Why 1-State Kalman?

# 

# A 2-state (position + velocity) model introduces:

# 

# ```

# DT² · P\[1,1] → noise leakage into position estimate

# ```

# 

# The 1-state model avoids this entirely:

# 

# \* simpler

# \* more stable

# \* predictable steady-state behavior

# 

# \---

# 

# \## Results (Real ADC Data)

# 

# | Stage       | Noise (RMS)  | ENOB Gain      |

# | ----------- | ------------ | -------------- |

# | Raw ADC     | 5.64 mV      | baseline       |

# | 4× MA       | 2.82 mV      | +1.00 bit      |

# | NN-Q Kalman | \*\*\~1.93 mV\*\* | \*\*+0.5 bit\*\*   |

# | \*\*Total\*\*   | —            | \*\*\~+1.5 bits\*\* |

# 

# Additional observations:

# 

# \* Correlation with input: \*\*0.999+\*\*

# \* No visible signal distortion

# \* Adaptive Q spans multiple orders of magnitude

# 

# \---

# 

# \## ENOB Validation (Sine Test)

# 

# Using IEEE-style sine-fit:

# 

# \* Input ENOB: \~7.1 bits

# \* Output ENOB: \~9.8 bits

# \* Improvement: \*\*\~+2.7 bits (near theoretical limit)\*\*

# 

# Note:

# This represents \*\*best-case performance\*\* under controlled conditions.

# 

# \---

# 

# \## Neural Network

# 

# Architecture:

# 

# ```

# 8 (innovation history)

# &#x20;→ 16 (tanh)

# &#x20;→ 8 (tanh)

# &#x20;→ 1 (softplus + clamp)

# ```

# 

# \* Parameters: \*\*288 weights\*\*

# \* Compute: \*\*264 MACs/sample\*\*

# \* Framework: \*\*pure NumPy\*\*

# 

# \---

# 

# \## Training Strategy

# 

# Two-stage approach:

# 

# 1\. \*\*Pool Training\*\*

# 

# &#x20;  \* Gyro signals (drift + motion)

# &#x20;  \* ECG signals (spikes + quiet regions)

# &#x20;  \* Synthetic oscillations (critical for frequency coverage)

# 

# 2\. \*\*Fine-tuning\*\*

# 

# &#x20;  \* Adapt Q scaling to the target signal

# 

# 3\. \*\*Calibration\*\*

# 

# &#x20;  \* Ensures Q reaches minimum during quiet periods

# 

# \---

# 

# \## Installation

# 

# ```bash

# git clone https://github.com/vedantvikrampathak8-cloud/nn-q-kalman.git

# cd nn-q-kalman

# pip install numpy scipy matplotlib

# ```

# 

# \---

# 

# \## Usage

# 

# ```bash

# \# Run on ADC CSV

# python nn\_kalman\_1state.py yourfile.csv

# 

# \# Raw ADC values

# python nn\_kalman\_1state.py yourfile.txt

# 

# \# ECG data

# python nn\_kalman\_1state.py file.mat

# 

# \# Synthetic demo

# python nn\_kalman\_1state.py

# 

# \# ENOB validation

# python nn\_kalman\_1state.py --test-enob

# ```

# 

# \---

# 

# \## Output

# 

# Each run generates:

# 

# \* `\*\_dashboard.png` → diagnostics (Q, P, residuals, etc.)

# \* `\*\_closeup.png` → zoomed signal view

# \* `\*\_output.csv` → processed data

# 

# \---

# 

# \## Interpreting Results

# 

# \* \*\*Q trace:\*\* should rise during transitions, fall in steady regions

# \* \*\*Variance P:\*\* should drop below measurement noise in quiet regions

# \* \*\*Innovations:\*\* tighter distribution = better filtering

# \* \*\*Residuals:\*\* NN-Q may deviate more from noisy input (expected)

# 

# \---

# 

# \## Limitations

# 

# \* Moving average reduces high-frequency content

# \* Kalman introduces small phase lag

# \* Performance depends on signal characteristics

# \* Training distribution affects generalization

# 

# \---

# 

# \## Contribution / Citation

# 

# If used in research:

# 

# ```bibtex

# @software{nn\_q\_kalman\_2026,

# &#x20; author = {Vedant},

# &#x20; title  = {NN-Q Adaptive Kalman Filter},

# &#x20; year   = {2026},

# &#x20; url    = {https://github.com/vedantvikrampathak8-cloud/nn-q-kalman}

# }

# ```

# 

# \---

# 

# \## License

# 

# Apache 2.0



