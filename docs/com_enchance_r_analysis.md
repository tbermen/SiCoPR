# COM Python Enhancement for R Analysis and Visualization

## Objective

The existing Python implementation must continue to reproduce IEEE COM behavior and outputs exactly.

In addition, create a secondary engineering-analysis output path that exports intermediate simulation data to a MATLAB-compatible `.mat` file for use in R.

The goal is NOT to modify COM calculations.

The goal is to expose internal variables and intermediate results that already exist during execution so they can be analyzed and visualized externally.

---

# Design Requirements

## 1. Preserve Existing Behavior

Do not change:

* COM calculations
* Numerical algorithms
* Output reports
* Existing plots
* Existing command-line arguments

The standard COM flow must continue to produce identical results.

All new functionality must be optional.

Example:

```python
sicopr.py config.xlsx 0 0 channel.s4p --export-mat
```

or

```python
export_mat = True
```

---

# 2. Create Engineering Export File

At the end of execution create:

```text
run_name.mat
```

using:

```python
scipy.io.savemat()
```

The MAT file must be readable by:

```R
R.matlab::readMat()
```

Store variables using simple arrays and structures.

Avoid Python-specific objects.

---

# 3. Export Frequency-Domain Data

Export all frequency-domain vectors that exist during COM execution.

## Frequency Axis

```python
f_Hz
f_GHz
```

---

## Channel Responses

Raw channel only:

```python
H_channel
```

After Tx equalization:

```python
H_tx
```

After CTLE:

```python
H_ctle
```

After DFE (if applicable):

```python
H_dfe
```

Final equalized response:

```python
H_final
```

Complex frequency responses should remain complex.

Do not convert to magnitude-only.

---

# 4. Export Component Responses Individually

Export transfer functions of individual blocks.

## TX FFE

```python
H_ffe
ffe_taps
```

## CTLE

```python
H_ctle
ctle_gain_db
ctle_parameters
```

## DFE

```python
dfe_taps
```

## Crosstalk Paths

If available:

```python
H_next
H_fext
```

---

# 5. Export Impulse Responses

Export time-domain impulse responses for all major stages.

## Time Axis

```python
t_s
t_ns
```

---

## Impulse Responses

```python
h_channel
h_tx
h_ctle
h_final
```

These are among the most useful outputs for debugging.

---

# 6. Export Pulse Responses

Store pulse responses at each stage.

```python
pulse_channel
pulse_tx
pulse_ctle
pulse_final
```

Include:

```python
samples_per_ui
ui_seconds
baud_rate
```

---

# 7. Export Eye-Diagram Data

Store the actual waveform used to generate the eye.

Do NOT only save rendered images.

Export:

```python
eye_waveform
eye_time
```

If COM internally builds a 2-UI or multi-UI eye matrix:

```python
eye_matrix
```

Export it directly.

This allows eye reconstruction in R.

---

# 8. Export COM Results

Export final metrics.

Examples:

```python
COM_dB
ERL_dB
IL_dB
SNR_dB
sigma_total
sigma_jitter
sigma_noise
```

Also export any intermediate metrics used during COM calculation.

---

# 9. Export Configuration Parameters

Store all settings used for the run.

Examples:

```python
config
```

containing:

```python
baud_rate
sample_rate
ffe_taps
ctle_settings
dfe_settings
victim_channel
aggressor_channels
```

This makes each MAT file self-describing.

---

# 10. Export Metadata

Store:

```python
timestamp
com_version
python_version
git_commit
input_s4p
input_config
```

---

# R Visualization Requirements

The exported MAT file must support generation of the following interactive plots in R.

---

## Plot 1: Channel Frequency Response

Interactive Plotly plot.

Curves:

* Channel only
* Channel + Tx FFE
* Channel + Tx FFE + CTLE
* Final equalized response

Display:

* Magnitude (dB)
* Frequency (GHz)

Legend toggles enabled.

---

## Plot 2: CTLE Response

Interactive Plotly plot.

Display:

* CTLE gain (dB)
* Frequency (GHz)

Hover tooltip showing:

* Frequency
* Gain

---

## Plot 3: Impulse Responses

Interactive Plotly plot.

Curves:

* Channel
* Tx EQ
* CTLE
* Final

Display:

* Amplitude
* Time (ns)

Zoom enabled.

---

## Plot 4: Pulse Responses

Interactive Plotly plot.

Curves:

* Channel
* Tx EQ
* CTLE
* Final

Mark:

* Cursor UI
* Sampling point

---

## Plot 5: Eye Diagram

Interactive Plotly eye diagram.

Capabilities:

* Zoom
* Hover coordinates
* Density visualization

If eye_matrix exists, use it directly.

---

## Plot 6: Equalizer Contribution Plot

Show cumulative impact of equalization.

Curves:

1. Channel
2. Channel + FFE
3. Channel + FFE + CTLE
4. Final response

This plot is particularly important for debugging COM behavior.

---

## Plot 7: COM Dashboard

Create a single HTML report containing:

* COM result
* CTLE settings
* DFE settings
* Frequency response plot
* Impulse response plot
* Pulse response plot
* Eye diagram

Output:

```text
run_name_report.html
```

using Plotly.

---

# Implementation Strategy

When adding exports:

1. Do not recalculate data.
2. Capture variables already computed by COM.
3. Export them immediately after creation.
4. Maintain numerical fidelity.
5. Preserve complex-valued data where available.

The exported MAT file should be considered a complete engineering debug snapshot of a COM run.
