"""
com_mat_export.py — engineering .mat snapshot of a COM run, for analysis in R.

This is an *additive* debug export. It does NOT change any COM calculation,
numerical algorithm, report, or figure. It captures the live pipeline objects
that the main COM flow has already computed for one package-test-case and writes
them to a MATLAB v5 .mat file readable by R.matlab::readMat().

Enabled by the optional `--export-mat` command-line flag (OP.EXPORT_MAT).

Provenance of the frequency-domain equalizer chain (see meta.notes in the file):
  * H_channel / H_channel_raw      — chdata[0].sdd21 / sdd21_raw   (genuine, FD)
  * H_ctle                         — fom_result.H_ctf              (genuine, FD)
  * H_ch_ctle (channel+CTLE)       — fom_result.sdd21ctf           (genuine, FD)
  * H_ffe                          — evaluated from the *selected* Tx FFE taps
                                     using COM's own Tx_FFE_Filter convention
  * H_tx (channel+FFE), H_final    — products of the above (DFE is a time-domain
                                     feedback term and is intentionally excluded
                                     from the FD chain).
Time-domain stage responses (h_*, pulse_*) are the genuine per-stage arrays COM
attaches to chdata / fom_result; pulse_ctle is the boxcar (1-UI moving average)
of the genuine channel+CTLE impulse response, exactly as Apply_EQ forms it.
"""
# Copyright 2025 802-COM Authors (upstream MATLAB reference)
# Copyright 2026 Todd Bermensolo (Python port)
# SPDX-License-Identifier: BSD-3-Clause

import os
import sys
import datetime
import subprocess

import numpy as np
from scipy.io import savemat
from scipy.special import erfcinv

COM_VERSION = "com_ieee8023_4p14p0"


# ── value sanitising for savemat (no None / SimpleNamespace / callables) ─────
def _is_pdf(v):
    return hasattr(v, "x") and hasattr(v, "y") and not isinstance(v, dict)


def _san(v, depth=0):
    """Convert a value into something scipy.io.savemat accepts, or None to skip."""
    if v is None or depth > 6:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float, complex, np.integer, np.floating, np.complexfloating)):
        return v
    if isinstance(v, str):
        return v
    if isinstance(v, np.ndarray):
        if v.dtype == object:
            return None
        return v
    if _is_pdf(v):
        return {"x": np.asarray(v.x).ravel(), "y": np.asarray(v.y).ravel()}
    if isinstance(v, dict):
        out = {}
        for k, vv in v.items():
            s = _san(vv, depth + 1)
            if s is not None and isinstance(k, str) and k.isidentifier():
                out[_fix_key(k, out)] = s
        return out or None
    # SimpleNamespace-like
    if hasattr(v, "__dict__"):
        return _san(vars(v), depth + 1)
    if isinstance(v, (list, tuple)):
        if len(v) == 0:
            return np.array([])
        try:
            arr = np.asarray(v)
            if arr.dtype != object:
                return arr
        except Exception:
            pass
        return None
    return None


def _fix_key(k, existing):
    """MATLAB struct/var field names are limited to 31 chars; truncate + dedupe."""
    if len(k) <= 31 and k not in existing:
        return k
    base = k[:31]
    if base not in existing:
        return base
    i = 1
    while True:
        cand = base[:31 - len(str(i)) - 1] + "_" + str(i)
        if cand not in existing:
            return cand
        i += 1


def _add(d, name, value):
    s = _san(value)
    if s is not None:
        d[_fix_key(name, d)] = s


# ── frequency-domain equalizer chain ────────────────────────────────────────
def _tx_ffe_fd(taps, f, fb):
    """FFE frequency response, matching COM's Tx_FFE_Filter convention."""
    taps = np.asarray(taps, dtype=float).ravel()
    f = np.asarray(f, dtype=float).ravel()
    icur = int(np.argmax(taps))
    H = np.zeros(len(f), dtype=complex)
    for k, c in enumerate(taps):
        H += c * np.exp(-1j * 2 * np.pi * (k - icur) * f / fb)
    return H


def _fd_ctle(freq, f_z, f_p1, f_p2, kacdc_dB):
    """CTLE frequency response, matching COM's FD_CTLE (OptFom_Compute_CTLE)."""
    freq = np.asarray(freq, dtype=float)
    num = 10 ** (kacdc_dB / 20.0) + 1j * freq / f_z
    den = (1 + 1j * freq / f_p1) * (1 + 1j * freq / f_p2)
    return num / den


def _git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


# ── public entry ─────────────────────────────────────────────────────────────
def export_case_mat(OP, param, chdata, fom_result, Noise_Struct, PDF, CDF,
                    COM_SNR_Struct, output_args, case_i):
    """Write one package-test-case's engineering snapshot to <RESULT_DIR>/<run>_caseNN.mat."""
    ch = chdata[0]
    fb = float(param.fb)
    M = int(param.samples_per_ui)
    f = np.asarray(ch.faxis, dtype=float).ravel()

    d = {}

    # 3. Frequency axis + channel responses --------------------------------
    _add(d, "f_Hz", f)
    _add(d, "f_GHz", f / 1e9)
    H_channel = np.asarray(ch.sdd21).ravel()
    _add(d, "H_channel", H_channel)
    _add(d, "H_channel_raw", np.asarray(ch.sdd21_raw).ravel())
    for nm in ("sdd11", "sdd11_raw", "sdd22", "sdd22_raw"):
        if hasattr(ch, nm):
            _add(d, "RL_" + nm, np.asarray(getattr(ch, nm)).ravel())

    # genuine CTLE FD (captured for the winning setting)
    H_ctle = getattr(fom_result, "H_ctf", None)
    H_ch_ctle = getattr(fom_result, "sdd21ctf", None)
    if H_ctle is not None:
        H_ctle = np.asarray(H_ctle).ravel()
        _add(d, "H_ctle", H_ctle)
    if H_ch_ctle is not None:
        H_ch_ctle = np.asarray(H_ch_ctle).ravel()
        _add(d, "H_ch_ctle", H_ch_ctle)

    # FFE FD (evaluated from selected taps) + derived cumulative chain
    txffe = np.asarray(fom_result.txffe, dtype=float).ravel()
    H_ffe = _tx_ffe_fd(txffe, f, fb)
    _add(d, "H_ffe", H_ffe)
    _add(d, "H_tx", H_channel * H_ffe)                       # channel + FFE
    if H_ch_ctle is not None and len(H_ch_ctle) == len(H_ffe):
        H_final = H_ch_ctle * H_ffe                          # channel + FFE + CTLE
        rx = getattr(fom_result, "RxFFE", None)
        if rx is not None and np.ndim(rx) >= 1 and len(np.asarray(rx).ravel()) > 1:
            H_final = H_final * _tx_ffe_fd(np.asarray(rx).ravel(), f, fb)
        _add(d, "H_final", H_final)

    # Rx bandwidth-limiting filter (Butterworth x Bessel-Thomson x Raised-Cosine)
    # and CTLE cascaded with it. Genuine reference filter (OptFom_Calc_Hr).
    try:
        import sicopr
        H_rx_filter = np.asarray(sicopr.OptFom_Calc_Hr(f, param, OP)).ravel()
        _add(d, "H_rx_filter", H_rx_filter)
        if H_ctle is not None and len(H_ctle) == len(H_rx_filter):
            _add(d, "H_ctle_rx", H_ctle * H_rx_filter)       # CTLE + Rx filter
    except Exception as _e:
        print("    [export-mat] Rx filter skipped: %s" % _e)

    # crosstalk paths (if present)
    H_next = [np.asarray(c.sdd21).ravel() for c in chdata if str(getattr(c, "type", "")) == "NEXT"]
    H_fext = [np.asarray(c.sdd21).ravel() for c in chdata if str(getattr(c, "type", "")) == "FEXT"]
    if H_next:
        _add(d, "H_next", np.array(H_next))
    if H_fext:
        _add(d, "H_fext", np.array(H_fext))

    # 4. Component responses ------------------------------------------------
    _add(d, "ffe_taps", txffe)
    _add(d, "ffe_cursor_index", int(np.argmax(txffe)) + 1)   # 1-based for R
    _add(d, "dfe_taps", np.asarray(fom_result.DFE_taps, dtype=float).ravel())
    if hasattr(fom_result, "DFE_taps_i"):
        _add(d, "dfe_tap_locations", np.asarray(fom_result.DFE_taps_i).ravel())
    if getattr(fom_result, "RxFFE", None) is not None:
        _add(d, "rxffe_taps", np.asarray(fom_result.RxFFE).ravel())
    _add(d, "ctle_gain_db", getattr(output_args, "CTLE_DC_gain_dB", None))
    _add(d, "ctle_index", int(fom_result.ctle))
    ci = int(fom_result.ctle) - 1
    ctle_par = {
        "CTLE_type": str(param.CTLE_type),
        "fz_Hz": float(np.asarray(param.CTLE_fz).ravel()[ci]),
        "fp1_Hz": float(np.asarray(param.CTLE_fp1).ravel()[ci]),
        "fp2_Hz": float(np.asarray(param.CTLE_fp2).ravel()[ci]),
        "gdc_dB": float(np.asarray(param.ctle_gdc_values).ravel()[ci]),
        "g_DC_HP": getattr(output_args, "g_DC_HP", None),
    }
    _add(d, "ctle_parameters", ctle_par)

    # full CTLE bank: every available DC-gain setting (no high-pass stage), so R
    # can plot all selectable CTLE curves. Computed from param via FD_CTLE.
    try:
        gdc_all = np.asarray(param.ctle_gdc_values, dtype=float).ravel()
        fz_all = np.asarray(param.CTLE_fz, dtype=float).ravel()
        fp1_all = np.asarray(param.CTLE_fp1, dtype=float).ravel()
        fp2_all = np.asarray(param.CTLE_fp2, dtype=float).ravel()
        n_ctle = len(gdc_all)
        idx = lambda a, i: a[i] if len(a) > i else a[-1]   # broadcast if scalar
        H_ctle_all = np.array([
            _fd_ctle(f, idx(fz_all, i), idx(fp1_all, i), idx(fp2_all, i), gdc_all[i])
            for i in range(n_ctle)])
        _add(d, "H_ctle_all", H_ctle_all)                  # (n_ctle x n_freq)
        _add(d, "ctle_gdc_values", gdc_all)
    except Exception as _e:
        print("    [export-mat] CTLE bank skipped: %s" % _e)

    # 5/6. Time axis + impulse + pulse responses ---------------------------
    t = np.asarray(ch.t, dtype=float).ravel()
    _add(d, "t_s", t)
    _add(d, "t_ns", t * 1e9)
    _add(d, "h_channel", np.asarray(ch.uneq_imp_response, dtype=float).ravel())
    if hasattr(ch, "ctle_imp_response"):
        h_ctle = np.asarray(ch.ctle_imp_response, dtype=float).ravel()
        _add(d, "h_ctle", h_ctle)                            # channel + CTLE (genuine)
    if hasattr(fom_result, "IR"):
        _add(d, "h_final", np.asarray(fom_result.IR, dtype=float).ravel())

    _add(d, "pulse_channel", np.asarray(ch.uneq_pulse_response, dtype=float).ravel())
    if hasattr(ch, "eq_imp_response"):
        from scipy.signal import lfilter
        eq_ir = np.asarray(ch.eq_imp_response, dtype=float).ravel()
        _add(d, "pulse_ctle", lfilter(np.ones(M), [1.0], eq_ir))   # channel + CTLE pulse
    if hasattr(ch, "ctle_pulse"):
        _add(d, "pulse_ctle_ffe", np.asarray(ch.ctle_pulse, dtype=float).ravel())
    _add(d, "pulse_final", np.asarray(ch.eq_pulse_response, dtype=float).ravel())
    _add(d, "sbr", np.asarray(fom_result.sbr, dtype=float).ravel())
    _add(d, "samples_per_ui", M)
    _add(d, "ui_seconds", float(param.ui))
    _add(d, "baud_rate", fb)
    _add(d, "cursor_sample_index", int(fom_result.t_s) + 1)  # 1-based for R

    # 7. Eye / bathtub data -------------------------------------------------
    tb = getattr(ch, "timing_bathtub", None)
    if isinstance(tb, dict):
        eye = {}
        if "phase_UI" in tb:
            eye["phase_UI"] = np.asarray(tb["phase_UI"]).ravel()
        if "ber_eyes" in tb:
            eye["ber_eyes"] = np.asarray(tb["ber_eyes"])
        if "eye_contour" in tb:
            eye["eye_contour"] = np.asarray(tb["eye_contour"])
        # Voltage-bathtub sweep: one BER curve per eye over decision threshold,
        # taken at the centre phase. Distinct from the timing bathtub, which
        # sweeps phase at each eye's own fixed threshold.
        for _k in ("eye_threshold_V", "vbt_threshold_V", "vbt_ber"):
            if _k in tb:
                eye[_k] = np.asarray(tb[_k])
        _add(d, "eye", eye)
    ec = getattr(COM_SNR_Struct, "eye_contour", None)
    if ec is not None and len(np.asarray(ec, dtype=object).ravel()) > 0:
        try:
            _add(d, "eye_contour_csnr", np.asarray(ec, dtype=float))
        except Exception:
            pass

    # 8. Combined PDF / CDF + noise components ------------------------------
    _add(d, "pdf_x", np.asarray(PDF.x, dtype=float).ravel())
    _add(d, "pdf_y", np.asarray(PDF.y, dtype=float).ravel())
    _add(d, "cdf", np.asarray(CDF, dtype=float).ravel())
    noise = {}
    for attr, key in (("noise_pdf", "noise"), ("isi_and_xtalk_pdf", "isi_xtalk"),
                      ("sci_pdf", "isi"), ("cci_pdf", "crosstalk")):
        p = getattr(Noise_Struct, attr, None)
        if _is_pdf(p):
            noise[key] = {"x": np.asarray(p.x).ravel(), "y": np.asarray(p.y).ravel()}
    if noise:
        _add(d, "noise_pdfs", noise)

    # ── stage 2: TDR / ERL ──────────────────────────────────────────────────
    # The PNG set has drawn these since 2026-08-28; the interactive report could
    # not, because none of it was exported. chdata[0].TDR11[izt] carries .ZSR
    # (the impedance profile), .t, .avgZport and the ERL scalars.
    #
    # Z_t is the TDR target and is SINGLE-ENDED (it defaults to R_0), while ZSR
    # is differential -- so a reference line belongs at 2*Z_t. Exported as
    # tdr_Z_ref already doubled, so no consumer has to rediscover that.
    tdr = {}
    for nm in ("TDR11", "TDR22"):
        lst = getattr(ch, nm, None)
        if not lst:
            continue
        e = lst[0] if isinstance(lst, (list, tuple)) else lst
        zsr = np.asarray(getattr(e, "ZSR", []), dtype=float).ravel()
        tt = np.asarray(getattr(e, "t", []), dtype=float).ravel()
        if zsr.size and tt.size == zsr.size:
            tdr[nm + "_t_ns"] = tt * 1e9
            tdr[nm + "_Z_ohm"] = zsr
            tdr[nm + "_avgZ_ohm"] = float(getattr(e, "avgZport", float("nan")))
        for fld in ("ERL", "ERL_CD", "ERL_DC", "ERL_CC"):
            v = getattr(e, fld, None)
            v = np.asarray(v, dtype=float).ravel() if v is not None else np.array([])
            if v.size == 1 and np.isfinite(v[0]):
                tdr[nm.replace("TDR", "ERL") + fld[3:]] = float(v[0])
    if tdr:
        zt = np.asarray(getattr(param, "Z_t", getattr(param, "Z0", 50.0)),
                        dtype=float).ravel()
        if zt.size:
            tdr["Z_ref_ohm"] = 2.0 * float(zt[0])
        _add(d, "tdr", tdr)

    # ── stage 4: FOM against sampling phase ─────────────────────────────────
    # optimize_fom builds FOM_TRACKER indexed [gffe, ctle, g_DC_HP, txffe, itick]
    # and exports it with its sample_range axis. Reduced here to the best FOM
    # found at each phase: the full 5-D array would bloat the .mat for no gain,
    # and best-per-phase is the question anyone actually asks of it -- how sharp
    # is the sampling optimum.
    ft = getattr(fom_result, "FOM_TRACKER", None)
    rng = getattr(fom_result, "sample_range", None)
    if ft is not None and rng is not None:
        ft = np.asarray(ft, dtype=float)
        rng = np.asarray(rng, dtype=float).ravel()
        if ft.ndim == 5 and ft.shape[-1] == rng.size and rng.size > 1:
            flat = ft.reshape(-1, ft.shape[-1]).copy()
            flat[flat == 0.0] = np.nan          # 0 marks a phase never evaluated
            if not np.all(np.isnan(flat)):
                best = np.nanmax(flat, axis=0)
                _add(d, "fom_vs_phase", {
                    "itick": rng,
                    "FOM_dB": best,
                    "selected_itick": float(getattr(fom_result, "itick", float("nan"))),
                })

    # ── stage 6: the individual noise terms ─────────────────────────────────
    # sigma_total_mV was the only noise scalar exported, which cannot say whether
    # the transmitter or the channel dominates.
    terms = {}
    for attr in ("sigma_TX", "sigma_G", "sigma_N", "sigma_rjit", "sigma_Q",
                 "sigma_hp", "cci_sigma", "sci_sigma", "sigma_before_clip",
                 "peak_clip"):
        v = getattr(Noise_Struct, attr, None)
        v = np.asarray(v, dtype=float).ravel() if v is not None else np.array([])
        if v.size == 1 and np.isfinite(v[0]):
            terms[attr + "_mV"] = 1000.0 * float(v[0])
    if terms:
        _add(d, "noise_terms", terms)

    # COM results (headline + full self-describing output_args) -------------
    A_s = float(COM_SNR_Struct.A_s)
    A_ni = float(COM_SNR_Struct.A_ni)
    COM_dB = getattr(COM_SNR_Struct, "COM", None)
    _add(d, "COM_dB", COM_dB)
    _add(d, "VEC_dB", getattr(COM_SNR_Struct, "VEC_dB", None))
    _add(d, "VEO_mV", getattr(COM_SNR_Struct, "VEO_mV", None))
    _add(d, "A_s_mV", 1000.0 * A_s)
    _add(d, "A_ni_mV", 1000.0 * A_ni)
    _add(d, "EW_UI", getattr(COM_SNR_Struct, "EW_UI", None))
    _add(d, "threshold_DER", getattr(COM_SNR_Struct, "threshold_DER", None))

    # Simplified "Gaussian COM" FOM -----------------------------------------
    # COM itself reads A_ni off the *statistical* combined interference+noise
    # CDF at specBER. FOM_gauss_dB is the COM you would get if that distribution
    # were purely Gaussian: A_ni_gauss = Q^-1(specBER) * sigma_total, where
    # sigma_total is the RSS sigma of COM's own combined PDF. Same signal
    # numerator COM uses (2*A_s for C2M, else A_s), so it differs from COM_dB
    # only by replacing the statistical tail with a Gaussian one.
    px = np.asarray(PDF.x, dtype=float).ravel()
    py = np.asarray(PDF.y, dtype=float).ravel()
    mu = float(np.sum(px * py))
    sigma_total = float(np.sqrt(max(np.sum((px - mu) ** 2 * py), 0.0)))
    ber = float(param.specBER)
    q_inv = float(np.sqrt(2.0) * erfcinv(2.0 * ber))     # one-sided Gaussian quantile
    eps = np.finfo(float).eps
    A_ni_gauss = q_inv * sigma_total
    numer = (2.0 * A_s) if float(getattr(param, "T_O", 0)) != 0 else A_s
    FOM_gauss_dB = 20.0 * np.log10(numer / max(A_ni_gauss, eps))
    _add(d, "FOM_gauss_dB", FOM_gauss_dB)
    _add(d, "sigma_total_mV", 1000.0 * sigma_total)
    _add(d, "A_ni_gauss_mV", 1000.0 * A_ni_gauss)
    # full results container (skip the big BREAD_CRUMBS object refs)
    skip = {"OP", "param", "chdata", "fom_result", "PDF", "CDF",
            "MLSE_results", "PSD_results", "eye_contour"}
    results_full = {k: v for k, v in vars(output_args).items() if k not in skip}
    _add(d, "results_full", results_full)

    # 9. Configuration ------------------------------------------------------
    cfg = {
        "baud_rate": fb,
        "sample_rate": fb * M,
        "samples_per_ui": M,
        "ui_seconds": float(param.ui),
        "levels": int(param.levels),
        "specBER": float(param.specBER),
        "ndfe": int(param.ndfe),
        "ffe_taps": txffe,
        "ctle_settings": ctle_par,
        "dfe_settings": {"ndfe": int(param.ndfe),
                         "dfe_taps": np.asarray(fom_result.DFE_taps, dtype=float).ravel()},
        "victim_channel": str(getattr(ch, "base", "")),
        "aggressor_channels": [str(getattr(c, "base", "")) for c in chdata[1:]],
        "package_testcase": int(case_i),
    }
    # generic shallow dump of scalar/array param fields (self-describing)
    pdump = {}
    for k, v in vars(param).items():
        if not k.isidentifier():
            continue
        if isinstance(v, (int, float, str, np.integer, np.floating)):
            pdump[k] = v
        elif isinstance(v, np.ndarray) and v.dtype != object:
            pdump[k] = v
    cfg["param"] = pdump
    _add(d, "config", cfg)

    # Run summary: what was swept, how many cases, wall-clock time ----------
    try:
        n_cases = len(np.asarray(OP.pkg_len_select).ravel())
    except Exception:
        n_cases = 1
    import time as _time
    # Per-case run time: wall-clock since the start of *this* case (set at the
    # top of the package loop). Falls back to the whole-run start if unavailable.
    elapsed_case_s = None
    case_t_start = getattr(OP, "export_case_t_start", None)
    if case_t_start is not None:
        elapsed_case_s = float(_time.time() - case_t_start)
    elapsed_total_s = None
    t_start = getattr(OP, "export_t_start", None)
    if t_start is not None:
        elapsed_total_s = float(_time.time() - t_start)
    sd = getattr(fom_result, "sweep_dims", None)
    run_summary = {
        "n_cases": int(n_cases),
        "package_testcase": int(case_i),
        "elapsed_s_this_case": elapsed_case_s,
        "elapsed_s_through_this_case": elapsed_total_s,
    }
    if sd is not None:
        run_summary["sweep"] = {
            "n_gffe": int(sd.n_gffe), "n_CTLE": int(sd.n_ctle),
            "n_g_DC_HP": int(sd.n_g_DC_HP), "n_TXFFE": int(sd.n_txffe),
            "n_itick": int(sd.n_itick), "total_evals": int(sd.total_evals),
            "local_search": int(sd.local_search),
            "description": ("%d gffe x %d CTLE x %d g_DC_HP x %d TXFFE x %d itick "
                            "= %d evals" % (sd.n_gffe, sd.n_ctle, sd.n_g_DC_HP,
                                            sd.n_txffe, sd.n_itick, sd.total_evals)),
        }
    _add(d, "run_summary", run_summary)

    # 10. Metadata ----------------------------------------------------------
    meta = {
        "timestamp": datetime.datetime.now().isoformat(),
        "com_version": COM_VERSION,
        "python_version": sys.version.split()[0],
        "git_commit": _git_commit(),
        "input_s4p": list(getattr(OP, "export_s4p_files", [])),
        "input_config": str(getattr(OP, "export_config_file", "")),
        "case_index": int(case_i),
        "notes": ("H_channel/H_ctle/H_ch_ctle are genuine FD responses captured "
                  "from the pipeline; H_ffe is evaluated from the selected Tx FFE "
                  "taps; H_tx/H_final are products of these (DFE excluded from FD). "
                  "COM is a statistical tool: the eye is a BER contour, not a "
                  "sampled-waveform eye matrix."),
    }
    _add(d, "meta", meta)

    # write -----------------------------------------------------------------
    run = str(getattr(OP, "export_run_name", "com_run")) or "com_run"
    out_dir = str(getattr(OP, "RESULT_DIR", "") or "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "%s_case%02d.mat" % (run, int(case_i)))
    savemat(out_path, d, do_compression=True, oned_as="column")
    print("    [export-mat] %d variables -> %s" % (len(d), out_path))
    return out_path
