"""
com_plots.py — standard COM figure set (live-data, MATLAB-equivalent tool output).

The production plotting layer for the Python COM tool. Each package-test-case
calls save_case_figures() with the live pipeline objects (chdata, fom_result,
Noise_Struct, PDF/CDF, COM_SNR_Struct); the figures are written straight to the
case's results directory. Controlled by OP.SAVE_FIGURES (read from the config),
NOT tied to any checkpoint/debug system.

Mirrors MATLAB's per-case figure output (eye, bathtubs, IL, SBR, PDFs).
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _db(s):
    return 20.0 * np.log10(np.abs(np.asarray(s)) + 1e-300)


def _match_created_to_modified(path):
    """Set a file's creation time equal to its last-modified time (Windows only).

    These figures are regenerated artifacts, so an unambiguous "Date" in Explorer
    is desirable. Plain delete+rewrite is not enough: NTFS "file tunneling"
    restores the previous creation time when a same-named file is recreated within
    ~15 s, leaving Date Created stale. Best-effort; silent no-op off Windows.
    """
    if os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes
        ticks = int(round(os.path.getmtime(path) * 1e7)) + 116444736000000000
        ft = wintypes.FILETIME(ticks & 0xFFFFFFFF, (ticks >> 32) & 0xFFFFFFFF)
        k32 = ctypes.windll.kernel32
        k32.CreateFileW.restype = wintypes.HANDLE
        handle = k32.CreateFileW(ctypes.c_wchar_p(path), 0x40000000, 0, None,
                                 3, 0x80, None)  # GENERIC_WRITE, OPEN_EXISTING
        if handle in (None, 0, ctypes.c_void_p(-1).value):
            return
        try:
            k32.SetFileTime(wintypes.HANDLE(handle), ctypes.byref(ft), None, None)
        finally:
            k32.CloseHandle(wintypes.HANDLE(handle))
    except Exception:
        pass


def _save(fig, outdir, name):
    fig.tight_layout()
    path = os.path.join(outdir, name)
    try:
        # Remove any existing file first (per file, by name) so the regenerated
        # figure is a genuinely new file; then align Date Created with Date
        # Modified so Explorer's per-type "Date" column is unambiguous.
        if os.path.exists(path):
            os.remove(path)
        fig.savefig(path, dpi=110)
    except (PermissionError, OSError) as e:
        plt.close(fig)
        raise RuntimeError(
            "could not write %s (%s). On Windows this usually means the PNG is "
            "open in an image viewer / Explorer preview / VS Code, which locks "
            "the file - close it and re-run." % (name, e))
    plt.close(fig)
    _match_created_to_modified(path)


def _band_ylim(ax, fG, curves, fmax, pad=6.0, floor_db=-90.0):
    """Scale y to the data actually on screen.

    Matplotlib autoscales over every plotted point, including the 60-100 GHz
    tail that the x-limit hides, so a deep out-of-band null stretches the axis
    to -175 dB and flattens the part being looked at.
    """
    band = (fG > 0) & (fG <= fmax)
    vals = np.concatenate([np.asarray(c)[band] for c in curves]) if band.any() else None
    if vals is None or not vals.size:
        return
    vals = vals[np.isfinite(vals)]
    if not vals.size:
        return
    ax.set_ylim(max(float(vals.min()) - pad, floor_db), float(vals.max()) + pad)


# ── individual figures (each guarded by the caller) ──────────────────────────
def _fig_sparams(outdir, chdata, param):
    ch = chdata[0]
    fG = np.asarray(ch.faxis, dtype=float).ravel() / 1e9
    fb = float(param.fb)
    fig, ax = plt.subplots(figsize=(9, 5))
    # Log frequency: the interesting structure (low-frequency loss slope, the
    # roll-off knee) is compressed into the first decade on a linear axis.
    pos = fG > 0                       # f = 0 cannot be shown on a log axis
    ax.plot(fG[pos], _db(ch.sdd21_raw)[pos], label="IL raw")
    ax.plot(fG[pos], _db(ch.sdd21)[pos], label="IL cascaded (+pkg/brd)")
    ax.axvline(fb/2/1e9, ls=":", color="grey")
    ax.set_xscale("log")
    ax.set_xlim(max(fG[pos][0], 1e-2), min(fG[-1], 60))
    _band_ylim(ax, fG, [_db(ch.sdd21_raw), _db(ch.sdd21)], min(fG[-1], 60))
    ax.set_xlabel("frequency [GHz]"); ax.set_ylabel("|SDD21| [dB]")
    ax.set_title("Insertion loss: raw vs cascaded"); ax.grid(True, ls=":", alpha=0.5); ax.legend(fontsize=8)
    _save(fig, outdir, "02_insertion_loss.png")
    fig, ax = plt.subplots(figsize=(9, 5))
    for arr, lab in [(ch.sdd11_raw, "RL11 raw"), (ch.sdd11, "RL11 cas"),
                     (ch.sdd22_raw, "RL22 raw"), (ch.sdd22, "RL22 cas")]:
        ax.plot(fG[pos], _db(arr)[pos], label=lab)
    ax.set_xscale("log")
    ax.set_xlim(max(fG[pos][0], 1e-2), min(fG[-1], 60))
    _band_ylim(ax, fG, [_db(ch.sdd11_raw), _db(ch.sdd11),
                        _db(ch.sdd22_raw), _db(ch.sdd22)], min(fG[-1], 60))
    ax.set_xlabel("frequency [GHz]"); ax.set_ylabel("[dB]")
    ax.set_title("Return loss: raw vs cascaded"); ax.grid(True, ls=":", alpha=0.5); ax.legend(fontsize=8)
    _save(fig, outdir, "02_return_loss.png")


def _fig_filters(outdir, chdata, param, OP):
    import com  # lazy: com is fully loaded by run time
    f = np.asarray(chdata[0].faxis, dtype=float).ravel()
    fG = f / 1e9
    H_r = com.OptFom_Calc_Hr(f, param, OP)                       # main path (no H_t)
    ttr = float(OP.transmitter_transition_time)
    H_t = np.exp(-(np.pi * f / 1e9 * ttr / 1.6832) ** 2)
    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.plot(fG, _db(H_r), lw=2, label="MAIN path H_r (no H_t)")
    ax.plot(fG, _db(H_r * H_t), lw=2, label="DISPLAY path H_r*H_t")
    ax.plot(fG, _db(H_t), ls="--", label="H_t (Gaussian)")
    ax.set_xscale("log")
    _fpos = fG[fG > 0]
    ax.set_xlim(max(_fpos[0], 1e-2) if _fpos.size else 1e-2, 60)
    # These filters are flat to ~20 GHz and then roll off, so a fixed -40..3 dB
    # window leaves most of the axis empty once x is logarithmic.
    _band_ylim(ax, fG, [_db(H_r), _db(H_r * H_t), _db(H_t)], 60, pad=2.0,
               floor_db=-40.0)
    ax.set_xlabel("frequency [GHz]"); ax.set_ylabel("[dB]")
    ax.set_title("FD response chain: main vs display (H_t is display-only)")
    ax.grid(True, ls=":", alpha=0.5); ax.legend(fontsize=8)
    _save(fig, outdir, "03_filters_main_vs_display.png")


def _fig_pulse(outdir, chdata, param):
    ch = chdata[0]
    PR = np.asarray(ch.uneq_pulse_response, dtype=float).ravel()
    t = np.asarray(ch.t, dtype=float).ravel()
    M = int(param.samples_per_ui); ui = float(param.ui); ipk = int(np.argmax(PR))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(t * 1e9, PR * 1000)
    ax.set_xlabel("time [ns]"); ax.set_ylabel("SBR [mV]"); ax.set_title("Single-bit response (unequalized)")
    ax.grid(True, ls=":", alpha=0.5)
    _save(fig, outdir, "04_sbr_full.png")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    # Same window as 06_eq_vs_uneq_sbr so the two charts can be read together;
    # the post-cursor tail the DFE works on runs well past +10 UI.
    _T0, _T1 = -5, 20
    tui = (t - t[ipk]) / ui; m = (tui >= _T0) & (tui <= _T1)
    ax.plot(tui[m], PR[m] * 1000, ".-", ms=3)
    for k in range(_T0, _T1 + 1): ax.axvline(k, ls=":", color="grey", lw=0.5)
    ax.axvline(0, color="red"); ax.set_xlim(_T0, _T1)
    ax.set_xlabel("time [UI from peak]"); ax.set_ylabel("SBR [mV]")
    ax.set_title("SBR zoom (-5 to +20 UI)"); ax.grid(True, axis="y", ls=":", alpha=0.5)
    _save(fig, outdir, "04_sbr_zoom.png")


def _fig_equalizer(outdir, chdata, fom_result, param):
    M = int(param.samples_per_ui)
    sbr = np.asarray(fom_result.sbr, dtype=float).ravel()
    uneq = np.asarray(chdata[0].uneq_pulse_response, dtype=float).ravel()
    t_s = int(fom_result.t_s); iup = int(np.argmax(uneq))
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot((np.arange(len(uneq)) - iup)/M, uneq*1000, label=f"uneq (cursor {uneq[iup]*1000:.1f} mV)", lw=1)
    ax.plot((np.arange(len(sbr)) - t_s)/M, sbr*1000, label=f"equalized (cursor {sbr[t_s]*1000:.1f} mV)", lw=1.3)
    ax.set_xlim(-5, 20); ax.axvline(0, color="red", lw=0.7)
    ax.set_xlabel("time [UI from cursor]"); ax.set_ylabel("SBR [mV] (absolute)")
    ax.set_title("Equalized vs unequalized SBR (absolute, Tx-referred)")
    ax.grid(True, ls=":", alpha=0.5); ax.legend(fontsize=8)
    _save(fig, outdir, "06_eq_vs_uneq_sbr.png")
    ft = getattr(fom_result, "FOM_TRACKER", None)
    if ft is not None:
        evals = np.asarray(ft, dtype=float).ravel(); evals = evals[evals != 0.0]
        if evals.size:
            cummax = np.maximum.accumulate(evals)
            fig, ax = plt.subplots(figsize=(9, 4.4))
            ax.plot(evals, ".", ms=2, alpha=0.4, label="FOM per evaluation")
            ax.plot(cummax, lw=1.5, label="accepted-best (running max)")
            ax.set_xlabel("evaluation index"); ax.set_ylabel("dB"); ax.set_title("FOM convergence")
            ax.grid(True, ls=":", alpha=0.5); ax.legend(fontsize=8)
            _save(fig, outdir, "06_fom_convergence.png")


def _fig_pdfs(outdir, Noise_Struct, PDF, CDF, param):
    specBER = float(param.specBER)
    px = np.asarray(PDF.x, dtype=float).ravel(); py = np.asarray(PDF.y, dtype=float).ravel()
    fig, ax = plt.subplots(figsize=(9.5, 5))
    for attr, lab in [("noise_pdf", "noise PDF"), ("isi_and_xtalk_pdf", "ISI PDF")]:
        p = getattr(Noise_Struct, attr, None)
        if p is not None:
            ax.semilogy(np.asarray(p.x).ravel()*1000, np.maximum(np.asarray(p.y).ravel(), 1e-20), label=lab)
    ax.semilogy(px*1000, np.maximum(py, 1e-20), lw=1.6, label="combined PDF")
    ax.axhline(specBER, ls="--", color="red", label=f"DER ({specBER:.0e})")
    cdf = np.asarray(CDF, dtype=float).ravel(); ix = np.where(cdf > specBER)[0]
    if len(ix):
        ani = abs(px[ix[0]])*1000; ax.set_xlim(-1.4*ani, 1.4*ani)
    ax.set_ylim(1e-18, 1); ax.set_xlabel("voltage [mV]"); ax.set_ylabel("probability")
    ax.set_title("Noise / ISI / combined PDFs"); ax.grid(True, which="both", ls=":", alpha=0.5); ax.legend(fontsize=8)
    _save(fig, outdir, "07_pdfs.png")


def _fig_bathtubs(outdir, chdata, PDF, COM_SNR_Struct, param):
    specBER = float(param.specBER)
    A_s = float(COM_SNR_Struct.A_s)
    px = np.asarray(PDF.x, dtype=float).ravel(); py = np.asarray(PDF.y, dtype=float).ravel()
    _tb = getattr(chdata[0], "timing_bathtub", None)
    _vb = _tb.get("vbt_ber") if isinstance(_tb, dict) else None
    if _vb is not None:
        # One curve per eye, each swept over the decision threshold at the
        # centre phase, so a PAM-4 link shows three bathtubs sitting at their
        # own eye levels. The previous version drew two walls built from
        # +/-A_s and the single combined PDF -- an NRZ picture that puts every
        # eye at the same place and cannot show the outer eyes at all.
        vax = np.asarray(_tb["vbt_threshold_V"], dtype=float).ravel() * 1000.0
        vb = np.asarray(_vb, dtype=float)
        vth = np.asarray(_tb.get("eye_threshold_V", []), dtype=float).ravel() * 1000.0
        n_eyes = vb.shape[0]
        labels = ["lower", "central", "upper"] if n_eyes == 3 else \
                 [f"eye{i + 1}" for i in range(n_eyes)]
        fig, ax = plt.subplots(figsize=(9.5, 5))
        for i in range(n_eyes):
            ln, = ax.semilogy(vax, np.maximum(vb[i], 1e-20), lw=1.4,
                              label=f"{labels[i]} eye")
            if i < vth.size:
                ax.axvline(vth[i], color=ln.get_color(), ls=":", lw=0.9)
        ax.axhline(specBER, ls="--", color="k", label=f"DER ({specBER:.0e})")
        ax.set_ylim(1e-15, 1)
        ax.set_xlabel("decision threshold [mV]"); ax.set_ylabel("BER")
        ax.set_title(f"VOLTAGE bathtub  (COM={float(COM_SNR_Struct.COM):.2f} dB)")
        ax.grid(True, which="both", ls=":", alpha=0.5); ax.legend(fontsize=8)
        _save(fig, outdir, "08_voltage_bathtub.png")
    else:
        # No per-level data (COM_eye_width did not run). Fall back to the
        # single-PDF view rather than emitting nothing.
        vbt_l = np.abs(0.5 - np.cumsum(0.5 * py))
        vbt_r = np.flip(0.5 - np.cumsum(np.flip(0.5 * py)))
        fig, ax = plt.subplots(figsize=(9.5, 5))
        ax.semilogy((px - A_s)*1000, np.maximum(vbt_l, 1e-20), "b", label="left wall (-A_s + noise)")
        ax.semilogy((px + A_s)*1000, np.maximum(vbt_r, 1e-20), "r", label="right wall (+A_s + noise)")
        ax.axhline(specBER, ls="--", color="k", label=f"DER ({specBER:.0e})")
        ax.plot([-A_s*1000, A_s*1000], [0.5, 0.5], "ok", ms=5)
        ax.set_ylim(1e-15, 1); ax.set_xlim(-2.2*A_s*1000, 2.2*A_s*1000)
        ax.set_xlabel("decision threshold [mV]"); ax.set_ylabel("BER")
        ax.set_title(f"VOLTAGE bathtub, combined PDF  (COM={float(COM_SNR_Struct.COM):.2f} dB)")
        ax.grid(True, which="both", ls=":", alpha=0.5); ax.legend(fontsize=8)
        _save(fig, outdir, "08_voltage_bathtub.png")
    # timing bathtub + eye contour (attached to chdata[0] by COM_eye_width)
    tb = getattr(chdata[0], "timing_bathtub", None)
    if tb is not None:
        p = np.asarray(tb["phase_UI"]).ravel(); b = np.asarray(tb["ber_eyes"])
        n_eyes = b.shape[0]
        worst = int(np.nanargmax([np.nanmin(b[i]) for i in range(n_eyes)]))
        labels = ["lower", "central", "upper"] if n_eyes == 3 else [f"eye{i}" for i in range(n_eyes)]
        fig, ax = plt.subplots(figsize=(9.5, 5))
        for i in range(n_eyes):
            ax.semilogy(p, np.maximum(b[i], 1e-20), lw=(2 if i == worst else 1),
                        label=labels[i] + (" (worst)" if i == worst else ""))
        ax.axhline(specBER, ls="--", color="k", label=f"DER ({specBER:.0e})")
        ax.set_ylim(1e-13, 1); ax.set_xlim(-0.5, 0.5)
        ax.set_xlabel("sample phase [UI from center]"); ax.set_ylabel("BER")
        ax.set_title("TIMING bathtub (BER vs sample time)"); ax.grid(True, which="both", ls=":", alpha=0.5)
        ax.legend(fontsize=8)
        _save(fig, outdir, "08_timing_bathtub.png")
        ec = np.asarray(tb["eye_contour"])
        fig, ax = plt.subplots(figsize=(9.5, 5))
        for i in range(ec.shape[1] // 2):
            ax.plot(p, ec[:, 2*i]*1000, "g", lw=1); ax.plot(p, ec[:, 2*i+1]*1000, "b", lw=1)
        ax.set_xlabel("sample phase [UI from center]"); ax.set_ylabel("voltage [mV]")
        ax.set_title("Eye-diagram contour at DER"); ax.grid(True, ls=":", alpha=0.5)
        _save(fig, outdir, "08_eye_contour.png")


def _fig_contribution_pie(outdir, Noise_Struct, PDF, COM_SNR_Struct, param):
    # COM contribution breakdown — same math as MATLAB plot_pie_com (L9011-9091):
    # split COM by the squared at-BER amplitude of each noise source.
    BER = float(param.specBER)
    def _iex(pdf):
        cs = np.cumsum(np.asarray(pdf.y, dtype=float))
        idx = np.where(np.abs(cs) >= BER)[0]
        return int(idx[0]) if len(idx) else len(cs) - 1
    sci, cci, npdf = Noise_Struct.sci_pdf, Noise_Struct.cci_pdf, Noise_Struct.noise_pdf
    maxn = np.array([abs(np.asarray(sci.x)[_iex(sci)]),
                     abs(np.asarray(npdf.x)[_iex(npdf)]),
                     abs(np.asarray(cci.x)[_iex(cci)])], dtype=float)
    maxn_tot = abs(np.asarray(PDF.x)[_iex(PDF)])
    A_s = float(COM_SNR_Struct.A_s)
    if maxn_tot <= 0:
        return
    # 20*log10(A_s/maxn_tot) is COM as the PDF path alone sees it, i.e.
    # COM_orig -- it predates any MLSE adjustment. The reported figure is
    # COM_SNR_Struct.COM (COM_orig + delta_COM), so split THAT: the slice
    # fractions are ratios and do not change, but the total the reader is
    # shown now matches the COM on the summary line.
    COM_pdf = 20.0 * np.log10(A_s / maxn_tot)
    COM = float(getattr(COM_SNR_Struct, "COM", COM_pdf))
    s2 = float(np.dot(maxn, maxn))
    com_per = COM * (maxn ** 2 / s2) if s2 > 0 else np.zeros(3)
    tot = float(np.sum(com_per))
    if tot <= 0:
        return
    labels = ["ISI", "System noise", "Crosstalk"]
    pct = 100.0 * com_per / tot
    keep = com_per > 0
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.pie(com_per[keep], startangle=90, counterclock=False,
           labels=[f"{labels[i]}\n{pct[i]:.1f}%" for i in range(3) if keep[i]])
    _ttl = f"COM contribution breakdown  (COM={COM:.2f} dB)"
    if abs(COM - COM_pdf) > 5e-3:
        _ttl += f"\nsplit of the reported COM; pre-MLSE COM_orig = {COM_pdf:.2f} dB"
    ax.set_title(_ttl)
    _save(fig, outdir, "09_contribution_pie.png")


# ── public entry ─────────────────────────────────────────────────────────────
def save_case_figures(case_dir, param, OP, chdata, fom_result, Noise_Struct, PDF, CDF, COM_SNR_Struct):
    """Write the full standard figure set for one package-test-case to case_dir."""
    os.makedirs(case_dir, exist_ok=True)
    jobs = [
        ("sparams",   lambda: _fig_sparams(case_dir, chdata, param)),
        ("filters",   lambda: _fig_filters(case_dir, chdata, param, OP)),
        ("pulse",     lambda: _fig_pulse(case_dir, chdata, param)),
        ("equalizer", lambda: _fig_equalizer(case_dir, chdata, fom_result, param)),
        ("pdfs",      lambda: _fig_pdfs(case_dir, Noise_Struct, PDF, CDF, param)),
        ("bathtubs",  lambda: _fig_bathtubs(case_dir, chdata, PDF, COM_SNR_Struct, param)),
        ("contrib",   lambda: _fig_contribution_pie(case_dir, Noise_Struct, PDF, COM_SNR_Struct, param)),
    ]
    import time
    t0 = time.time() - 1.0  # small slack for filesystem timestamp granularity
    failed = []
    for label, job in jobs:
        try:
            job()
        except Exception as e:
            failed.append(label)
            print(f"    [com_plots] {label} FAILED: {e}")
    # Count figures actually (re)written by THIS run, not whatever is on disk —
    # otherwise stale files mask write failures (e.g. a locked/open PNG).
    n_written = sum(1 for x in os.listdir(case_dir) if x.endswith(".png")
                    and os.path.getmtime(os.path.join(case_dir, x)) >= t0)
    n_present = len([x for x in os.listdir(case_dir) if x.endswith(".png")])
    msg = f"    [com_plots] {n_written} figures written this run ({n_present} present) -> {case_dir}"
    if failed:
        msg += f"  [WARNING: {len(failed)} job(s) FAILED — existing figures NOT updated]"
    print(msg)
    return n_written
