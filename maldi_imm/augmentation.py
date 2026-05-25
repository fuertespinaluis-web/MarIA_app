"""
MALDI-TOF data augmentation routines.

Algorithms based on:
Guerrero-López, Alejandro, et al. "Overcoming variability challenges for
Clostridioides difficile via data augmentation techniques."
bioRxiv (2024): 2024-10. https://doi.org/10.1101/2024.10.29.620907

Testing new data augmentation techniques for DINOv2
"""

import numpy as np
from typing import Optional
from .SpectrumObject import SpectrumObject


class DataAugmenter:
    """
    Unified MALDI-TOF augmenter with two modes.

    Parameters
    ----------
    random_state : Optional[int]
        Seed for reproducibility.
    cdm_training : bool
        If True (default), use only the two legacy augmentations used for CDM training
        to keep the CDM's data distribution intact. If False, use the richer, label-agnostic
        MALDI+DINO pipeline (drift/warp/jitter/blur/peak-drop/micro-peaks).
    renorm_tic : bool
        (Reserved; not used here. Keep for compatibility if you later want to re-TIC every step.)

    Notes
    -----
    • All methods operate on binned spectra and return a fresh SpectrumObject.
    • ppm handling uses ×1e-6 (correct scale).
    """

    def __init__(self, random_state: Optional[int] = None, cdm_training: bool = True, renorm_tic: bool = True):
        self.rng = np.random.default_rng(random_state)
        self.cdm_training = cdm_training
        # NOTE: renorm_tic is currently unused; kept for compatibility with callers.

    # ---------- sklearn-ish API ----------
    def fit(self, X, y=None):
        """No-op to mimic sklearn fit/transform API."""
        return self

    def transform(self, X):
        """Apply `augment` to a list of SpectrumObject."""
        return [self.augment(s) for s in X]

    # ---------- main entry ----------
    def augment(self, spectrum: SpectrumObject) -> SpectrumObject:
        """
        Entry point.

        CDM mode  : apply the original pair (shifting + machine_variability).
        DINO mode : apply the MALDI+DINO pipeline (mild always-on + stochastic extras).
        """
        if self.cdm_training:
            s = self.shifting(spectrum, p0=0.5, p3=0.3, p6=0.15, p9=0.05)
            s = self.machine_variability(s, p1=0.01, v3=0.001)
            return s
        else:
            return self._augment_dino(spectrum)

    # ---------- DINO multiview ----------
    def multiview(self, spectrum: SpectrumObject, n_global: int = 2, n_local: int = 6, local_width_da: float = 1500.0):
        """
        Build DINO-style views: 2 global (mild) + N local crops (stronger).

        Returns
        -------
        list[SpectrumObject]
        """
        views = []
        # global (mild)
        for _ in range(n_global):
            views.append(self._augment_dino(spectrum, stronger=False))
        # local (crop + stronger)
        for _ in range(n_local):
            lc = self.crop_local(spectrum, width_da=local_width_da)
            views.append(self._augment_dino(lc, stronger=True))
        return views

    # ---------- DINO pipeline (private) ----------
    def _augment_dino(self, spectrum: SpectrumObject, stronger: bool = False) -> SpectrumObject:
        """
        MALDI+DINO pipeline.
        Always-on mild ops, plus stochastic extras. `stronger=True` dials them up a bit.
        """
        s = SpectrumObject(spectrum.mz.copy(), spectrum.intensity.copy(), spectrum.meta.copy())

        # Always-on mild globals (drift + small intensity jitter; exact magnitudes capped by rel_rms_max)
        s = self.mz_drift_ppm(s)
        s = self.intensity_jitter(
            s,
            gain_std=10 if not stronger else 15,   # NOTE: actual effect limited by rel_rms_max
            noise_rel=10 if not stronger else 15,  # (values are large but budget-capped below)
        )

        # Stochastic extras
        if self.rng.random() < (0.50 if not stronger else 0.70):
            s = self.local_calib_warp(s, n_knots=10, max_ppm=20 if not stronger else 50)
        if self.rng.random() < (0.50 if not stronger else 0.70):
            s = self.resolution_blur(s, sigma_bins=(0.5, 1.5) if not stronger else (0.8, 1.8))
        if self.rng.random() < (0.50 if not stronger else 0.60):
            s = self.peak_dropout(
                s,
                q=0.90 if not stronger else 0.95,
                select_min=5 if not stronger else 8,
                select_max=10 if not stronger else 15,
                window_da=6.0 if not stronger else 8.0,
                atten_range=(0.35, 0.7) if not stronger else (0.25, 0.5),
                hw_detect=10,
            )
        if self.rng.random() < (0.30 if not stronger else 0.40):
            s = self.spurious_micropeaks(
                s,
                rate=0.002 if not stronger else 0.003,
                amp_rel=0.02 if not stronger else 0.03,
            )

        return s

    # ---------- DINO components ----------
    def mz_drift_ppm(self, s: SpectrumObject) -> SpectrumObject:
        """
        Uniform calibration drift in ppm (parts per million).

        • 90% of the time: N(0, 20 ppm)
        • 10% of the time: N(0, 100 ppm)  ← "bad day" tails
        """
        drift_ppm = self.rng.normal(0, 100.0) if self.rng.random() < 0.1 else self.rng.normal(0, 20.0)
        drift = drift_ppm * 1e-6  # ppm → fractional
        # NOTE: The sign flip below applies globally (it does not affect “left vs right” halves).
        if self.rng.random() < 0.5:
            drift = -drift
        mz_new = s.mz * (1.0 + drift)
        # Resample back to the original grid so we keep `s.mz` as the axis.
        inten = np.interp(s.mz, mz_new, s.intensity, left=0, right=0)
        return SpectrumObject(s.mz, inten, s.meta.copy())

    def local_calib_warp(self, s: SpectrumObject, n_knots: int = 6, max_ppm: float = 6.0) -> SpectrumObject:
        """
        Smooth, monotone, m/z-dependent calibration warp.

        • Interpolates random ppm offsets at `n_knots` along the index domain.
        • 10% chance to expand range to ±80 ppm to mimic occasional large errors.
        • Enforces monotonicity and resamples intensities back to original grid.
        """
        mz = s.mz
        L = len(mz)
        idx = np.linspace(0, L - 1, n_knots, dtype=int)
        if self.rng.random() < 0.1:
            max_ppm = 80.0
        warp_ppm = self.rng.uniform(-max_ppm, max_ppm, size=n_knots) * 1e-6
        warp = np.interp(np.arange(L), idx, warp_ppm)
        mz_warped = mz * (1.0 + warp)
        mz_warped = np.maximum.accumulate(mz_warped)  # keep mapping monotone
        inten = np.interp(mz, mz_warped, s.intensity, left=0, right=0)
        return SpectrumObject(mz, inten, s.meta.copy())

    def intensity_jitter(
        self,
        s: SpectrumObject,
        gain_std: float = 10,
        noise_rel: float = 10,
        alpha: float = 1.0,
        rel_rms_max: float = 0.05,
        renorm_tic: bool = True,
    ) -> SpectrumObject:
        """
        Very simple, gentle intensity jitter.

        Steps
        -----
        1) Global gain ~ N(1, gain_std)
        2) Add small homoscedastic noise with std = noise_rel * mean(I)
        3) Blend with original using `alpha`
        4) Cap the relative RMS change to `rel_rms_max` (this is the effective knob)
        5) Optional re-TIC

        Notes
        -----
        • Although `gain_std` and `noise_rel` are large by default in calls,
          the 5% RRMS cap keeps this augmentation mild in practice.
        """
        I0 = s.intensity.astype(float)
        if I0.size == 0 or I0.sum() <= 0:
            return SpectrumObject(s.mz, I0, s.meta.copy())

        g = self.rng.normal(1.0, gain_std)
        noise_sigma = noise_rel * (I0.mean() + 1e-12)
        jittered = np.clip(g * I0 + self.rng.normal(0.0, noise_sigma, size=I0.shape), 0.0, None)

        y = (1.0 - alpha) * I0 + alpha * jittered

        diff = y - I0
        rrms = float(np.sqrt((diff**2).mean()) / (np.sqrt((I0**2).mean()) + 1e-12))
        if rrms > rel_rms_max:
            lam = rel_rms_max / rrms
            y = I0 + lam * diff

        if renorm_tic:
            ssum = y.sum()
            if ssum > 0:
                y = y / ssum

        return SpectrumObject(s.mz, y, s.meta.copy())

    def resolution_blur(self, s: SpectrumObject, sigma_bins=(0.5, 1.5)) -> SpectrumObject:
        """
        Resolution blur (Gaussian, constant σ in bins).

        • Smooths sharp apices to mimic lower resolving power / spot quality.
        • Kernel is normalized, so TIC is approximately preserved (edge effects aside).
        """
        sig = self.rng.uniform(*sigma_bins)
        rad = int(np.ceil(3 * sig))
        x = np.arange(-rad, rad + 1)
        ker = np.exp(-0.5 * (x / (sig + 1e-12)) ** 2)
        ker /= ker.sum()
        inten = np.convolve(s.intensity, ker, mode="same")
        return SpectrumObject(s.mz, np.clip(inten, 0.0, None), s.meta.copy())

    def peak_dropout(
        self,
        s: SpectrumObject,
        q: float = 0.90,
        select_min: int = 5,
        select_max: int = 10,
        window_da: float = 6.0,
        atten_range=(0.35, 0.7),
        hw_detect: int = 10,
    ) -> SpectrumObject:
        """
        Peak-level smoothing/attenuation.

        • Find local maxima; keep those ≥ q-th percentile of peak heights.
        • Randomly select `select_min..select_max` peaks.
        • Within ±window_da around each selected apex, replace with a heavily
          smoothed trace scaled by a factor in `atten_range` (not hard zero).

        This mimics stricter centroiding or a weak prep where a strong peak
        is mostly gone but not a flat zero.
        """
        mz = s.mz
        y = s.intensity.astype(float).copy()
        L = len(y)
        if L < 5:
            return SpectrumObject(mz, y, s.meta.copy())

        # Local maxima via sliding window argmax
        win = 2 * hw_detect + 1
        if L < win:
            return SpectrumObject(mz, y, s.meta.copy())
        sw = np.lib.stride_tricks.sliding_window_view(y, win)
        centers = sw[:, hw_detect]
        maxima = np.where(
            (centers > sw[:, :hw_detect].max(axis=1)) &
            (centers >= sw[:, hw_detect + 1:].max(axis=1))
        )[0] + hw_detect
        if maxima.size == 0:
            return SpectrumObject(mz, y, s.meta.copy())

        # Top-q peaks among maxima
        thr = np.quantile(y[maxima], q)
        strong = maxima[y[maxima] >= thr]
        if strong.size == 0:
            return SpectrumObject(mz, y, s.meta.copy())

        # Randomly select targets
        n_sel = int(self.rng.integers(low=select_min, high=select_max + 1))
        n_sel = min(n_sel, strong.size)
        sel = self.rng.choice(strong, size=n_sel, replace=False)

        # Heavy blur once, then reuse for replacements
        da = float(np.median(np.diff(mz)))
        rad = max(1, int(round(window_da / max(da, 1e-12))))
        sigma_bins = max(1.0, rad / 2.5)
        kx = np.arange(-rad, rad + 1)
        ker = np.exp(-0.5 * (kx / (sigma_bins + 1e-12)) ** 2)
        ker /= ker.sum()
        y_blur = np.convolve(y, ker, mode="same")

        for i in sel:
            a = max(0, i - rad)
            b = min(L, i + rad + 1)
            alpha = float(self.rng.uniform(*atten_range))
            y[a:b] = alpha * y_blur[a:b]

        return SpectrumObject(mz, y, s.meta.copy())

    def spurious_micropeaks(self, s: SpectrumObject, rate: float = 0.002, amp_rel: float = 0.02) -> SpectrumObject:
        """
        Add a few tiny bumps (optional nuisance).

        • Choose ~rate*L random bins and add small positive noise there.
        • Amplitude tied to q90, offset by a bit of the 20th percentile baseline.

        NOTE: This always adds at least one bump (n>=1). Reduce `rate` if too visible.
        """
        L = len(s.intensity)
        n = max(1, int(L * rate))
        idx = self.rng.choice(L, size=n, replace=False)
        base = np.quantile(s.intensity, 0.2)
        amp = amp_rel * (np.quantile(s.intensity, 0.9) + 1e-12)
        inten = s.intensity.copy()
        inten[idx] += self.rng.exponential(scale=amp, size=n) + 0.5 * base
        return SpectrumObject(s.mz, inten, s.meta.copy())

    def crop_local(self, s: SpectrumObject, width_da: float = 1500.0) -> SpectrumObject:
        """
        Random local crop of width `width_da` in Da. If spectrum is shorter than
        the requested width, returns the input unchanged.
        """
        mz = s.mz
        mmin, mmax = mz[0], mz[-1]
        if (mmax - mmin) <= width_da:
            return SpectrumObject(mz, s.intensity.copy(), s.meta.copy())
        c = self.rng.uniform(mmin + width_da / 2, mmax - width_da / 2)
        left, right = c - width_da / 2, c + width_da / 2
        mask = (mz >= left) & (mz <= right)
        return SpectrumObject(mz[mask], s.intensity[mask], s.meta.copy())

    # ---------- original CDM augmentations (unchanged) ----------
    def shifting(self, spectrum: SpectrumObject, p0=0.5, p3=0.3, p6=0.15, p9=0.05) -> SpectrumObject:
        """
        Legacy 'shifting' augmentation from Guerrero-López et al.

        • Randomly pick disjoint index subsets and shift their m/z by {0, ±3, ±6, ±9} Da.
        • Interpolate intensities back onto the original m/z grid.

        NOTE: To keep axis+intensity consistent with the other augs, we return the
        original m/z grid with warped intensities.
        """
        s = SpectrumObject(spectrum.mz.copy(), spectrum.intensity.copy(), spectrum.meta.copy())
        N = len(s.mz)
        min_intensity, max_intensity = float(np.min(s.intensity)), float(np.max(s.intensity))

        shift_probs = [p0, p3, p6, p9]
        shifts = [0, 3, 6, 9]  # Da
        all_idxs = np.arange(N)
        shifted_mz = s.mz.copy()
        idxs_remaining = set(all_idxs)

        for prob, shift in zip(shift_probs, shifts):
            n = int(prob * N)
            if n == 0 or not idxs_remaining:
                continue
            idx = self.rng.choice(list(idxs_remaining), n, replace=False)
            directions = self.rng.choice([-1, 1], n)
            shifted_mz[idx] += directions * shift
            idxs_remaining -= set(idx)

        # Resample onto the ORIGINAL grid to keep (mz, intensity) consistent.
        interp_intensity = np.interp(s.mz, shifted_mz, s.intensity, left=0, right=0)
        interp_intensity = np.clip(interp_intensity, min_intensity, max_intensity)

        # IMPORTANT: return s.mz here (not shifted_mz), to match all other augs.
        return SpectrumObject(mz=s.mz, intensity=interp_intensity, meta=s.meta.copy())

    def machine_variability(self, spectrum: SpectrumObject, p1: float = 0.05, v3: float = 0.01) -> SpectrumObject:
        """
        Legacy 'machine variability' augmentation from Guerrero-López et al.

        • Pick p1 fraction of bins that are at the current minimum and add Gaussian noise.
        • Clip to [min, max] to stay in-range.

        This crudely simulates occasional low-level bumps in the baseline.
        """
        s = SpectrumObject(spectrum.mz.copy(), spectrum.intensity.copy(), spectrum.meta.copy())
        min_intensity, max_intensity = float(np.min(s.intensity)), float(np.max(s.intensity))
        zeros = np.flatnonzero(s.intensity == min_intensity)
        n_perturb = int(p1 * len(zeros))
        if n_perturb > 0:
            idx = self.rng.choice(zeros, n_perturb, replace=False)
            s.intensity[idx] += self.rng.normal(0, np.sqrt(v3), n_perturb)
            s.intensity = np.clip(s.intensity, min_intensity, max_intensity)
        return s
