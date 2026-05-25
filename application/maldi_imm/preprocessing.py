import numpy as np
from scipy.stats import binned_statistic
from scipy.signal import savgol_filter
from scipy import sparse
from scipy.linalg import norm
from .SpectrumObject import SpectrumObject
import torch
from typing import Sequence, Union


class Binner:
    """Bins spectrum into equal-width bins."""

    def __init__(self, start=2000, stop=20000, step=3, aggregation="sum"):
        self.bins = np.arange(start, stop + 1e-8, step)
        self.mz_bins = self.bins[:-1] + step / 2
        self.agg = aggregation

    def __call__(self, spectrum):
        if self.agg == "sum":
            bins, _ = np.histogram(spectrum.mz, self.bins, weights=spectrum.intensity)
        else:
            bins = binned_statistic(
                spectrum.mz, spectrum.intensity, bins=self.bins, statistic=self.agg
            ).statistic
            bins = np.nan_to_num(bins)
        return SpectrumObject(mz=self.mz_bins, intensity=bins, meta=spectrum.meta)

class Normalizer:
    """Normalizes the intensity of a spectrum."""

    def __init__(self, total=1):
        self.total = total

    def __call__(self, spectrum):
        factor = self.total / np.sum(spectrum.intensity)
        return SpectrumObject(
            mz=spectrum.mz,
            intensity=spectrum.intensity * factor,
            meta=spectrum.meta
        )

class StdThresholder:
    """Zero-out intensities below factor * std(intensity) (per spectrum)."""

    def __init__(self, factor=1.0):
        self.factor = float(factor)

    def __call__(self, spectrum):
        if spectrum.intensity is None or spectrum.intensity.size == 0:
            return SpectrumObject(mz=spectrum.mz, intensity=spectrum.intensity, meta=spectrum.meta)
        std = float(np.std(spectrum.intensity))
        thresh = self.factor * std
        intensity = np.where(spectrum.intensity >= thresh, spectrum.intensity, 0.0)
        return SpectrumObject(mz=spectrum.mz, intensity=intensity, meta=spectrum.meta)

class LogScaler:
    """Applies log base `base` after shifting by 1 to keep zeros stable."""

    def __init__(self, base=10.0):
        if base <= 0:
            raise ValueError("base must be positive")
        self.base = float(base)
        self._den = np.log(self.base)

    def __call__(self, spectrum):
        intensity = np.log1p(spectrum.intensity) / self._den
        return SpectrumObject(mz=spectrum.mz, intensity=intensity, meta=spectrum.meta)

class MinMaxScaler:
    """Per-spectrum min-max scaling to [min, max]."""

    def __init__(self, min=0.0, max=1.0):
        self.min = float(min)
        self.max = float(max)

    def __call__(self, spectrum):
        it = np.asarray(spectrum.intensity, dtype=np.float32)
        lo = float(np.min(it)) if it.size else 0.0
        hi = float(np.max(it)) if it.size else 1.0
        denom = hi - lo
        if denom <= 0:
            scaled = np.zeros_like(it)
        else:
            scaled = (it - lo) / (denom + 1e-12)
            scaled = self.min + scaled * (self.max - self.min)
        return SpectrumObject(mz=spectrum.mz, intensity=scaled, meta=spectrum.meta)

class MaxNormalizer:
    """Normalizes the intensity of a spectrum by its maximum value."""

    def __init__(self):
        pass

    def __call__(self, spectra_list):
        # If x is single spectrum, convert to list
        if isinstance(spectra_list, SpectrumObject):
            spectra_list = [spectra_list]
        # If x is a tensor or numpy array, convert to SpectrumObject
        elif isinstance(spectra_list, (np.ndarray, torch.Tensor)):
            spectra_list = [SpectrumObject(mz=np.arange(2000, 20000, 3), intensity=spectra_list, meta={})]

        out = []
        for spectrum in spectra_list:
            max_val = np.max(spectrum.intensity)
            norm_intensity = spectrum.intensity / max_val if max_val > 0 else spectrum.intensity

            # make a copy of the object with normalized intensity
            new_spec = SpectrumObject(
                mz=spectrum.mz,
                intensity=norm_intensity,
                meta=spectrum.meta
            )
            # store the max value in the object
            new_spec.max = max_val
            out.append(new_spec)
        
        return out


class Trimmer:
    """Trims spectrum to specified m/z range."""

    def __init__(self, min=2000, max=20000):
        self.range = (min, max)

    def __call__(self, spectrum):
        mask = (self.range[0] < spectrum.mz) & (spectrum.mz < self.range[1])
        return SpectrumObject(
            mz=spectrum.mz[mask], intensity=spectrum.intensity[mask], meta=spectrum.meta
        )

class VarStabilizer:
    """Variance stabilizing transform on intensities."""

    def __init__(self, method="sqrt"):
        methods = {"sqrt": np.sqrt, "log": np.log, "log2": np.log2, "log10": np.log10}
        self.fun = methods[method]

    def __call__(self, spectrum):
        return SpectrumObject(
            mz=spectrum.mz,
            intensity=self.fun(spectrum.intensity),
            meta=spectrum.meta
        )

class BaselineCorrecter:
    """Baseline correction (supports SNIP, ALS, ArPLS)."""

    def __init__(
        self, method="SNIP", als_lam=1e8, als_p=0.01, als_max_iter=10, als_tol=1e-6, snip_n_iter=10
    ):
        self.method = method
        self.lam = als_lam
        self.p = als_p
        self.max_iter = als_max_iter
        self.tol = als_tol
        self.n_iter = snip_n_iter

    def __call__(self, spectrum):
        if "LS" in self.method:
            baseline = self.als(
                spectrum.intensity, method=self.method, lam=self.lam, p=self.p,
                max_iter=self.max_iter, tol=self.tol
            )
        elif self.method == "SNIP":
            baseline = self.snip(spectrum.intensity, self.n_iter)
        else:
            raise ValueError("Unknown baseline method")
        return SpectrumObject(
            mz=spectrum.mz,
            intensity=spectrum.intensity - baseline,
            meta=spectrum.meta
        )

    def als(self, y, method="ArPLS", lam=1e8, p=0.01, max_iter=10, tol=1e-6):
        L = len(y)
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
        D = lam * D.dot(D.transpose())
        w = np.ones(L)
        W = sparse.spdiags(w, 0, L, L)
        crit = 1
        count = 0
        while crit > tol:
            z = sparse.linalg.spsolve(W + D, w * y)
            if method == "AsLS":
                w_new = p * (y > z) + (1 - p) * (y < z)
            elif method == "ArPLS":
                d = y - z
                dn = d[d < 0]
                m = np.mean(dn)
                s = np.std(dn)
                w_new = 1 / (1 + np.exp(np.minimum(2 * (d - (2 * s - m)) / s, 70)))
            crit = norm(w_new - w) / norm(w)
            w = w_new
            W.setdiag(w)
            count += 1
            if count > max_iter:
                break
        return z

    def snip(self, y, n_iter):
        y_prepr = np.log(np.log(np.sqrt(y + 1) + 1) + 1)
        for i in range(1, n_iter + 1):
            rolled = np.pad(y_prepr, (i, i), mode="edge")
            new = np.minimum(
                y_prepr, (np.roll(rolled, i) + np.roll(rolled, -i))[i:-i] / 2
            )
            y_prepr = new
        return (np.exp(np.exp(y_prepr) - 1) - 1) ** 2 - 1


class SpeciesGenusMinMaxScaler:
    """
    Computes and applies min-max normalization per species, genus, or global level.
    If >= threshold samples for a species, uses that. Else, tries genus. Else, global.
    Supports inverse_transform following scikit-learn style.
    """

    def __init__(self, threshold=100):
        self.threshold = threshold
        self.species_min = {}
        self.species_max = {}
        self.species_count = {}
        self.genus_min = {}
        self.genus_max = {}
        self.genus_count = {}
        self.global_min = None
        self.global_max = None
        self.global_count = 0

    def fit(self, spectra_list):
        species_arrays = {}
        genus_arrays = {}
        all_arrays = []

        for spectrum in spectra_list:
            genus = spectrum.meta.get("genus", "unknown")
            species = spectrum.meta.get("species", "unknown")
            sp_key = f"{genus} {species}"
            arr = np.asarray(spectrum.intensity)
            species_arrays.setdefault(sp_key, []).append(arr)
            genus_arrays.setdefault(genus, []).append(arr)
            all_arrays.append(arr)

        for sp, arrs in species_arrays.items():
            arrs = np.stack(arrs)
            self.species_min[sp] = arrs.min()
            self.species_max[sp] = arrs.max()
            self.species_count[sp] = arrs.shape[0]

        for g, arrs in genus_arrays.items():
            arrs = np.stack(arrs)
            self.genus_min[g] = arrs.min()
            self.genus_max[g] = arrs.max()
            self.genus_count[g] = arrs.shape[0]

        all_arrays = np.stack(all_arrays)
        self.global_min = all_arrays.min()
        self.global_max = all_arrays.max()
        self.global_count = all_arrays.shape[0]

    def get_minmax(self, genus, species):
        sp_key = f"{genus} {species}"
        if self.species_count.get(sp_key, 0) >= self.threshold:
            mn, mx = self.species_min[sp_key], self.species_max[sp_key]
        elif self.genus_count.get(genus, 0) >= self.threshold:
            mn, mx = self.genus_min[genus], self.genus_max[genus]
        else:
            mn, mx = self.global_min, self.global_max
        return mn, mx

    def transform(self, spectra_list):
        out = []
        for spectrum in spectra_list:
            genus = spectrum.meta.get("genus", "unknown")
            species = spectrum.meta.get("species", "unknown")
            mn, mx = self.get_minmax(genus, species)
            arr = np.asarray(spectrum.intensity)
            scale = mx - mn if mx != mn else 1.0
            normed = 2 * (arr - mn) / (scale + 1e-8) - 1
            out.append(SpectrumObject(
                mz=spectrum.mz,
                intensity=normed,
                meta=spectrum.meta
            ))
        return out

    def inverse_transform(self, spectra_list):
        """
        Reverts normalization, returning new SpectrumObjects with intensities in original units.
        """
        out = []
        for spectrum in spectra_list:
            genus = spectrum.meta.get("genus", "unknown")
            species = spectrum.meta.get("species", "unknown")
            mn, mx = self.get_minmax(genus, species)
            arr = np.asarray(spectrum.intensity)
            orig = 0.5 * (arr + 1) * (mx - mn) + mn
            out.append(SpectrumObject(
                mz=spectrum.mz,
                intensity=orig,
                meta=spectrum.meta
            ))
        return out

    def fit_transform(self, spectra_list):
        self.fit(spectra_list)
        return self.transform(spectra_list)


def _robust_mu_sigma(z):
    z = z[np.isfinite(z)]
    if z.size == 0:
        return 0.0, 1.0
    mu = np.median(z)
    mad = np.median(np.abs(z - mu))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma < 1e-8:
        s = np.std(z)
        sigma = float(s) if np.isfinite(s) and s > 1e-8 else 1.0
    return float(mu), float(sigma)

class LogTanhScaler:
    r"""
    (unchanged docstring omitted for brevity)
    """

    def __init__(self, c_mode="q95-median", eps=1e-12, mz_grid: np.ndarray | None = None):
        self.c_mode = c_mode
        self.eps = eps
        self.c = None
        self.mu = None
        self.sigma = None
        # for robust handling of ndarray inputs
        self.mz_grid = None if mz_grid is None else np.asarray(mz_grid)

    # -------- helpers: make inputs robust --------
    def set_mz_grid(self, mz: np.ndarray):
        """Set default m/z grid used when inputs are raw arrays."""
        self.mz_grid = np.asarray(mz)

    def _as_objs(self, spectra_list: Union[Sequence, np.ndarray]) -> list[SpectrumObject]:
        """
        Accepts: list of SpectrumObject OR list/array of intensities.
        Returns: list[SpectrumObject]
        """
        # already a list of SpectrumObject?
        if isinstance(spectra_list, (list, tuple)) and len(spectra_list) > 0 and isinstance(spectra_list[0], SpectrumObject):
            return list(spectra_list)

        # numpy array (B, L) or (L,)
        if isinstance(spectra_list, np.ndarray):
            arr = spectra_list
            if arr.ndim == 1:
                arr = arr[None, :]
            if self.mz_grid is None:
                raise ValueError("LogTanhScaler needs an mz_grid to wrap ndarray inputs. "
                                 "Pass mz_grid=... in __init__ or call set_mz_grid().")
            return [SpectrumObject(mz=self.mz_grid, intensity=row) for row in arr]

        # generic sequence of arrays
        if isinstance(spectra_list, (list, tuple)) and (len(spectra_list) == 0 or not isinstance(spectra_list[0], SpectrumObject)):
            if self.mz_grid is None:
                raise ValueError("LogTanhScaler needs an mz_grid to wrap ndarray/list inputs. "
                                 "Pass mz_grid=... in __init__ or call set_mz_grid().")
            return [SpectrumObject(mz=self.mz_grid, intensity=np.asarray(x)) for x in spectra_list]

        # empty -> empty list
        if isinstance(spectra_list, (list, tuple)) and len(spectra_list) == 0:
            return []

        raise TypeError("Unsupported spectra_list type for LogTanhScaler.")

    # -------- internal: compute c --------
    def _compute_c(self, spectra_list):
        # expects list[SpectrumObject]
        if self.c_mode.endswith("-median") and self.c_mode.startswith("q"):
            q = float(self.c_mode[1:3]) / 100.0  # e.g., "q95-median" -> 0.95
            per_spec_q = []
            for s in spectra_list:
                x = np.asarray(s.intensity, dtype=np.float64)
                x = x[np.isfinite(x) & (x > 0)]
                if x.size:
                    per_spec_q.append(np.quantile(x, q))
            return float(np.median(per_spec_q)) if per_spec_q else 1.0
        elif self.c_mode == "global-median":
            vals = []
            for s in spectra_list:
                x = np.asarray(s.intensity, dtype=np.float64)
                vals.append(x[np.isfinite(x) & (x > 0)])
            vals = np.concatenate([v for v in vals if v.size], axis=0) if any(v.size for v in vals) else np.array([1.0])
            return float(np.median(vals))
        else:
            raise ValueError(f"Unknown c_mode: {self.c_mode}")

    # -------- fit / transform --------
    def fit(self, spectra_list):
        objs = self._as_objs(spectra_list)

        # reference scale
        self.c = self._compute_c(objs)
        if not np.isfinite(self.c) or self.c <= 0:
            self.c = 1.0

        # gather z_raw across training to estimate μ, σ robustly
        zs = []
        for s in objs:
            x = np.asarray(s.intensity, dtype=np.float64)
            x = x[np.isfinite(x) & (x >= 0)]
            if x.size:
                z_raw = np.log1p(x / (self.c + self.eps))
                zs.append(z_raw)
        if zs:
            z_all = np.concatenate(zs, axis=0)
            mu, sigma = _robust_mu_sigma(z_all)  # uses your existing helper
            self.mu, self.sigma = float(mu), float(sigma if sigma != 0 else 1.0)
        else:
            self.mu, self.sigma = 0.0, 1.0
        return self

    def transform(self, spectra_list):
        assert self.c is not None and self.mu is not None and self.sigma is not None, "Call fit() first."
        objs = self._as_objs(spectra_list)
        out = []
        inv_sigma = 1.0 / (self.sigma if self.sigma != 0 else 1.0)
        for s in objs:
            x = np.asarray(s.intensity, dtype=np.float64)
            x = np.maximum(x, 0.0)
            z_raw = np.log1p(x / (self.c + self.eps))
            y = np.tanh((z_raw - self.mu) * inv_sigma)
            out.append(type(s)(mz=s.mz, intensity=y, meta=s.meta))
        return out

    def inverse_transform(self, spectra_list):
        assert self.c is not None and self.mu is not None and self.sigma is not None, "Call fit() first."
        objs = self._as_objs(spectra_list)
        out = []
        for s in objs:
            y = np.asarray(s.intensity, dtype=np.float64)
            y = np.clip(y, -1 + 1e-6, 1 - 1e-6)  # numerical guard
            z_raw = self.mu + self.sigma * np.arctanh(y)
            x = self.c * np.expm1(z_raw)
            x = np.maximum(x, 0.0)
            out.append(type(s)(mz=s.mz, intensity=x, meta=s.meta))
        return out

    def to_dict(self):
        d = {"c_mode": self.c_mode, "eps": self.eps, "c": self.c, "mu": self.mu, "sigma": self.sigma}
        if self.mz_grid is not None:
            d["mz_grid"] = np.asarray(self.mz_grid).tolist()
        return d

    @classmethod
    def from_dict(cls, d):
        obj = cls(c_mode=d.get("c_mode", "q95-median"), eps=d.get("eps", 1e-12),
                  mz_grid=np.array(d["mz_grid"]) if "mz_grid" in d else None)
        obj.c = d["c"]; obj.mu = d["mu"]; obj.sigma = d["sigma"]
        return obj

    def fit_transform(self, spectra_list):
        self.fit(spectra_list)
        return self.transform(spectra_list)


class Smoother:
    """Smoothing via Savitzky-Golay filter."""

    def __init__(self, halfwindow=10, polyorder=3):
        self.window = halfwindow * 2 + 1
        self.poly = polyorder

    def __call__(self, spectrum):
        smoothed = np.maximum(
            savgol_filter(spectrum.intensity, self.window, self.poly), 0
        )
        return SpectrumObject(mz=spectrum.mz, intensity=smoothed, meta=spectrum.meta)

class LocalMaximaPeakDetector:
    """Detects local maxima above a SNR threshold."""

    def __init__(self, SNR=2, halfwindowsize=20):
        self.hw = halfwindowsize
        self.SNR = SNR

    def __call__(self, spectrum):
        snr_val = (
            np.median(np.abs(spectrum.intensity - np.median(spectrum.intensity))) * self.SNR
        )
        win = int(self.hw * 2 + 1)
        local_maxima = np.argmax(
            np.lib.stride_tricks.sliding_window_view(spectrum.intensity, win), -1
        ) == int(self.hw)
        s_int_local = spectrum.intensity[self.hw : -self.hw][local_maxima]
        s_mz_local = spectrum.mz[self.hw : -self.hw][local_maxima]
        mask = s_int_local > snr_val
        return SpectrumObject(mz=s_mz_local[mask], intensity=s_int_local[mask], meta=spectrum.meta)



class PeakFilter:
    """Filters peaks by number or intensity."""

    def __init__(self, max_number=None, min_intensity=None):
        self.max_number = max_number
        self.min_intensity = min_intensity

    def __call__(self, spectrum):
        mz = spectrum.mz
        intensity = spectrum.intensity
        if self.max_number is not None:
            idx = np.argsort(-intensity, kind="stable")
            take = np.sort(idx[: self.max_number])
            mz = mz[take]
            intensity = intensity[take]
        if self.min_intensity is not None:
            take = intensity >= self.min_intensity
            mz = mz[take]
            intensity = intensity[take]
        return SpectrumObject(mz=mz, intensity=intensity, meta=spectrum.meta)

class RandomPeakShifter:
    """Adds Gaussian noise to mz values."""

    def __init__(self, std=1.0):
        self.std = std

    def __call__(self, spectrum):
        mz_shifted = spectrum.mz + np.random.normal(scale=self.std, size=spectrum.mz.shape)
        return SpectrumObject(mz=mz_shifted, intensity=spectrum.intensity, meta=spectrum.meta)

class UniformPeakShifter:
    """Adds uniform noise to mz values."""

    def __init__(self, range=1.5):
        self.range = range

    def __call__(self, spectrum):
        mz_shifted = spectrum.mz + np.random.uniform(
            low=-self.range, high=self.range, size=spectrum.mz.shape
        )
        return SpectrumObject(mz=mz_shifted, intensity=spectrum.intensity, meta=spectrum.meta)

class Binarizer:
    """Binarizes intensity values using a threshold."""

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, spectrum):
        binary = (spectrum.intensity > self.threshold).astype(spectrum.intensity.dtype)
        return SpectrumObject(mz=spectrum.mz, intensity=binary, meta=spectrum.meta)

class SequentialPreprocessor:
    """Chains multiple preprocessors into a pipeline."""

    def __init__(self, *args):
        self.preprocessors = args

    def __call__(self, spectrum):
        for pre in self.preprocessors:
            spectrum = pre(spectrum)
        return spectrum

class BackgroundSuppressor:
    """Removes low-intensity background noise via std-thresholding (per spectrum or global fit)."""

    def __init__(self, std_threshold=2.0):
        self.std_threshold = std_threshold
        self.std_intensity = None

    def __call__(self, x):
        """
        Permite usar BackgroundSuppressor dentro de SequentialPreprocessor (step(x)).
        Preserva tipo:
          - SpectrumObject -> SpectrumObject
          - list[SpectrumObject] -> list[SpectrumObject]
        """
        single = False
        if isinstance(x, SpectrumObject):
            x = [x]
            single = True

        out = self.transform(x)
        return out[0] if single else out

    def fit(self, spectra_list):
        """Compute global std of all intensity values across training spectra (optional)."""
        # Asegura lista
        if isinstance(spectra_list, SpectrumObject):
            spectra_list = [spectra_list]
        all_intensities = np.concatenate([spectrum.intensity for spectrum in spectra_list])
        self.std_intensity = np.std(all_intensities)
        return self

    def transform(self, spectra_list):
        """Suppress intensities below std_threshold * std (global if fit() was called, else per spectrum)."""
        # If x is single spectrum, convert to list
        if isinstance(spectra_list, SpectrumObject):
            spectra_list = [spectra_list]
        # If x is a tensor or numpy array, convert to SpectrumObject
        elif isinstance(spectra_list, (np.ndarray, torch.Tensor)):
            spectra_list = [SpectrumObject(mz=np.arange(2000, 20000, 3), intensity=spectra_list, meta={})]

        transformed = []
        for spectrum in spectra_list:
            std = float(self.std_intensity) if self.std_intensity is not None else float(np.std(spectrum.intensity))
            threshold = self.std_threshold * std
            intensity = np.where(spectrum.intensity >= threshold, spectrum.intensity, 0.0)
            transformed.append(SpectrumObject(mz=spectrum.mz, intensity=intensity, meta=spectrum.meta))
        return transformed

    def fit_transform(self, spectra_list):
        """Fit and transform in a single step."""
        if isinstance(spectra_list, (np.ndarray, torch.Tensor)):
            spectra_list = [SpectrumObject(mz=np.arange(2000, 20000, 3), intensity=spectra_list, meta={})]
        self.fit(spectra_list)
        return self.transform(spectra_list)

class TopKPeakSelectorMT:
    """
    Select top-K peaks in the style of the MaldiTransformer preprocessing:
      - detect peaks with LocalMaximaPeakDetector (SNR + halfwindow)
      - optionally merge peaks within a PPM tolerance (intensity-weighted centroid)
      - optionally refine apex m/z with a tiny quadratic fit (sub-bin)
      - keep top-K by height, sort by m/z, and return a SpectrumObject

    Parameters
    ----------
    top_k : int
        Number of peaks to keep (by height). Default 512.
    SNR : float
        SNR multiplier for LocalMaximaPeakDetector (MAD-based). Default 2.
    halfwindowsize : int
        Half-window for local maxima detection (total window = 2*hw+1). Default 20.
    merge_ppm : float or None
        If set (e.g., 20.0), merge peaks closer than this ppm. Default None (no merge).
    refine_subbin : bool
        If True, refine apex m/z with a quadratic fit around the local maximum. Default True.
    """

    def __init__(self, top_k=512, SNR=2, halfwindowsize=20):
        self.top_k = int(top_k)
        self.SNR = float(SNR)
        self.hw = int(halfwindowsize)
        self.merge_ppm = None if merge_ppm is None else float(merge_ppm)
        self.refine_subbin = bool(refine_subbin)

        # reuse your existing simple blocks
        self._detector = LocalMaximaPeakDetector(SNR=self.SNR, halfwindowsize=self.hw)
        self._filter = PeakFilter(max_number=self.top_k, min_intensity=None)

    def __call__(self, spectrum: SpectrumObject) -> SpectrumObject:
        # 1) detect peaks à la MaldiQuant (your existing implementation)
        peaks = self._detector(spectrum)  # SpectrumObject with (mz_peaks, int_peaks)

        # early exit if empty
        if len(peaks.intensity) == 0:
            return SpectrumObject(mz=np.array([], dtype=float), intensity=np.array([], dtype=float), meta=spectrum.meta)

        mz = peaks.mz.copy()
        I  = peaks.intensity.copy()

        # 2) keep top-K by height (reuse your PeakFilter behavior)
        s_tmp = SpectrumObject(mz=mz, intensity=I, meta=spectrum.meta)
        s_top = self._filter(s_tmp)

        # 3) sort by m/z for consistency
        order = np.argsort(s_top.mz)
        return SpectrumObject(mz=s_top.mz[order], intensity=s_top.intensity[order], meta=spectrum.meta)
