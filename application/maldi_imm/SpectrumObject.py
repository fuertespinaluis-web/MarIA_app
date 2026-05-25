import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SpectrumObject:
    """Base Spectrum Object class.

    Can be instantiated directly with 1-D np.arrays for mz and intensity.
    Optionally supports a meta dictionary for metadata (hospital, species, etc).

    Parameters
    ----------
    mz : 1-D np.array, optional
        mz values, by default None
    intensity : 1-D np.array, optional
        intensity values, by default None
    meta : dict, optional
        sample-level metadata (hospital, code, species...), by default None
    """

    def __init__(self, mz=None, intensity=None, meta=None):
        self.mz = np.asarray(mz, dtype=np.float32) if mz is not None else None
        self.intensity = np.asarray(intensity, dtype=np.float32) if intensity is not None else None
        self.meta = meta if meta is not None else {}
        self.max = None if self.intensity is None else float(np.max(self.intensity))
        self.min = None if self.intensity is None else float(np.min(self.intensity))

    def __getitem__(self, index):
        return SpectrumObject(mz=self.mz[index], intensity=self.intensity[index], meta=self.meta.copy())

    def __len__(self):
        return 0 if self.mz is None else self.mz.shape[0]

    def plot(self, as_peaks=False, **kwargs):
        """Plot a spectrum via matplotlib

        Parameters
        ----------
        as_peaks : bool, optional
            draw points as peaks, else connect the spectrum, by default False
        """
        if self.mz is None or self.intensity is None:
            print("Nothing to plot.")
            return
        if as_peaks:
            mz_plot = np.stack([self.mz - 1, self.mz, self.mz + 1]).T.reshape(-1)
            int_plot = np.stack([
                np.zeros_like(self.intensity),
                self.intensity,
                np.zeros_like(self.intensity)
            ]).T.reshape(-1)
        else:
            mz_plot, int_plot = self.mz, self.intensity
        plt.plot(mz_plot, int_plot, **kwargs)
        plt.xlabel('m/z')
        plt.ylabel('Intensity')
        plt.title(self.meta.get("code", "Spectrum"))

    def __repr__(self):
        summary = f"SpectrumObject(mz={self.mz[:2]}...{self.mz[-2:] if len(self.mz) > 2 else ''}, " \
                  f"intensity={self.intensity[:2]}...{self.intensity[-2:] if len(self.intensity) > 2 else ''}, " \
                  f"meta={self.meta})"
        return summary

    @staticmethod
    def tof2mass(ML1, ML2, ML3, TOF):
        A = ML3
        B = np.sqrt(1e12 / ML1)
        C = ML2 - TOF
        if A == 0:
            return (C * C) / (B * B)
        else:
            return ((-B + np.sqrt((B * B) - (4 * A * C))) / (2 * A)) ** 2

    @classmethod
    def from_bruker(cls, acqu_file, fid_file):
        """Read a spectrum from Bruker's format

        Parameters
        ----------
        acqu_file : str
            "acqu" file bruker folder
        fid_file : str
            "fid" file in bruker folder

        Returns
        -------
        SpectrumObject
        """
        with open(acqu_file, "rb") as f:
            lines = [line.decode("utf-8", errors="replace").rstrip() for line in f]
        for l in lines:
            if l.startswith("##$TD"):
                TD = int(l.split("= ")[1])
            if l.startswith("##$DELAY"):
                DELAY = int(l.split("= ")[1])
            if l.startswith("##$DW"):
                DW = float(l.split("= ")[1])
            if l.startswith("##$ML1"):
                ML1 = float(l.split("= ")[1])
            if l.startswith("##$ML2"):
                ML2 = float(l.split("= ")[1])
            if l.startswith("##$ML3"):
                ML3 = float(l.split("= ")[1])
            if l.startswith("##$BYTORDA"):
                BYTORDA = int(l.split("= ")[1])
            if l.startswith("##$NTBCal"):
                NTBCal = l.split("= ")[1]

        intensity = np.fromfile(fid_file, dtype={0: "<i", 1: ">i"}[BYTORDA])

        if len(intensity) < TD:
            TD = len(intensity)
        TOF = DELAY + np.arange(TD) * DW

        mass = cls.tof2mass(ML1, ML2, ML3, TOF)

        intensity[intensity < 0] = 0

        return cls(mz=mass, intensity=intensity)

    @classmethod
    def from_tsv(cls, file, sep=" "):
        s = pd.read_table(
            file, sep=sep, index_col=None, comment="#", header=None
        ).values
        mz = s[:, 0]
        intensity = s[:, 1]
        return cls(mz=mz, intensity=intensity)

    @classmethod
    def from_dataframe_row(cls, row, meta_fields=None):
        """
        Build SpectrumObject from a pandas DataFrame row (e.g., from parquet).

        Parameters
        ----------
        row : pd.Series
            DataFrame row with 'mz' and 'intensity' (should be arrays or lists).
        meta_fields : list, optional
            List of column names to add as metadata, by default all except mz/intensity.
        """
        mz = np.array(row['mz'])
        intensity = np.array(row['intensity'])
        if meta_fields is None:
            meta_fields = [k for k in row.index if k not in ['mz', 'intensity']]
        meta = {k: row[k] for k in meta_fields}
        return cls(mz=mz, intensity=intensity, meta=meta)
