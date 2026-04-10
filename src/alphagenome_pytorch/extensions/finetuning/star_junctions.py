"""Utilities for reading and processing STAR splice junction files.

STAR outputs a file (SJ.out.tab) with one row per detected splice junction.
These utilities read, filter, and convert junction data into arrays suitable
for use as training targets.

STAR SJ.out.tab column layout:
    0: chromosome
    1: intron start (1-based)
    2: intron end (1-based)
    3: strand (0=undefined, 1=+, 2=-)
    4: intron motif
    5: annotated junction (0=novel, 1=annotated)
    6: number of uniquely mapping reads
    7: number of multi-mapping reads
    8: maximum spliced alignment overhang
"""

from __future__ import annotations

import numpy as np
import pandas as pd


_STRAND_MAP = {"0": ".", "1": "+", "2": "-"}


def read_star_junctions(path: str) -> pd.DataFrame:
    """Read a STAR SJ.out.tab file into a DataFrame.

    Args:
        path: Path to SJ.out.tab file.

    Returns:
        DataFrame with columns: chrom, intron_start, intron_end, strand,
        intron_motif, annotated, n_uniquely_mapped_reads, n_multi_mapped_reads,
        max_overhang.
    """
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=[
            "chrom",
            "intron_start",
            "intron_end",
            "strand_code",
            "intron_motif",
            "annotated",
            "n_uniquely_mapped_reads",
            "n_multi_mapped_reads",
            "max_overhang",
        ],
        dtype={
            "chrom": str,
            "intron_start": np.int64,
            "intron_end": np.int64,
            "strand_code": str,
            "intron_motif": np.int64,
            "annotated": np.int64,
            "n_uniquely_mapped_reads": np.int64,
            "n_multi_mapped_reads": np.int64,
            "max_overhang": np.int64,
        },
    )
    df["strand"] = df["strand_code"].map(_STRAND_MAP).fillna(".")
    df = df.drop(columns=["strand_code"])
    return df


def junctions_to_splice_sites(junctions: pd.DataFrame) -> pd.DataFrame:
    """Extract unique splice site positions from a junctions DataFrame.

    Each junction has a donor and an acceptor site. A site is defined by
    (chrom, position, strand, role) where role is 'donor' or 'acceptor'.
    Exon coordinates (1-based) are derived as:
        donor   = intron_start - 1  (last base of the upstream exon)
        acceptor = intron_end + 1   (first base of the downstream exon)

    Args:
        junctions: DataFrame from read_star_junctions (or a subset).

    Returns:
        DataFrame with columns: chrom, position, strand, role.
        Rows are unique splice sites.
    """
    donors = junctions[["chrom", "strand"]].copy()
    donors["position"] = junctions["intron_start"] - 1  # 1-based exon end
    donors["role"] = "donor"

    acceptors = junctions[["chrom", "strand"]].copy()
    acceptors["position"] = junctions["intron_end"] + 1  # 1-based exon start
    acceptors["role"] = "acceptor"

    sites = pd.concat([donors, acceptors], ignore_index=True)
    sites = sites.drop_duplicates(subset=["chrom", "position", "strand", "role"])
    return sites.reset_index(drop=True)


def compute_splice_site_usage(
    all_junctions: pd.DataFrame,
    chrom: str,
    position: int,
    strand: str,
) -> dict:
    """Compute splice-site usage (PSI-like) for a single position.

    Usage = reads crossing this site / total reads at this site.
    Total reads = sum of all junctions that share this donor or acceptor.

    Args:
        all_junctions: Full junctions DataFrame (with 'exon_start', 'exon_end',
            'strand', 'count' columns).
        chrom: Chromosome name.
        position: 1-based exon coordinate of the splice site.
        strand: Strand ('+' or '-').

    Returns:
        Dict with keys: position, splice_site_usage, total_reads.
    """
    chrom_mask = (all_junctions["chrom"] == chrom) & (all_junctions["strand"] == strand)

    donor_mask = chrom_mask & (all_junctions["exon_start"] == position)
    acceptor_mask = chrom_mask & (all_junctions["exon_end"] == position)

    site_reads = all_junctions.loc[donor_mask | acceptor_mask, "count"].sum()
    total_reads = float(site_reads)
    usage = 1.0 if total_reads == 0 else float(site_reads) / total_reads

    return {
        "position": position,
        "splice_site_usage": usage,
        "total_reads": total_reads,
    }


def filter_intervals_with_junctions(
    intervals: pd.DataFrame,
    junctions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only intervals that contain at least one complete splice junction.

    A junction is 'complete' within an interval if both exon_start and exon_end
    fall inside [interval.start, interval.end).

    Args:
        intervals: DataFrame with columns chrom, start, end (0-based half-open).
        junctions: DataFrame with columns chrom, exon_start, exon_end (1-based).

    Returns:
        (filtered_intervals, overlapping_junctions) - both as DataFrames.
    """
    keep_interval = []
    keep_junc = []

    for i, interval in intervals.iterrows():
        chrom, start, end = interval["chrom"], interval["start"], interval["end"]
        mask = (
            (junctions["chrom"] == chrom)
            & (junctions["exon_start"] >= start)
            & (junctions["exon_end"] < end)
        )
        hits = junctions.loc[mask]
        if len(hits) > 0:
            keep_interval.append(i)
            keep_junc.append(hits)

    if not keep_interval:
        return intervals.iloc[:0].copy(), junctions.iloc[:0].copy()

    filtered = intervals.loc[keep_interval].reset_index(drop=True)
    overlaps = pd.concat(keep_junc, ignore_index=True).drop_duplicates()
    return filtered, overlaps


def splice_sites_to_array(
    splice_sites: pd.DataFrame,
    seq_len: int,
    start: int,
) -> np.ndarray:
    """Convert splice sites to a boolean array over the sequence.

    Args:
        splice_sites: DataFrame with columns chrom, position, strand, role.
        seq_len: Length of the sequence window.
        start: 1-based genomic start of the window (positions are 1-based).

    Returns:
        Boolean array of shape (3, seq_len):
            row 0: donor sites
            row 1: acceptor sites
            row 2: any splice site (donor OR acceptor)
    """
    arr = np.zeros((3, seq_len), dtype=bool)
    if splice_sites.empty:
        return arr

    for _, site in splice_sites.iterrows():
        idx = int(site["position"]) - start  # convert to 0-based relative index
        if 0 <= idx < seq_len:
            if site["role"] == "donor":
                arr[0, idx] = True
            elif site["role"] == "acceptor":
                arr[1, idx] = True
            arr[2, idx] = True

    return arr
