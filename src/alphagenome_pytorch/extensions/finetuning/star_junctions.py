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
    """Compute splice-site usage for a single position.

    Usage = reads at this site / total reads across all junctions on the same
    chrom/strand.  The denominator is the sum of all junction read counts on
    that chrom/strand, so values are in [0, 1] and reflect the relative
    importance of this site compared to all splicing activity nearby.

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

    site_reads = float(all_junctions.loc[donor_mask | acceptor_mask, "count"].sum())
    total_reads = float(all_junctions.loc[chrom_mask, "count"].sum())
    usage = 0.0 if total_reads == 0 else site_reads / total_reads

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


# 5-class label indices matching the JAX SpliceSitesClassificationHead
_CLASS_MAP = {
    ("donor", "+"): 0,   # Donor+
    ("acceptor", "+"): 1, # Acceptor+
    ("donor", "-"): 2,   # Donor-
    ("acceptor", "-"): 3, # Acceptor-
}
_CLASS_NONE = 4


def junctions_to_classification_array(
    all_juncs_list: list[pd.DataFrame],
    chrom: str,
    start: int,
    seq_len: int,
) -> np.ndarray:
    """Build a 5-class one-hot classification array over the sequence window.

    Takes the union of splice sites across all junction DataFrames (samples)
    and assigns each position to one of five classes:
        0: Donor on + strand
        1: Acceptor on + strand
        2: Donor on - strand
        3: Acceptor on - strand
        4: None (background)

    Args:
        all_juncs_list: List of junction DataFrames (one per sample), each
            having columns: chrom, exon_start (1-based), exon_end (1-based),
            strand, count.
        chrom: Chromosome name of the window.
        start: 0-based genomic start of the window.
        seq_len: Length of the window in base pairs.

    Returns:
        Float32 array of shape (seq_len, 5) — one-hot over the 5 classes.
    """
    arr = np.zeros((seq_len, 5), dtype=np.float32)
    arr[:, _CLASS_NONE] = 1.0  # default all positions to None

    # Collect all unique sites across samples
    end = start + seq_len  # 0-based exclusive
    rows = []
    for junc_df in all_juncs_list:
        mask = junc_df["chrom"] == chrom
        local = junc_df.loc[mask]
        if local.empty:
            continue
        # Donors: exon_start is 1-based → 0-based index = exon_start - 1 - start
        donors = local[["strand", "exon_start"]].copy()
        donors["role"] = "donor"
        donors = donors.rename(columns={"exon_start": "pos1based"})
        # Acceptors: exon_end is 1-based
        acceptors = local[["strand", "exon_end"]].copy()
        acceptors["role"] = "acceptor"
        acceptors = acceptors.rename(columns={"exon_end": "pos1based"})
        rows.extend([donors, acceptors])

    if not rows:
        return arr

    sites = pd.concat(rows, ignore_index=True).drop_duplicates(
        subset=["strand", "pos1based", "role"]
    )

    for _, site in sites.iterrows():
        idx = int(site["pos1based"]) - 1 - start  # convert 1-based to 0-based relative
        if 0 <= idx < seq_len:
            cls = _CLASS_MAP.get((site["role"], site["strand"]))
            if cls is not None:
                arr[idx, _CLASS_NONE] = 0.0
                arr[idx, cls] = 1.0

    return arr


def junctions_to_junction_matrix(
    all_juncs_list: list[pd.DataFrame],
    cls_arr: np.ndarray,
    chrom: str,
    start: int,
    seq_len: int,
    max_splice_sites: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a sparse donor×acceptor count matrix for a genomic window.

    Selects the top ``max_splice_sites`` donor and acceptor positions (separately
    per strand) from ``cls_arr`` — the 5-class classification array already
    computed for this window.  For every STAR junction whose donor and acceptor
    both fall within the selected positions, the read count is placed at the
    corresponding matrix entry.

    Args:
        all_juncs_list: List of junction DataFrames (one per sample), each with
            columns: chrom, exon_start (1-based), exon_end (1-based), strand, count.
        cls_arr: Float32 array of shape ``(seq_len, 5)`` — the 5-class one-hot
            classification array (Donor+, Acceptor+, Donor−, Acceptor−, None).
        chrom: Chromosome name of the window.
        start: 0-based genomic start of the window.
        seq_len: Length of the window in base pairs.
        max_splice_sites: Maximum number of splice sites per role (padded with -1).

    Returns:
        positions: int32 array of shape ``(4, max_splice_sites)`` — relative
            0-based positions for [pos_donors, pos_acceptors, neg_donors,
            neg_acceptors], padded with ``-1``.
        matrix: float32 array of shape
            ``(max_splice_sites, max_splice_sites, 2 * n_samples)`` —
            ``matrix[d, a, s]`` is the read count for the positive-strand
            junction from ``pos_donors[d]`` to ``pos_acceptors[a]`` in sample
            ``s``.  The second half of the last dimension (indices
            ``n_samples … 2*n_samples-1``) holds the same for the negative
            strand using ``neg_donors`` and ``neg_acceptors``.
    """
    n_samples = len(all_juncs_list)

    # Extract positions for each role from cls_arr columns
    # 0=Donor+, 1=Acceptor+, 2=Donor-, 3=Acceptor-
    pos_donor_pos = np.where(cls_arr[:, 0] > 0)[0][:max_splice_sites]
    pos_accept_pos = np.where(cls_arr[:, 1] > 0)[0][:max_splice_sites]
    neg_donor_pos = np.where(cls_arr[:, 2] > 0)[0][:max_splice_sites]
    neg_accept_pos = np.where(cls_arr[:, 3] > 0)[0][:max_splice_sites]

    def _pad(arr: np.ndarray) -> np.ndarray:
        out = np.full(max_splice_sites, -1, dtype=np.int32)
        out[: len(arr)] = arr
        return out

    positions = np.stack([
        _pad(pos_donor_pos),
        _pad(pos_accept_pos),
        _pad(neg_donor_pos),
        _pad(neg_accept_pos),
    ])  # (4, max_splice_sites)

    matrix = np.zeros(
        (max_splice_sites, max_splice_sites, 2 * n_samples), dtype=np.float32
    )

    # Build reverse-lookup maps: 0-based relative position → index in positions array
    pos_donor_map = {int(p): i for i, p in enumerate(pos_donor_pos)}
    pos_accept_map = {int(p): i for i, p in enumerate(pos_accept_pos)}
    neg_donor_map = {int(p): i for i, p in enumerate(neg_donor_pos)}
    neg_accept_map = {int(p): i for i, p in enumerate(neg_accept_pos)}

    end = start + seq_len

    for s, junc_df in enumerate(all_juncs_list):
        mask = (
            (junc_df["chrom"] == chrom)
            & (junc_df["exon_start"] > start)   # 1-based exon_start > start → ≥ start+1
            & (junc_df["exon_start"] <= end)
            & (junc_df["exon_end"] > start)
            & (junc_df["exon_end"] <= end)
        )
        local = junc_df.loc[mask]
        if local.empty:
            continue

        for _, junc in local.iterrows():
            # 1-based exon coords → 0-based relative index
            d_rel = int(junc["exon_start"]) - 1 - start
            a_rel = int(junc["exon_end"]) - 1 - start
            count = float(junc["count"])
            strand = junc["strand"]

            if strand == "+":
                d_idx = pos_donor_map.get(d_rel)
                a_idx = pos_accept_map.get(a_rel)
                if d_idx is not None and a_idx is not None:
                    matrix[d_idx, a_idx, s] += count
            elif strand == "-":
                d_idx = neg_donor_map.get(d_rel)
                a_idx = neg_accept_map.get(a_rel)
                if d_idx is not None and a_idx is not None:
                    matrix[d_idx, a_idx, n_samples + s] += count

    return positions, matrix


def junctions_to_usage_array(
    junc_df: pd.DataFrame,
    chrom: str,
    start: int,
    seq_len: int,
) -> np.ndarray:
    """Build a per-position usage array for one sample within a window.

    Each splice site position receives the fraction of total junction reads
    (within the window) that pass through that site.  A junction contributes
    its fractional count to both its donor and acceptor positions, so the
    array values sum to ≤ 2.0 and each individual value is in [0, 1].

    Args:
        junc_df: Junction DataFrame for one sample with columns: chrom,
            exon_start (1-based), exon_end (1-based), strand, count.
        chrom: Chromosome name of the window.
        start: 0-based genomic start of the window.
        seq_len: Length of the window in base pairs.

    Returns:
        Float32 array of shape (seq_len,).
    """
    arr = np.zeros(seq_len, dtype=np.float32)
    end = start + seq_len  # 0-based exclusive

    mask = (
        (junc_df["chrom"] == chrom)
        & (junc_df["exon_start"] > start)   # exon_start is 1-based; >start ≡ ≥start+1
        & (junc_df["exon_start"] <= end)
        & (junc_df["exon_end"] > start)
        & (junc_df["exon_end"] <= end)
    )
    local = junc_df.loc[mask]
    if local.empty:
        return arr

    total = float(local["count"].sum())
    if total == 0:
        return arr

    for _, junc in local.iterrows():
        frac = junc["count"] / total
        donor_idx = int(junc["exon_start"]) - 1 - start   # 1-based → 0-based relative
        accept_idx = int(junc["exon_end"]) - 1 - start
        if 0 <= donor_idx < seq_len:
            arr[donor_idx] += frac
        if 0 <= accept_idx < seq_len:
            arr[accept_idx] += frac

    return arr
