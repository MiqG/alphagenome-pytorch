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


def gtf_splice_sites_to_junctions(gtf_file: str) -> pd.DataFrame:
    """Extract canonical splice junctions from a GENCODE GTF or parquet file.

    Junctions are derived from consecutive exon pairs within each transcript.
    Coordinate convention matches STAR output used by this module:
        exon_start (1-based) = last base of upstream exon
        exon_end   (1-based) = first base of downstream exon

    Returned rows have count=0 (annotation-only, no RNA-seq evidence).

    Args:
        gtf_file: Path to a GTF (.gtf / .gtf.gz) or parquet file.

    Returns:
        DataFrame with columns: chrom, exon_start, exon_end, strand, count.
    """
    if gtf_file.endswith(".parquet"):
        df = pd.read_parquet(gtf_file)
        # Normalise column names from pyranges-style parquet
        df = df.rename(columns={"Chromosome": "chrom", "Start": "start", "End": "end",
                                 "Strand": "strand", "Feature": "feature",
                                 "transcript_id": "transcript_id"})
        exons = df.loc[
            df["feature"].str.lower() == "exon",
            ["chrom", "start", "end", "strand", "transcript_id"],
        ].copy()
    else:
        import pyranges as pr  # type: ignore
        gr = pr.read_gtf(gtf_file)
        exons = gr.df.rename(columns={"Chromosome": "chrom", "Start": "start",
                                       "End": "end", "Strand": "strand",
                                       "Feature": "feature",
                                       "transcript_id": "transcript_id"})
        exons = exons.loc[
            exons["feature"].str.lower() == "exon",
            ["chrom", "start", "end", "strand", "transcript_id"],
        ].copy()

    exons = exons.loc[exons["strand"].isin(["+", "-"])].copy()
    exons["start"] = exons["start"].astype(int)
    exons["end"] = exons["end"].astype(int)

    # Vectorized junction extraction: sort by transcript+start, then pair consecutive
    # exons within the same transcript using shift(-1).
    exons = exons.sort_values(["transcript_id", "start"]).reset_index(drop=True)
    nxt = exons.shift(-1)
    same_tx = exons["transcript_id"].to_numpy() == nxt["transcript_id"].to_numpy()

    pairs = exons[same_tx][["chrom", "end", "strand"]].copy()
    pairs["exon_start"] = pairs["end"].astype(int)
    pairs["exon_end"] = nxt.loc[same_tx, "start"].astype(int) + 1
    pairs = pairs.drop(columns="end")

    result = pairs.drop_duplicates(subset=["chrom", "exon_start", "exon_end", "strand"])
    result["count"] = 0
    return result.reset_index(drop=True)


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


def normalize_junctions_to_cpm(junc_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize junction counts to CPM (counts per million) within a sample.

    Divides each junction's count by the total junction reads in the sample,
    scaled to 1 million.

    Args:
        junc_df: Junction DataFrame with 'count' column.

    Returns:
        DataFrame with counts normalized to CPM.
    """
    df = junc_df.copy()
    total_reads = df["count"].sum()
    if total_reads > 0:
        df["count"] = (df["count"] / total_reads) * 1e6
    else:
        df["count"] = 0.0
    return df


def normalize_junctions_per_sample(junc_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize junction counts within a single sample (full pipeline).

    Applies three normalization steps:
    1. CPM normalize: counts per million total filtered reads
    2. Clip at 99.99th percentile: removes extreme outliers
    3. Scale by mean of nonzero values: centers scale around expressed junctions

    This matches the AlphaGenome paper's preprocessing for fine-tuning.

    Args:
        junc_df: Junction DataFrame with 'count' column.

    Returns:
        Normalized DataFrame with same structure.

    Example:
        >>> junc = read_star_junctions('sample.sj.out.tab')
        >>> junc = junc.loc[junc['n_uniquely_mapped_reads'] >= 1].copy()
        >>> junc['count'] = junc['n_uniquely_mapped_reads']
        >>> normalized = normalize_junctions_per_sample(junc)
    """
    df = junc_df.copy()

    # Step 1: CPM normalize
    total_reads = df["count"].sum()
    if total_reads > 0:
        df["count"] = (df["count"] / total_reads) * 1e6
    else:
        return df  # All zeros, return as-is

    # Step 2: Clip at 99.99th percentile
    threshold = float(np.percentile(df["count"], 99.99))
    df["count"] = np.minimum(df["count"], threshold)

    # Step 3: Scale by mean of nonzero values
    nonzero = df.loc[df["count"] > 0, "count"]
    if len(nonzero) > 0:
        mean_val = float(nonzero.mean())
        if mean_val > 0:
            df["count"] = df["count"] / mean_val

    return df


def compute_tissue_normalization_params(
    junc_dfs_by_sample: dict[str, pd.DataFrame],
    sample_to_tissue: dict[str, str],
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute tissue-specific clipping threshold and scaling mean.

    For each tissue, computes:
    1. 99.99th percentile of all junction counts in that tissue
    2. Mean of clipped, nonzero junction counts in that tissue

    Args:
        junc_dfs_by_sample: Dict mapping sample name to junction DataFrame
            (should be CPM-normalized already).
        sample_to_tissue: Dict mapping sample name to tissue label.

    Returns:
        (thresholds, means) tuple where:
        - thresholds: Dict mapping tissue to 99.99th percentile value
        - means: Dict mapping tissue to mean of nonzero clipped counts
    """
    # Group samples by tissue
    tissue_samples: dict[str, list[str]] = {}
    for sample, tissue in sample_to_tissue.items():
        if tissue not in tissue_samples:
            tissue_samples[tissue] = []
        tissue_samples[tissue].append(sample)

    thresholds: dict[str, float] = {}
    means: dict[str, float] = {}

    for tissue, samples in tissue_samples.items():
        # Collect all counts in this tissue
        all_counts: list[float] = []
        for sample in samples:
            if sample in junc_dfs_by_sample:
                all_counts.extend(junc_dfs_by_sample[sample]["count"].values)

        if not all_counts:
            continue

        all_counts_arr = np.asarray(all_counts, dtype=np.float32)

        # Compute 99.99th percentile
        threshold = float(np.percentile(all_counts_arr, 99.99))
        thresholds[tissue] = threshold

        # Compute mean of clipped, nonzero values
        clipped_counts = np.minimum(all_counts_arr, threshold)
        nonzero_clipped = clipped_counts[clipped_counts > 0]
        if len(nonzero_clipped) > 0:
            means[tissue] = float(np.mean(nonzero_clipped))
        else:
            means[tissue] = 1.0  # Fallback if all zero

    return thresholds, means


def normalize_junctions_tissue_level(
    junc_dfs_by_sample: dict[str, pd.DataFrame],
    sample_to_tissue: dict[str, str],
    thresholds: dict[str, float] | None = None,
    means: dict[str, float] | None = None,
) -> dict[str, pd.DataFrame]:
    """Apply tissue-specific clipping and scaling to junction counts.

    For each tissue:
    1. Clips counts at 99.99th percentile
    2. Scales by dividing by mean of expressed junctions (count > 0)

    Args:
        junc_dfs_by_sample: Dict mapping sample name to junction DataFrame
            (should be CPM-normalized).
        sample_to_tissue: Dict mapping sample name to tissue label.
        thresholds: Optional pre-computed tissue thresholds (99.99th percentile).
            If None, computed from the data.
        means: Optional pre-computed tissue means. If None, computed from data.

    Returns:
        Dict mapping sample name to normalized DataFrame.

    Example:
        >>> # First normalize each sample to CPM
        >>> cpm_dfs = {s: normalize_junctions_to_cpm(df)
        ...            for s, df in junc_dfs.items()}
        >>> # Then apply tissue-level scaling
        >>> normalized = normalize_junctions_tissue_level(
        ...     cpm_dfs,
        ...     sample_to_tissue={'sample1': 'tissue_A', 'sample2': 'tissue_A'},
        ... )
    """
    if thresholds is None or means is None:
        thresholds, means = compute_tissue_normalization_params(
            junc_dfs_by_sample, sample_to_tissue
        )

    normalized: dict[str, pd.DataFrame] = {}
    for sample, junc_df in junc_dfs_by_sample.items():
        tissue = sample_to_tissue.get(sample)
        if tissue is None:
            # No tissue mapping, return as-is
            normalized[sample] = junc_df.copy()
            continue

        df = junc_df.copy()

        # Clip at tissue threshold
        threshold = thresholds.get(tissue, np.inf)
        df["count"] = np.minimum(df["count"], threshold)

        # Scale by tissue mean
        mean = means.get(tissue, 1.0)
        if mean > 0:
            df["count"] = df["count"] / mean

        normalized[sample] = df

    return normalized


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


def junctions_to_usage_arrays_by_strand(
    junc_df: pd.DataFrame,
    chrom: str,
    start: int,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build per-position usage arrays per strand for one sample within a window.

    Returns separate usage arrays for positive and negative strands.
    Each array represents the fraction of that strand's junction reads
    passing through each splice site position.

    Args:
        junc_df: Junction DataFrame with columns: chrom, exon_start (1-based),
            exon_end (1-based), strand, count.
        chrom: Chromosome name of the window.
        start: 0-based genomic start of the window.
        seq_len: Length of the window in base pairs.

    Returns:
        Tuple of two float32 arrays of shape (seq_len,):
            - pos_arr: usage for positive strand
            - neg_arr: usage for negative strand
    """
    pos_arr = np.zeros(seq_len, dtype=np.float32)
    neg_arr = np.zeros(seq_len, dtype=np.float32)
    end = start + seq_len  # 0-based exclusive

    mask = (
        (junc_df["chrom"] == chrom)
        & (junc_df["exon_start"] > start)
        & (junc_df["exon_start"] <= end)
        & (junc_df["exon_end"] > start)
        & (junc_df["exon_end"] <= end)
    )
    local = junc_df.loc[mask]
    if local.empty:
        return pos_arr, neg_arr

    # Compute total reads per strand
    pos_total = float(local[local["strand"] == "+"]["count"].sum())
    neg_total = float(local[local["strand"] == "-"]["count"].sum())

    # Build usage arrays per strand
    for _, junc in local.iterrows():
        donor_idx = int(junc["exon_start"]) - 1 - start
        accept_idx = int(junc["exon_end"]) - 1 - start

        if junc["strand"] == "+":
            if pos_total > 0:
                frac = junc["count"] / pos_total
                if 0 <= donor_idx < seq_len:
                    pos_arr[donor_idx] += frac
                if 0 <= accept_idx < seq_len:
                    pos_arr[accept_idx] += frac
        elif junc["strand"] == "-":
            if neg_total > 0:
                frac = junc["count"] / neg_total
                if 0 <= donor_idx < seq_len:
                    neg_arr[donor_idx] += frac
                if 0 <= accept_idx < seq_len:
                    neg_arr[accept_idx] += frac

    return pos_arr, neg_arr

    return arr
