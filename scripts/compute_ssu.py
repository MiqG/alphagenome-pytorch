#!/usr/bin/env python
"""Compute splice site usage (SSU) from STAR junction data, optionally with a BAM.

For each splice site in a SJ.out.tab file, computes:

  SSU approx  = α / (α + β2)         [junction-only, no BAM needed]
  SSU full    = α / (α + β1 + β2)    [α/β2 from junctions, β1 from BAM]
  SSU spliser = α / (α + β1 + β2)    [all counts from BAM, equivalent to SpliSER]

where:
  α  = split reads using this site
  β1 = reads spanning the site continuously without splicing (no N CIGAR)
  β2 = reads using a competing site for the same partner

Coordinates are 1-based.  Each splice site is reported with both its exonic
coordinate (last exon base for donor, first exon base for acceptor) and its
intronic coordinate (first intron base for donor, last intron base for acceptor).

Usage:
    # Junction-only approximation
    python scripts/compute_ssu.py \\
        --junctions second_pass.SJ.out.tab \\
        --output ssu.parquet

    # Full SSU with BAM ground truth (3 metrics)
    python scripts/compute_ssu.py \\
        --junctions second_pass.SJ.out.tab \\
        --bam second_pass.Aligned.sortedByCoord.out.filtered.bam \\
        --output ssu.parquet
"""

from __future__ import annotations

import argparse
import bisect
import sys
from pathlib import Path

# Allow importing from the alphagenome-pytorch package bundled in this repo
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from alphagenome_pytorch.extensions.finetuning.star_junctions import read_star_junctions


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute splice site usage (SSU) from STAR junction data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    io = p.add_argument_group("Input / output")
    io.add_argument("--junctions", "-j", required=True,
                    help="STAR SJ.out.tab file")
    io.add_argument("--bam", "-b", default=None,
                    help="Coordinate-sorted, indexed BAM (enables SSU full)")
    io.add_argument("--output", "-o", required=True,
                    help="Output Parquet file path")

    filt = p.add_argument_group("Filtering")
    filt.add_argument("--min-unique-reads", type=int, default=1,
                      help="Minimum uniquely mapped reads to retain a junction (default: 1)")
    filt.add_argument("--mapq", type=int, default=30,
                      help="Minimum MAPQ for β1 BAM reads (default: 30)")

    out = p.add_argument_group("Output")
    out.add_argument("--compression", "-c",
                     choices=["snappy", "gzip", "zstd", "none"],
                     default="zstd",
                     help="Parquet compression codec (default: zstd)")

    return p.parse_args()


# ------------------------------------------------------------------ #
# Step 1: load and filter junctions
# ------------------------------------------------------------------ #

def load_junctions(path: str | Path, min_unique_reads: int) -> "pd.DataFrame":
    """Read and quality-filter a STAR SJ.out.tab file.

    Args:
        path: Path to SJ.out.tab.
        min_unique_reads: Minimum n_uniquely_mapped_reads threshold.

    Returns:
        DataFrame with added columns exon_start, exon_end, count.
    """
    import pandas as pd

    junctions = read_star_junctions(str(path))
    junctions = junctions.loc[
        (junctions["n_uniquely_mapped_reads"] >= min_unique_reads)
        & (junctions["strand"].isin(["+", "-"]))
    ].copy()
    junctions["exon_start"] = junctions["intron_start"] - 1  # last exon base (1-based)
    junctions["exon_end"]   = junctions["intron_end"]   + 1  # first exon base (1-based)
    junctions["count"]      = junctions["n_uniquely_mapped_reads"]
    return junctions.reset_index(drop=True)


# ------------------------------------------------------------------ #
# Step 2: α and β2 from junction data
# ------------------------------------------------------------------ #

def compute_alpha_beta2(
    junctions: "pd.DataFrame",
) -> "tuple[pd.Series, pd.Series, pd.Series, pd.Series]":
    """Compute per-site α and β2 from junction counts.

    β2(D) = Σ_{A: D→A} acceptor_total(A) − α(D)
    β2(A) = Σ_{D: D→A} donor_total(D) − α(A)

    Fully vectorized (no iterrows).

    Args:
        junctions: DataFrame with exon_start, exon_end, strand, chrom, count.

    Returns:
        (donor_alpha, acceptor_alpha, donor_beta2, acceptor_beta2) as Series
        indexed by (chrom, 1-based position, strand).
    """
    donor_alpha = (
        junctions.groupby(["chrom", "exon_start", "strand"])["count"].sum()
        .rename("donor_total")
    )
    acceptor_alpha = (
        junctions.groupby(["chrom", "exon_end", "strand"])["count"].sum()
        .rename("acceptor_total")
    )

    j = junctions.join(acceptor_alpha, on=["chrom", "exon_end", "strand"])
    j = j.join(donor_alpha,            on=["chrom", "exon_start", "strand"])

    donor_beta2 = (
        j.groupby(["chrom", "exon_start", "strand"])["acceptor_total"].sum()
        - donor_alpha
    ).rename("donor_beta2")
    acceptor_beta2 = (
        j.groupby(["chrom", "exon_end", "strand"])["donor_total"].sum()
        - acceptor_alpha
    ).rename("acceptor_beta2")

    return donor_alpha, acceptor_alpha, donor_beta2, acceptor_beta2


# ------------------------------------------------------------------ #
# Step 3: β1 from BAM (optional)
# ------------------------------------------------------------------ #

def build_beta1_counts(
    bam_path: str | Path,
    junctions: "pd.DataFrame",
    mapq_min: int = 30,
) -> "dict[tuple[str, int], int]":
    """Count unspliced reads spanning each splice site (β1).

    Reads the BAM once per chromosome.  A read contributes β1 at a site when
    it overlaps the site, has MAPQ >= mapq_min, is not a duplicate, has no
    N CIGAR operation covering the position, and matches the site's strand
    (via XS tag).

    Args:
        bam_path: Path to coordinate-sorted, indexed BAM.
        junctions: Filtered junctions DataFrame (used to derive site positions).
        mapq_min: Minimum MAPQ filter.

    Returns:
        Dict mapping (chrom, 0-based position) to β1 count.
    """
    try:
        import pysam
    except ImportError as e:
        raise ImportError("pysam is required for β1 computation (--bam mode)") from e

    # Collect all sites: {chrom → {0-based pos → set of strands}}
    sites_by_chrom: dict[str, dict[int, set[str]]] = {}
    for _, row in junctions[["chrom", "exon_start", "exon_end", "strand"]].drop_duplicates().iterrows():
        chrom, strand = row["chrom"], row["strand"]
        for pos_1based in (row["exon_start"], row["exon_end"]):
            p0 = int(pos_1based) - 1
            sites_by_chrom.setdefault(chrom, {}).setdefault(p0, set()).add(strand)

    beta1: dict[tuple[str, int], int] = {
        (chrom, pos): 0
        for chrom, sites in sites_by_chrom.items()
        for pos in sites
    }

    bam = pysam.AlignmentFile(str(bam_path), "rb")
    try:
        for chrom, site_strands in sites_by_chrom.items():
            sites_sorted = sorted(site_strands)
            chrom_start  = sites_sorted[0]
            chrom_end    = sites_sorted[-1] + 1

            for read in bam.fetch(chrom, chrom_start, chrom_end):
                if read.is_unmapped or read.is_duplicate:
                    continue
                if read.mapping_quality < mapq_min:
                    continue
                if not read.cigartuples:
                    continue

                try:
                    read_strand = read.get_tag("XS")
                except KeyError:
                    read_strand = None

                introns: list[tuple[int, int]] = []
                ref_pos = read.reference_start
                for op, length in read.cigartuples:
                    if op == 3:
                        introns.append((ref_pos, ref_pos + length))
                        ref_pos += length
                    elif op in (0, 2, 7, 8):
                        ref_pos += length

                read_start = read.reference_start
                read_end   = read.reference_end

                lo = bisect.bisect_left(sites_sorted, read_start)
                hi = bisect.bisect_right(sites_sorted, read_end - 1)

                for site_pos in sites_sorted[lo:hi]:
                    if read_strand is not None:
                        if read_strand not in site_strands[site_pos]:
                            continue
                    if not any(iv_s <= site_pos < iv_e for iv_s, iv_e in introns):
                        beta1[(chrom, site_pos)] += 1
    finally:
        bam.close()

    return beta1


# ------------------------------------------------------------------ #
# Step 3b: SpliSER-equivalent counts from BAM (all α, β1, β2 from BAM)
# ------------------------------------------------------------------ #

def _check_strand_from_flag(flag: int, strandedType: str = "rf") -> str | None:
    """Determine transcript strand from SAM flag bits (mirrors SpliSER check_strand)."""
    is_paired   = bool(flag & 0x1)
    is_reverse  = bool(flag & 0x10)
    is_read1    = bool(flag & 0x40)

    if not is_paired:
        mate = 1
    elif is_read1:
        mate = 1
    else:
        mate = 2

    if strandedType == "rf":
        if mate == 1:
            return "+" if is_reverse else "-"
        else:
            return "-" if is_reverse else "+"
    elif strandedType == "fr":
        if mate == 1:
            return "-" if is_reverse else "+"
        else:
            return "+" if is_reverse else "-"
    return None


def compute_spliser_counts(
    bam_path: str | Path,
    junctions: "pd.DataFrame",
    mapq_min: int = 30,
    strandedType: str = "rf",
) -> "pd.DataFrame":
    """Compute SpliSER-equivalent α, β1, β2 for all splice sites from BAM.

    Single BAM pass using find_introns.  Returns DataFrame with columns:
    chrom, position, strand, role, alpha_bam, beta1_bam, beta2_bam, ssu_spliser.
    Position uses 1-based exon convention (last exon base for donors,
    first exon base for acceptors).
    """
    try:
        import pysam
    except ImportError as e:
        raise ImportError("pysam is required for SpliSER computation") from e

    import pandas as pd

    bam = pysam.AlignmentFile(str(bam_path), "rb")

    donor_alpha_bam:    dict[tuple[str, int, str], int] = {}
    acceptor_alpha_bam: dict[tuple[str, int, str], int] = {}

    chroms = junctions["chrom"].unique()

    for chrom in chroms:
        for strand in ("+", "-"):
            gen = (
                r for r in bam.fetch(chrom)
                if not r.is_unmapped
                and not r.is_secondary
                and not r.is_supplementary
                and r.mapping_quality >= mapq_min
                and _check_strand_from_flag(r.flag, strandedType) == strand
            )
            for (iv_s, iv_e), count in bam.find_introns(gen).items():
                donor_alpha_bam[(chrom, iv_s, strand)]    = donor_alpha_bam.get((chrom, iv_s, strand), 0)    + count
                acceptor_alpha_bam[(chrom, iv_e, strand)] = acceptor_alpha_bam.get((chrom, iv_e, strand), 0) + count

    all_targets: dict[str, list[int]] = {}
    target_roles: dict[str, dict[int, list[str]]] = {}
    acceptor_scan_to_alpha: dict[tuple[int, str], int] = {}
    for strand in ("+", "-"):
        pos_set: dict[int, list[str]] = {}
        for (_, pos, s) in donor_alpha_bam:
            if s == strand:
                pos_set.setdefault(pos, []).append("donor")
        for (_, pos, s) in acceptor_alpha_bam:
            if s == strand:
                pos_set.setdefault(pos - 1, []).append("acceptor")
        all_targets[strand] = sorted(pos_set)
        target_roles[strand] = pos_set

    beta1_bam: dict[tuple[int, str, str], int] = {}
    beta2_bam: dict[tuple[int, str, str], int] = {}

    for chrom in chroms:
        for read in bam.fetch(chrom):
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            if read.mapping_quality < mapq_min:
                continue
            if not read.cigartuples:
                continue

            read_strand = _check_strand_from_flag(read.flag, strandedType)
            if read_strand not in ("+", "-"):
                continue

            introns: list[tuple[int, int]] = []
            ref_pos = read.reference_start
            for op, length in read.cigartuples:
                if op == 3:
                    introns.append((ref_pos, ref_pos + length))
                    ref_pos += length
                elif op in (0, 2, 7, 8):
                    ref_pos += length

            read_start = read.reference_start
            read_end   = read.reference_end

            targets = all_targets.get(read_strand, [])
            lo = bisect.bisect_left(targets, read_start)
            hi = bisect.bisect_right(targets, read_end - 1)

            for target_pos in targets[lo:hi]:
                for role in target_roles[read_strand][target_pos]:
                    key = (target_pos, read_strand, role)

                    if not any(iv_s <= target_pos < iv_e for iv_s, iv_e in introns):
                        beta1_bam[key] = beta1_bam.get(key, 0) + 1

                    if role == "acceptor":
                        is_alpha = any(iv_e_r == target_pos + 1 for _, iv_e_r in introns)
                        if not is_alpha and any(iv_s < target_pos < iv_e for iv_s, iv_e in introns):
                            beta2_bam[key] = beta2_bam.get(key, 0) + 1
                    else:
                        if any(iv_s < target_pos < iv_e for iv_s, iv_e in introns):
                            beta2_bam[key] = beta2_bam.get(key, 0) + 1

    bam.close()

    rows = []
    for (chrom, pos, strand), alpha in donor_alpha_bam.items():
        key  = (pos, strand, "donor")
        b1   = beta1_bam.get(key, 0)
        b2   = beta2_bam.get(key, 0)
        denom = alpha + b1 + b2
        rows.append({
            "chrom":       chrom,
            "position":    pos,
            "strand":      strand,
            "role":        "donor",
            "alpha_bam":   int(alpha),
            "beta1_bam":   int(b1),
            "beta2_bam":   int(b2),
            "ssu_spliser": alpha / denom if denom > 0 else float("nan"),
        })
    for (chrom, pos, strand), alpha in acceptor_alpha_bam.items():
        scan_key = (pos - 1, strand, "acceptor")
        b1   = beta1_bam.get(scan_key, 0)
        b2   = beta2_bam.get(scan_key, 0)
        denom = alpha + b1 + b2
        rows.append({
            "chrom":       chrom,
            "position":    pos + 1,
            "strand":      strand,
            "role":        "acceptor",
            "alpha_bam":   int(alpha),
            "beta1_bam":   int(b1),
            "beta2_bam":   int(b2),
            "ssu_spliser": alpha / denom if denom > 0 else float("nan"),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.drop_duplicates(subset=["chrom", "position", "strand", "role"]).reset_index(drop=True)


# ------------------------------------------------------------------ #
# Step 4: assemble site table
# ------------------------------------------------------------------ #

def assemble_site_table(
    junctions: "pd.DataFrame",
    donor_alpha: "pd.Series",
    acceptor_alpha: "pd.Series",
    donor_beta2: "pd.Series",
    acceptor_beta2: "pd.Series",
    beta1_counts: "dict[tuple[str, int], int] | None",
) -> "pd.DataFrame":
    """Build one row per splice site with SSU scores.

    Args:
        junctions: Filtered junctions DataFrame (for intron coordinates).
        donor_alpha: Series indexed by (chrom, exon_start, strand).
        acceptor_alpha: Series indexed by (chrom, exon_end, strand).
        donor_beta2: Series indexed by (chrom, exon_start, strand).
        acceptor_beta2: Series indexed by (chrom, exon_end, strand).
        beta1_counts: Optional dict (chrom, 0-based pos) → count; None skips SSU full.

    Returns:
        DataFrame with columns: chrom, strand, role, exon_pos, intron_pos,
        alpha_juncs, beta2_juncs, ssu_approx, [beta1_bam, ssu_full].
    """
    import numpy as np
    import pandas as pd

    # Build a lookup from exon coord → intron coord
    donor_intron   = junctions.groupby(["chrom", "exon_start", "strand"])["intron_start"].first()
    acceptor_intron = junctions.groupby(["chrom", "exon_end",   "strand"])["intron_end"].first()

    rows: list[dict] = []

    for (chrom, exon_pos, strand), alpha in donor_alpha.items():
        intron_pos = int(donor_intron.get((chrom, exon_pos, strand), exon_pos + 1))
        b2 = int(donor_beta2.get((chrom, exon_pos, strand), 0))
        d_approx = alpha + b2
        row: dict = {
            "chrom":        chrom,
            "strand":       strand,
            "role":         "donor",
            "exon_pos":     int(exon_pos),
            "intron_pos":   intron_pos,
            "alpha_juncs":  int(alpha),
            "beta2_juncs":  b2,
            "ssu_approx":   alpha / d_approx if d_approx > 0 else float("nan"),
        }
        if beta1_counts is not None:
            b1 = beta1_counts.get((chrom, int(exon_pos) - 1), 0)
            d_full = alpha + b1 + b2
            row["beta1_bam"] = int(b1)
            row["ssu_full"]  = alpha / d_full if d_full > 0 else float("nan")
        rows.append(row)

    for (chrom, exon_pos, strand), alpha in acceptor_alpha.items():
        intron_pos = int(acceptor_intron.get((chrom, exon_pos, strand), exon_pos - 1))
        b2 = int(acceptor_beta2.get((chrom, exon_pos, strand), 0))
        d_approx = alpha + b2
        row = {
            "chrom":        chrom,
            "strand":       strand,
            "role":         "acceptor",
            "exon_pos":     int(exon_pos),
            "intron_pos":   intron_pos,
            "alpha_juncs":  int(alpha),
            "beta2_juncs":  b2,
            "ssu_approx":   alpha / d_approx if d_approx > 0 else float("nan"),
        }
        if beta1_counts is not None:
            b1 = beta1_counts.get((chrom, int(exon_pos) - 1), 0)
            d_full = alpha + b1 + b2
            row["beta1_bam"] = int(b1)
            row["ssu_full"]  = alpha / d_full if d_full > 0 else float("nan")
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.drop_duplicates(
        subset=["chrom", "strand", "role", "exon_pos"]
    ).reset_index(drop=True)


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main() -> None:
    args = parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading junctions from {args.junctions!r} …")
    junctions = load_junctions(args.junctions, args.min_unique_reads)
    print(f"  {len(junctions):,} junctions after quality filtering")

    if junctions.empty:
        print("No junctions remain after filtering. Check --min-unique-reads.", file=sys.stderr)
        sys.exit(1)

    print("Computing α and β2 …")
    donor_alpha, acceptor_alpha, donor_beta2, acceptor_beta2 = compute_alpha_beta2(junctions)
    print(f"  {len(donor_alpha):,} donor sites, {len(acceptor_alpha):,} acceptor sites")

    beta1_counts = None
    df_spliser = None
    if args.bam is not None:
        print(f"Computing β1 from BAM {args.bam!r} …")
        beta1_counts = build_beta1_counts(args.bam, junctions, args.mapq)
        total_b1 = sum(beta1_counts.values())
        print(f"  total β1 reads counted: {total_b1:,}")

        print(f"Computing SpliSER-equivalent (all counts from BAM) …")
        df_spliser = compute_spliser_counts(args.bam, junctions, args.mapq)
        print(f"  {len(df_spliser):,} sites with BAM counts")

    print("Assembling site table …")
    df = assemble_site_table(
        junctions,
        donor_alpha, acceptor_alpha,
        donor_beta2, acceptor_beta2,
        beta1_counts,
    )
    print(f"  {len(df):,} unique splice sites")

    if df_spliser is not None and not df_spliser.empty:
        df = df.merge(
            df_spliser[["chrom", "position", "strand", "role",
                        "alpha_bam", "beta1_bam", "beta2_bam", "ssu_spliser"]],
            left_on=["chrom", "exon_pos", "strand", "role"],
            right_on=["chrom", "position", "strand", "role"],
            how="left",
        ).drop(columns=["position"])
    elif args.bam is not None:
        for col in ("alpha_bam", "beta1_bam", "beta2_bam", "ssu_spliser"):
            df[col] = float("nan")

    n_b2_zero = int((df["beta2_juncs"] == 0).sum())
    print(f"  sites with β2=0 (uncontested, ssu_approx=1.0): "
          f"{n_b2_zero} ({100 * n_b2_zero / len(df):.1f}%)")

    compression_arg = None if args.compression == "none" else args.compression
    df.to_parquet(out_path, index=False, compression=compression_arg)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {out_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
