# src/trnazap/cli/compare_conditions_cli.py

"""CLI subcommands for condition-level tRNA analysis and comparison."""

from ..visualize.alignment_viz.compare_conditions import (
    load_condition,
    one_condition_figures,
    compare_conditions as _compare_conditions,
)


def register_subparsers(subparsers):
    """Register compare-conditions and analyze-condition subcommands."""
    _register_compare_conditions(subparsers)
    _register_analyze_condition(subparsers)


def _register_compare_conditions(subparsers):
    parser = subparsers.add_parser(
        "compare-conditions",
        help="Compare two tRNA alignment conditions",
        description=(
            "Generate differential analysis figures between two tRNA alignment conditions. "
            "Produces per-condition figures and comparison plots (volcano, TPM scatter, etc.)."
        ),
    )

    parser.add_argument("bam1", type=str, help="BAM file for condition A")
    parser.add_argument("label1", type=str, help="Label for condition A (e.g. 'Control')")
    parser.add_argument("bam2", type=str, help="BAM file for condition B")
    parser.add_argument("label2", type=str, help="Label for condition B (e.g. 'Treatment')")

    parser.add_argument(
        "--model", "-m", type=str, required=True,
        choices=["yeast", "e_coli"],
        help="Reference model: 'yeast' or 'e_coli'",
    )
    parser.add_argument(
        "--aligner", "-a", type=str, default="zap",
        choices=["zap", "bwa"],
        help="Aligner used to produce the BAM files (default: zap)",
    )
    parser.add_argument(
        "--out_dir", "-od", type=str, required=True,
        help="Output directory for figures and tables",
    )
    parser.add_argument(
        "--out_prefix", "-op", type=str, default="",
        help="Filename prefix for all outputs (default: none)",
    )
    parser.add_argument(
        "--threads", "-t", type=int, default=8,
        help="Threads for parallel BAM loading (default: 8)",
    )
    parser.add_argument(
        "--ident_threshold", type=float, default=0.75,
        help="Minimum alignment identity to include a read (default: 0.75)",
    )
    parser.add_argument(
        "--min_coverage", type=int, default=25,
        help="Minimum alignment length to include a read (default: 25)",
    )
    parser.add_argument(
        "--fc_threshold", type=float, default=1.5,
        help="Fold-change threshold for volcano plot (default: 1.5)",
    )
    parser.add_argument(
        "--pval_threshold", type=float, default=0.05,
        help="P-value threshold for volcano plot (default: 0.05)",
    )

    parser.set_defaults(func=run_compare_conditions)


def _register_analyze_condition(subparsers):
    parser = subparsers.add_parser(
        "analyze-condition",
        help="Generate figures for a single tRNA alignment condition",
        description=(
            "Load a tRNA alignment BAM file and produce a standard set of per-condition "
            "figures: read counts, identity distributions, coverage heatmaps, and more."
        ),
    )

    parser.add_argument("bam", type=str, help="BAM file to analyze")
    parser.add_argument("label", type=str, help="Condition label (e.g. 'WT')")

    parser.add_argument(
        "--model", "-m", type=str, required=True,
        choices=["yeast", "e_coli"],
        help="Reference model: 'yeast' or 'e_coli'",
    )
    parser.add_argument(
        "--aligner", "-a", type=str, default="zap",
        choices=["zap", "bwa"],
        help="Aligner used to produce the BAM file (default: zap)",
    )
    parser.add_argument(
        "--out_dir", "-od", type=str, required=True,
        help="Output directory for figures and tables",
    )
    parser.add_argument(
        "--out_prefix", "-op", type=str, default="",
        help="Filename prefix for all outputs (default: none)",
    )
    parser.add_argument(
        "--threads", "-t", type=int, default=8,
        help="Threads for parallel BAM loading (default: 8)",
    )
    parser.add_argument(
        "--ident_threshold", type=float, default=0.75,
        help="Minimum alignment identity to include a read (default: 0.75)",
    )
    parser.add_argument(
        "--min_coverage", type=int, default=25,
        help="Minimum alignment length to include a read (default: 25)",
    )

    parser.set_defaults(func=run_analyze_condition)


def run_compare_conditions(args):
    """Execute the compare-conditions subcommand."""
    cond1 = load_condition(
        args.bam1, args.model, args.aligner,
        args.threads, args.ident_threshold, args.min_coverage,
    )
    cond2 = load_condition(
        args.bam2, args.model, args.aligner,
        args.threads, args.ident_threshold, args.min_coverage,
    )
    one_condition_figures(
        cond1, args.label1, args.model,
        args.out_dir, args.out_prefix, args.ident_threshold,
    )
    one_condition_figures(
        cond2, args.label2, args.model,
        args.out_dir, args.out_prefix, args.ident_threshold,
    )
    _compare_conditions(
        cond1, args.label1,
        cond2, args.label2,
        args.model, args.out_dir, args.out_prefix,
        args.ident_threshold, args.fc_threshold, args.pval_threshold,
    )


def run_analyze_condition(args):
    """Execute the analyze-condition subcommand."""
    cond = load_condition(
        args.bam, args.model, args.aligner,
        args.threads, args.ident_threshold, args.min_coverage,
    )
    one_condition_figures(
        cond, args.label, args.model,
        args.out_dir, args.out_prefix, args.ident_threshold,
    )
