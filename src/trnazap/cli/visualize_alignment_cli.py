# src/trnazap/cli/visualize_cli.py

"""Visualize subcommand for trnazap."""
from ..visualize.alignment_viz.aligner.compare_aligners import generate_aligner_comparison_figures
from importlib.resources import files
import matplotlib.pyplot as plt
import seaborn as sns

def register_subparser(subparsers):
    """Register the alignment visualize subcommand."""
    parser = subparsers.add_parser(
        "alignment_visualize",
        help="Visualize tRNA data",
        description="Create visualizations of tRNA sequences and alignments",
    )

    # General Parameters
    parser.add_argument("--model",
                        "-m",
                        required=True,
                        type=str,
                        choices = ["e_coli", "yeast"]
                       )

    parser.add_argument("--out_dir",
                        "-od",
                        required=True,
                        type=str)

    parser.add_argument("--out_prefix",
                        "-op",
                        required=True,
                        type=str)

    parser.add_argument("--threads",
                        "-t",
                        type=int,
                        default=8)


    parser.add_argument("--bwa_path",
                        default=None,
                        type=str,
                        help="Existing bwa alignment")

    parser.add_argument("--zir",
                        "-zp",
                        type=str)

    parser.add_argument("--zap_path",
                        default=None,
                        help="Existing zap alignment")

    parser.add_argument("--reference",
                        default=None,
                        help="Path to a custom BWA reference FASTA. Overrides the bundled reference for the selected model. Use only if you have a non-standard reference.")

    parser.add_argument("--five_offset",
                        default=36,
                        type=int,
                        help="Bases to trim from the 5' end of each BWA reference before scoring. "
                             "Default 36 matches the 5' biosplint on the bundled BWA references. "
                             "Set to the 5' padding of your own reference when using --reference (0 if unpadded).")

    parser.add_argument("--three_offset",
                        default=42,
                        type=int,
                        help="Bases to trim from the 3' end of each BWA reference before scoring. "
                             "Default 42 matches the 3' biosplint on the bundled BWA references. "
                             "Set to the 3' padding of your own reference when using --reference (0 if unpadded).")


    parser.set_defaults(func=run_alignment_visualize)

def run_alignment_visualize(args):
    from ..visualize.matplotlib_stylesheets.genometechlab_plotting import setup_style
    setup_style()

    generate_aligner_comparison_figures(
        reference=args.reference,
        model=args.model,
        bwa_bam=args.bwa_path,
        zap_bam=args.zap_path,
        threads=args.threads,
        five_offset=args.five_offset,
        three_offset=args.three_offset,
        out_prefix=args.out_prefix,
        out_dir=args.out_dir)
        
                   
    return 0