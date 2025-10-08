# src/trnazap/cli/label_cli.py
from ..label.zap_label import zap_label
from importlib.resources import files

"""Label subcommand for trnazap."""


def register_subparser(subparsers):
    """Register the label subcommand."""
    parser = subparsers.add_parser(
        "label",
        help="Label tRNA sequences",
        description="Assign labels to tRNA sequences",
    )
    parser.add_argument("--bam", required=True, help="Aligned tRNA file")
    parser.add_argument("--out", required=True, help="Outpath")
    parser.add_argument("--model", choices = ['yeast', 'ecoli'], required=True, help="Model that labeling is being performed for.")
    parser.set_defaults(func=run_label)


def run_label(args):
    """Execute the label subcommand."""

    decoder_path = files('trnazap').joinpath('label')
    if args.model == 'yeast':
        decoder = decoder_path / "yeast_decoder.pkl"
    elif args.model == "ecoli":
        decoder = decoder_path / "ecoli_decoder.pkl"

    ref_path = files('trnazap').joinpath('references')
    if args.model == 'yeast':
        ref = ref_path / "label_references" / "sacCer3-mature-tRNAs_biosplint_subset.fa"
    elif args.model == 'ecoli':
        ref = ref_path / "label_references" / "eschColi_K_12_MG1655-mature-tRNAs_with_splints.fa"
    
    zap_label(args.bam,
              ref,
              args.out,
              decoder
              )
