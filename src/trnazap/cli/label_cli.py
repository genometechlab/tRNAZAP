# src/trnazap/cli/label_cli.py
from ..label.zap_label import zap_label

"""Label subcommand for trnazap."""


def register_subparser(subparsers):
    """Register the label subcommand."""
    parser = subparsers.add_parser(
        "label",
        help="Label tRNA sequences",
        description="Assign labels to tRNA sequences",
    )
    parser.add_argument("--bam", required=True, help="Aligned tRNA file")
    parser.add_argument("--ref", required=True, help="Reference (should be long splints)")
    parser.add_argument("--out", required=True, help="Outpath")
    parser.add_argument("--decoder_dict", required=True, help="Decoder disambiguation dict")
    parser.set_defaults(func=run_label)


def run_label(args):
    """Execute the label subcommand."""
    zap_label(args.bam,
              args.ref,
              args.out,
              args.decoder_dict
              )
