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
    parser.add_argument("--ref", required=False, help="Reference")
    parser.add_argument("--decoder", required=False, default=None, help="Decoder")
    parser.add_argument("--model", choices = ['yeast', 'ecoli'], default = None, required=False, help="Model that labeling is being performed for.")
    parser.add_argument("--min-ident", required=False, default = 0.9)
    parser.set_defaults(func=run_label)


def run_label(args):
    """Execute the label subcommand."""
    if args.model is not None:
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

    else:
        zap_label(args.bam,
                  args.ref,
                  args.out,
                  args.decoder,
                  args.min_ident
                 )
        
