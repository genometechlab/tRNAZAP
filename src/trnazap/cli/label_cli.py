# src/trnazap/cli/label_cli.py
from ..label.zap_label import zap_label, FIVE_PRIME_ADAPTER_LEN, THREE_PRIME_ADAPTER_LEN, MIN_MAPQ
from importlib.resources import files

"""Label subcommand for trnazap."""


def register_subparser(subparsers):
    """Register the label subcommand."""
    parser = subparsers.add_parser(
        "label",
        help="Label tRNA sequences",
        description="Assign labels to tRNA sequences. Without a decoder, each read keeps its BWA reference as its label.",
    )
    parser.add_argument("--bam", required=True, help="Aligned tRNA file")
    parser.add_argument("--out", required=True, help="Outpath")
    parser.add_argument("--ref", required=False, help="Reference")
    parser.add_argument("--decoder", required=False, default=None, help="Decoder (omit to keep the BWA label)")
    parser.add_argument("--model", choices = ['yeast', 'e_coli', 'human'], default = None, required=False,
                        help="Model that labeling is being performed for. 'human' keeps the BWA label unless --human-ivt.")
    parser.add_argument("--min-ident", required=False, default=0.9, type=float)
    parser.add_argument("--human-ivt", required=False, default=False, action="store_true")
    parser.add_argument("--five-prime-len", required=False, default=FIVE_PRIME_ADAPTER_LEN, type=int,
                        help="Reference positions below this are 5' adapter (5' flank incl. splint base).")
    parser.add_argument("--three-prime-len", required=False, default=THREE_PRIME_ADAPTER_LEN, type=int,
                        help="3' adapter = reference positions > ref_len - this, i.e. pass the 3' flank length + 1.")
    parser.add_argument("--min-mapq", required=False, default=MIN_MAPQ, type=int,
                        help="Skip reads below this mapping quality (0 keeps multi-mapping reads).")
    parser.set_defaults(func=run_label)


def run_label(args):
    """Execute the label subcommand."""
    if args.model is None and args.ref is None:
        raise ValueError("Either --model or --ref must be provided.")

    if args.model is not None:
        decoder_path = files('trnazap').joinpath('label')
        if args.model == 'yeast':
            decoder = decoder_path / "yeast_decoder.pkl"
        elif args.model == "e_coli":
            decoder = decoder_path / "ecoli_decoder.pkl"
        elif args.model == "human":
            # the human decoder only fits the --human-ivt path; native human keeps the BWA label
            decoder = decoder_path / "human_decoder.pkl" if args.human_ivt else None

        ref_path = files('trnazap').joinpath('references')
        if args.model == 'yeast':
            ref = ref_path / "label_references" / "sacCer3-mature-tRNAs_biosplint_subset.fa"
        elif args.model == 'e_coli':
            ref = ref_path / "label_references" / "eschColi_K_12_MG1655-mature-tRNAs_with_splints.fa"
        elif args.model == 'human':
            ref = ref_path / "label_references" / "hg38-mature-tRNAs_biosplint.fa"
        if args.ref is not None:
            ref = args.ref
    else:
        ref, decoder = args.ref, args.decoder

    zap_label(args.bam,
              ref,
              args.out,
              decoder,
              args.min_ident,
              args.human_ivt,
              five_prime_len=args.five_prime_len,
              three_prime_len=args.three_prime_len,
              min_mapq=args.min_mapq,
              )
