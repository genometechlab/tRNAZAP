"""Infer subcommand for trnazap."""


def register_subparser(subparsers):
    """Register the infer subcommand."""
    parser = subparsers.add_parser(
        "infer",
        help="Run inference on tRNA data",
        description="Perform inference using trained models",
    )

    # Add arguments here as you develop
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        help="Input file path",
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Configuration file (YAML)",
    )

    parser.set_defaults(func=run_infer)


def run_infer(args):
    """Execute the infer subcommand."""
    print(f"Running infer subcommand...")
    print(f"  Input: {args.input}")
    print(f"  Config: {args.config}")
    print("(Not yet implemented)")
    return 0