# src/trnazap/__main__.py

import argparse
import sys
from trnazap.cli import align_cli, infer_cli, infer_multi_cli, label_cli, visualize_alignment_cli, visualize_inference_cli


def main():
    """Main entry point for trnazap CLI."""
    parser = argparse.ArgumentParser(
        prog="trnazap",
        description="tRNA-Zap: Tools for tRNA sequence analysis and alignment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0" 
    )
    
    # Create subparsers for subcommands
    subparsers = parser.add_subparsers(
        title="subcommands",
        description="Available trnazap commands",
        dest="command",
        help="Use 'trnazap <command> --help' for command-specific help",
        required=True,
    )
    
    # Register each subcommand
    align_cli.register_subparser(subparsers)
    infer_cli.register_subparser(subparsers)
    infer_multi_cli.register_subparser(subparsers)
    label_cli.register_subparser(subparsers)
    visualize_alignment_cli.register_subparser(subparsers)
    visualize_inference_cli.register_subparser(subparsers)
    
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute the appropriate subcommand
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())