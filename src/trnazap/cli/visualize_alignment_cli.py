# src/trnazap/cli/visualize_cli.py

"""Visualize subcommand for trnazap."""
#from ..visualize.alignment_viz.comparison_figures import create_figures
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
                        type=str)


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


    parser.set_defaults(func=run_alignment_visualize)  

def run_alignment_visualize(args):
    from ..visualize.matplotlib_stylesheets.genometechlab_plotting import setup_style
    setup_style()
    
    refs = files('trnazap').joinpath('references')
    if args.model == 'e_coli':
        bwa_ref = str(refs / 'bwa_align_references' / 'eschColi_K_12_MG1655-mature-tRNAs_bwa_subset.biosplints.fa')
        zap_ref = str(refs / 'zap_align_references' / 'eschColi_K_12_MG1655-mature-tRNAs_zap_ref.fa')

    if args.model == 'yeast':
        bwa_ref = str(refs / 'bwa_align_references' / 'sacCer3-mature-tRNAs_bwa_subset_biosplints.fa')
        zap_ref = str(refs / 'zap_align_references' / 'sacCer3-mature-tRNAs_zap_ref.fa')

    fig,ax = plt.subplots(1)
    sns.lineplot([0, 5, 7.5, 6, 22.43, 11])
    fig.get_figure().savefig("/Volumes/T7/test_fig.png")
    
    create_figures(
        bwa_ref=bwa_ref,
        zap_ref=zap_ref,
        bwa_bam=args.bwa_path,
        zap_bam=args.zap_path,
        threads=args.threads,
        out_pre=args.out_prefix,
        out_dir=args.out_dir)
        
                   
    return 0