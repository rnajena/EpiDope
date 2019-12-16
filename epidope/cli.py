from argparse import ArgumentParser, SUPPRESS
# from epidope import __version__
# from ._version import get_versions
# __version__ = get_versions()['version']
import os


def cli(args=None):
    parser = ArgumentParser(
        description="Prediction of B-cell epitopes from amino acid sequences using deep neural networks.",
        conflict_handler='resolve', add_help=False
    )
    required = parser.add_argument_group('required arguments')
    required.add_argument('-i', '-infile', help='Multi- or Singe- Fasta file with protein sequences.', metavar='<File>',
                          required=True)

    optional = parser.add_argument_group('optional arguments')
    # Add back help
    optional.add_argument('-h', '--help', action='help', default=SUPPRESS, help='show this help message and exit')

    optional.add_argument('-o', '-outdir', help='Specifies output directory. Default = .', default='.',
                          metavar='<Folder>')
    optional.add_argument('-delim', help='Delimiter char for fasta header. Default = White space.', default=' ',
                          metavar='<String>')
    optional.add_argument('-idpos', help='Position of gene ID in fasta header. Zero based. Default = 0.', default=0,
                          metavar='<Integer>')
    optional.add_argument('-t', '-threshold', help='Threshold for epitope score. Default = 0.818.', default=0.818,
                          metavar='<Float>')
    optional.add_argument('-l', '-slicelen', help='Length of the sliced predicted epitopes. Default = 15.', default=15,
                          metavar='<Int>')
    optional.add_argument('-s', '-slice_shiftsize', help='Shiftsize of the slices on predited epitopes. Default = 5.',
                          default=5, metavar='<Int>')
    optional.add_argument('-p', '-processes', help='Number of processes used for predictions. Default = #CPU-cores.',
                          default=1000, metavar='<Int>')
    optional.add_argument('-e', '-epitopes', help='File containing a list of known epitope sequences for plotting.',
                          default=None, metavar='<File>')
    optional.add_argument('-n', '-nonepitopes', help='File containing a list of non epitope sequences for plotting.',
                          default=None, metavar='<File>')

    # parser.add_argument(
    #     '-V', '--version',
    #     action='version',
    #     help='Show the conda-prefix-replacement version number and exit.',
    #     version="epidope %s" % __version__,
    # )

    args = parser.parse_args(args)

    # do something with the args
    args.idpos, args.t, args.l, args.s, args.p = int(args.idpos), float(args.t), int(args.l), int(args.s), int(args.p)
    assert args.t >= 0 and args.t <= 1, "error parameter --threshold: only thresholds between 0 and 1 allowed"
    assert args.p >= 0, "error parameter --processes: Number of processes needs to be higher than 0"
    assert os.path.isfile(args.i), f"error {args.i} is not a file"

    import epidope2
    epidope2.start_pipeline(multifasta=args.i, outdir=args.o, delim=args.delim, idpos=args.idpos,
                            epitope_threshold=args.t, epitope_slicelen=args.l, slice_shiftsize=args.s, threads=args.p,
                            epi_seqs=args.e, non_epi_seqs=args.n)
    # No return value means no error.
    # Return a value of 1 or higher to signify an error.
    # See https://docs.python.org/3/library/sys.html#sys.exit


if __name__ == '__main__':
    import sys

    cli(sys.argv[1:])
