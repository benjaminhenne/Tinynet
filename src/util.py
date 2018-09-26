import argparse

def str2bool(arg):
    if arg.lower() in ('true', 'yes', 't', 'y', '1'):
        return True
    elif arg.lower() in ('false', 'no', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')