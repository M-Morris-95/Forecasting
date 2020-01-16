import argparse

def GetParser():
    parser = argparse.ArgumentParser(
        description='M-Morris-95 Foprecasting')

    parser.add_argument('--Server', '-S',
                        type=bool,
                        help='is it on the server?',
                        default=False,
                        required = False)

    parser.add_argument('--Model',
                        type=str,
                        help='which model use? Encoder, GRU etc.',
                        default='GRU',
                        required=False)

    parser.add_argument('--Lag',
                        type=int,
                        help='how much lag should be in the data, 28 or 112?',
                        default=28,
                        required=False)

    return parser