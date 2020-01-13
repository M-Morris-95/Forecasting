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

    return parser