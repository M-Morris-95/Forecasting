import argparse

def GetParser():
    parser = argparse.ArgumentParser(
        description='M-Morris-95 Foprecasting')

    parser.add_argument('--Server', '-S',
                        type=bool,
                        help='is it on the server?',
                        default=False,
                        required = False)

    parser.add_argument('--Regularizer', '-R',
                        type=bool,
                        help='Does it use l2 norm?',
                        default=False,
                        required = False)

    parser.add_argument('--DOTY',
                        type=str,
                        help='use day of the year data?',
                        default='True',
                        required=False)

    parser.add_argument('--Weather',
                        type=str,
                        help='use weather data?',
                        default='True',
                        required=False)

    parser.add_argument('--Save_Model',
                        type=bool,
                        help='save models?',
                        default=False,
                        required = False)


    parser.add_argument('--K',
                        type=int,
                        help='number of iterations',
                        default=1,
                        required=False)

    parser.add_argument('--Epochs', '--E',
                        type=int,
                        help='number of epochs',
                        default=50,
                        required=False)

    parser.add_argument('--Batch_Size', '--B',
                        type=int,
                        help='Batch_Size',
                        default=128,
                        required=False)

    parser.add_argument('--Model',
                        type=str,
                        help='which model use? Encoder, GRU etc.',
                        default='GRU',
                        required=False)

    parser.add_argument('--Init',
                        type=str,
                        help='how to initialise variables',
                        default='GRU',
                        required=False)

    parser.add_argument('--Lag',
                        type=int,
                        help='how much lag should be in the data, 28 or 112?',
                        default=28,
                        required=False)

    parser.add_argument('--Look_Ahead',
                        type=int,
                        help='how much far ahead should it forcast? 7, 14, 21, All?',
                        default=14,
                        required=False)

    parser.add_argument('--Country',
                        type=str,
                        help='which country? eng, or us?',
                        default='eng',
                        required=False)

    return parser