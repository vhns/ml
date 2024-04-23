import argparse
import batch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', '-p', default=None, required=False,\
                            type=str, help='Specifies the path in which the dataset\
                                            lives in')
    parser.add_argument('--train', '-tr', default=None, required=False, type=float)
    parser.add_argument('--test', '-tt', default=None, required=False, type=float)
    parser.add_argument('--validation', '-vf', default=None, required=False, type=float)
    parser.add_argument('--university', '-uni', default=None, required=False, type=str)

    path = parser.parse_args().path
    train = parser.parse_args().train
    test = parser.parse_args().test
    validation = parser.parse_args().validation
    university = parser.parse_args().university

    batch.batchdata(path,train,test,validation,university)
