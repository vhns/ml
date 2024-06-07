import argparse
import batch
import logging

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', '-p', default=None, required=False,
                            type=str, help='Specifies the path in which the dataset\
                                            lives in')
    parser.add_argument('--train', '-tr', default=None, required=False, type=float)
    parser.add_argument('--test', '-tt', default=None, required=False, type=float)
    parser.add_argument('--validation', '-vf', default=None, required=False, type=float)
    parser.add_argument('--university', '-uni', default=None, required=False, type=str)
    parser.add_argument('--batch', '-b', default=None, required=False, type=str)
    parser.add_argument('--model-path', '-mp', default=None, required=False, type=str,
                                                nargs='+')
    parser.add_argument('--gen-graphs', '-gg', default=None, required=False, type=bool)
    parser.add_argument('--logpath', '-l', default='/dev/null', required=False, type=str)

    path = parser.parse_args().path
    train = parser.parse_args().train
    test = parser.parse_args().test
    validation = parser.parse_args().validation
    university = parser.parse_args().university
    batch_run = parser.parse_args().batch
    model_path = parser.parse_args().model_path
    gen_graphs = parser.parse_args().gen_graphs
    logpath = parser.parse_args().log

    logger = getLogger(__name__)

    logger.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
                       filename=logpath, encoding='utf-8', level=logging.INFO)

    if batch_run:
        batch.batchdata(path,train,test,validation,university)

    if gen_graphs:
        batch.gen_graphs(model_path)
