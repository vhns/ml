import base
import json
import argparse
import logging

def dump_json_dataset(university, train, test, validation, output_file, dataset_path, overlap_days, logger):
    dataset = base.dataset_gen(dataset_path, train, test, validation, university, overlap_days, logger)
    # Serialize data into file:
    json.dump(dataset, open(output_file, 'w'), indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--university', '-u',
                        default='UFPR04', required=False, type=str)
    parser.add_argument('--generate-dataset', '-g',
                        required=False, type=bool, default=None)
    parser.add_argument('--train', '-tr', default='0.4',
                        required=False, type=float)
    parser.add_argument('--test', '-tt', default='0.5',
                        required=False, type=float)
    parser.add_argument('--validation', '-vf', default='0.1',
                        required=False, type=float)
    parser.add_argument('--output-file', '-o', default=None,
                       required=False, type=str)
    parser.add_argument('--overlap-days', '-ov', default=None,
                       required=False, type=bool)
    parser.add_argument('--dataset-path', '-dp', default='./', required=False,
                        type=str, help='Specifies the path in which the dataset\
                                            lives in')

    generate_dataset = parser.parse_args().generate_dataset
    university = parser.parse_args().university
    train = parser.parse_args().train
    test = parser.parse_args().test
    validation = parser.parse_args().validation
    output_file = parser.parse_args().output_file
    dataset_path = parser.parse_args().dataset_path
    overlap_days = parser.parse_args().overlap_days

    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
                            filename='log.txt', encoding='utf-8', level=logging.INFO)

    logger = logging.getLogger(__name__)

    if generate_dataset:
        dump_json_dataset(university, train, test, validation, output_file, dataset_path, overlap_days, logger)
    else:
        print("No options were given!")
