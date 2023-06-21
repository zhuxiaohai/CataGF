#coding: utf-8

from time import time
from tqdm import tqdm
import argparse
import numpy as np
import pickle

from confgf import utils

import multiprocessing
from functools import partial


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='confgf')
    parser.add_argument('--input', type=str)
    parser.add_argument('--core', type=int, default=6)
    parser.add_argument('--threshold', type=float, default=0.5, help='threshold of COV score')

    args = parser.parse_args()
    print(args)

    with open(args.input, 'rb') as fin:
        filtered_data_list = pickle.load(fin)

    cnt_conf = 0
    for i in range(len(filtered_data_list)):
        cnt_conf += filtered_data_list[i].num_pos_ref.item()
    print('use %d mols with total %d confs' % (len(filtered_data_list), cnt_conf))

    pool = multiprocessing.Pool(args.core)
    func = partial(utils.get_fp_simularity, threshold=args.threshold)

    covs = []
    mats = []
    for result in tqdm(pool.imap(func, filtered_data_list), total=len(filtered_data_list)):
        covs.append(result[0])
        mats.append(result[1])
    covs = np.array(covs)
    mats = np.array(mats)

    print('Coverage Mean: %.4f | Coverage Median: %.4f | Match Mean: %.4f | Match Median: %.4f' % \
                        (covs.mean(), np.median(covs), mats.mean(), np.median(mats)))
    pool.close()
    pool.join()
