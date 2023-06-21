#coding: utf-8

from time import time
from tqdm import tqdm
import argparse
import numpy as np
import pickle
from ase.io import write
from confgf import utils

import multiprocessing
from functools import partial


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='confgf')
    parser.add_argument('--input', type=str, default='/home/zhuxiaohai/confgf_test/cata/ConfGF_s0e100epoch106min_sig0.000.pkl')
    parser.add_argument('--core', type=int, default=6)
    parser.add_argument('--threshold', type=float, default=1, help='threshold of COV score')

    args = parser.parse_args()
    print(args)

    with open(args.input, 'rb') as fin:
        filtered_data_list = pickle.load(fin)

    cnt_conf = 0
    for i in range(len(filtered_data_list)):
        cnt_conf += filtered_data_list[i].num_pos_ref.item()
    print('use %d mols with total %d confs' % (len(filtered_data_list), cnt_conf))

    # pool = multiprocessing.Pool(args.core)
    # func = partial(utils.get_fp_simularity, threshold=args.threshold)

    covs = []
    mats = []
    error_count = 0
    for i in range(len(filtered_data_list)):
        try:
            result = utils.get_fp_simularity(filtered_data_list[i], args.threshold)
            write('/home/zhuxiaohai/confgf_test/cata/gen-{}.cif'.format(i), result[2])
            write('/home/zhuxiaohai/confgf_test/cata/ref-{}.cif'.format(i), result[3])
        except:
            error_count += 1
            result = (np.nan, np.nan)
        covs.append(result[0])
        mats.append(result[1])
    print(error_count)
    # for result in tqdm(pool.imap(func, filtered_data_list), total=len(filtered_data_list)):
    #     covs.append(result[0])
    #     mats.append(result[1])
    covs = np.array(covs)
    mats = np.array(mats)

    print('Coverage Mean: %.4f | Coverage Median: %.4f | Match Mean: %.4f | Match Median: %.4f' % \
                        (covs.mean(), np.median(covs), mats[~np.isnan(mats)].mean(), np.median(mats[~np.isnan(mats)])))
    # pool.close()
    # pool.join()
