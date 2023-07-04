#coding: utf-8

import os
from time import time
from tqdm import tqdm
import argparse
import numpy as np
import pickle
from ase.io import write
from confgf import utils

import multiprocessing
from functools import partial


def metric_func(x, base_path):
    try:
        result = utils.get_fp_simularity(x)
        write(os.path.join(base_path, 'cif', 'gen-{}-{}.cif'.format(x.file_id.item(), x.molecule_id.item())), result[2])
        write(os.path.join(base_path, 'cif', 'ref-{}-{}.cif'.format(x.file_id.item(), x.molecule_id.item())), result[3])
        return result[:2]
    except:
        return (np.nan, np.nan)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='confgf')
    parser.add_argument('--base_path', type=str, default='/home/zhuxiaohai/confgf_test/cata_v2')
    parser.add_argument('--input', type=str, default='ConfGF_s0e200epoch270min_sig0.000.pkl')
    parser.add_argument('--core', type=int, default=4)

    args = parser.parse_args()
    print(args)
    if not os.path.exists(os.path.join(args.base_path, 'cif')):
        os.makedirs(os.path.join(args.base_path, 'cif'))
    else:
        for f in os.listdir(os.path.join(args.base_path, 'cif')):
            file_path = os.path.join(os.path.join(args.base_path, 'cif'), f)
            os.remove(file_path)

    with open(os.path.join(args.base_path, args.input), 'rb') as fin:
        filtered_data_list = pickle.load(fin)

    cnt_conf = 0
    for i in range(len(filtered_data_list)):
        cnt_conf += filtered_data_list[i].num_pos_ref.item()
    print('use %d mols with total %d confs' % (len(filtered_data_list), cnt_conf))

    pool = multiprocessing.Pool(args.core)
    func = partial(metric_func, base_path=args.base_path)

    covs = []
    mats = []
    # error_count = 0
    # for i in range(len(filtered_data_list)):
    #     try:
    #         result = utils.get_fp_simularity(filtered_data_list[i])
    #         write('/home/zhuxiaohai/confgf_test/cata/cif/gen-{}.cif'.format(i), result[2])
    #         write('/home/zhuxiaohai/confgf_test/cata/cif/ref-{}.cif'.format(i), result[3])
    #     except:
    #         error_count += 1
    #         result = (np.nan, np.nan)
    #     covs.append(result[0])
    #     mats.append(result[1])
    # print(error_count)
    for result in tqdm(pool.imap(func, filtered_data_list), total=len(filtered_data_list)):
        covs.append(result[0])
        mats.append(result[1])
    covs = np.array(covs)
    mats = np.array(mats)

    print('Coverage Mean: %.4f | Coverage Median: %.4f | Match Mean: %.4f | Match Median: %.4f' % \
          (covs[~np.isnan(covs)].mean(), np.median(covs[~np.isnan(covs)]),
           mats[~np.isnan(mats)].mean(), np.median(mats[~np.isnan(mats)])))
    pool.close()
    pool.join()
