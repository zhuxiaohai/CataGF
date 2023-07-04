import os
import argparse
import pickle
import pandas as pd

from confgf import dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default='/home/zhuxiaohai/confgf_data/CATA')
    parser.add_argument('--dataset_name', type=str, default='CATA_v2')
    parser.add_argument('--tot_mol_size', type=int, default=160000)
    parser.add_argument('--conf_per_mol', type=int, default=1)
    parser.add_argument('--train_size', type=float, default=0.8)
    parser.add_argument('--val_size', type=float, default=0.2)
    parser.add_argument('--test_mol_size', type=int, default=200)
    parser.add_argument('--analysis_split', type=dict, default={'val': 100, 'test': 100})
    args = parser.parse_args()
    assert args.train_size + args.val_size == 1

    processed_data_path = os.path.join(args.base_path, '%s_processed' % args.dataset_name)
    os.makedirs(processed_data_path, exist_ok=True)
    analysis_suffix = '_analysis' if args.analysis_split else ''

    if args.tot_mol_size and (args.tot_mol_size > 0):
        cata_folder_path = os.path.join(args.base_path, 'train')
        train_data, val_data, test_data, index2split, error = dataset.preprocess_CATA_dataset(cata_folder_path,
            train_size=args.train_size, val_size=args.val_size,
            tot_mol_size=args.tot_mol_size, seed=2021,
            analysis_split=args.analysis_split)

        # save train and val data
        num = len(train_data)
        if num > 0:
            name = '%dk' % (num // args.conf_per_mol) // 1000 if (num // args.conf_per_mol >= 1000) else '%d' % num
            with open(os.path.join(processed_data_path, 'train_data_%s%s.pkl' % (
                    name, analysis_suffix)), "wb") as fout:
                pickle.dump(train_data, fout)
            print('save train %s%s done' % (name, analysis_suffix))

        num = len(val_data)
        if num > 0:
            name = '%dk' % (num // args.conf_per_mol) // 1000 if (num // args.conf_per_mol >= 1000) else '%d' % num
            with open(os.path.join(processed_data_path, 'val_data_%s%s.pkl' % (
                    name, analysis_suffix)), "wb") as fout:
                pickle.dump(val_data, fout)
            print('save val %s%s done' % (name, analysis_suffix))
        del test_data
        df = pd.DataFrame()
        for key in error.keys():
            df_i = pd.DataFrame()
            df_i['temp'] = error[key]
            df_i['error_type'] = key
            df_i['file_id'] = df_i['temp'].apply(lambda x: x[0])
            df_i['molecule_id'] = df_i['temp'].apply(lambda x: x[1])
            df_i['data_set'] = df_i['temp'].apply(lambda x: x[2])
            df = pd.concat([df, df_i], axis=0)
        df = df.drop('temp', axis=1)
        df.to_csv(os.path.join(processed_data_path, 'train_val_error_%d%s.csv' %
                               (len(val_data)+len(train_data), analysis_suffix)),
                  index=False)

    if args.test_mol_size and (args.test_mol_size > 0):
        cata_folder_path = os.path.join(args.base_path, 'test')
        test_data, error = dataset.get_CATA_testset(cata_folder_path,
                                                    tot_mol_size=args.test_mol_size, seed=2021,
                                                    analysis_split=args.analysis_split)
        with open(os.path.join(processed_data_path, 'test_data_%d%s.pkl' % (
                len(test_data), analysis_suffix)), "wb") as fout:
            pickle.dump(test_data, fout)
        print('save test %d%s done' % (len(test_data), analysis_suffix))
        df = pd.DataFrame()
        for key in error.keys():
            df_i = pd.DataFrame()
            df_i['temp'] = error[key]
            df_i['error_type'] = key
            df_i['file_id'] = df_i['temp'].apply(lambda x: x[0])
            df_i['molecule_id'] = df_i['temp'].apply(lambda x: x[1])
            df_i['data_set'] = df_i['temp'].apply(lambda x: x[2])
            df = pd.concat([df, df_i], axis=0)
        df = df.drop('temp', axis=1)
        df.to_csv(os.path.join(processed_data_path, 'test_error_%d%s.csv' % (len(test_data), analysis_suffix)), index=False)




