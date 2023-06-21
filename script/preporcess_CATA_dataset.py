import os
import argparse
import pickle

from confgf import dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default='/home/zhuxiaohai/confgf_data/CATA')
    parser.add_argument('--dataset_name', type=str, default='CATA_v2')
    parser.add_argument('--tot_mol_size', type=int, default=200)
    parser.add_argument('--conf_per_mol', type=int, default=1)
    parser.add_argument('--train_size', type=float, default=0.8)
    parser.add_argument('--val_size', type=float, default=0.2)
    parser.add_argument('--test_mol_size', type=int, default=150)
    args = parser.parse_args()
    assert args.train_size + args.val_size == 1
    cata_folder_path = os.path.join(args.base_path, 'train')

    processed_data_path = os.path.join(args.base_path, '%s_processed' % args.dataset_name)
    os.makedirs(processed_data_path, exist_ok=True)

    train_data, val_data, test_data, index2split = dataset.preprocess_CATA_dataset(cata_folder_path,
        train_size=args.train_size, val_size=args.val_size,
        tot_mol_size=args.tot_mol_size, seed=2021)

    # save train and val data
    with open(os.path.join(processed_data_path, 'train_data_%dk.pkl' % ((len(train_data) // args.conf_per_mol) // 1000)), "wb") as fout:
        pickle.dump(train_data, fout)
    print('save train %dk done' % ((len(train_data) // args.conf_per_mol) // 1000))

    with open(os.path.join(processed_data_path, 'val_data_%dk.pkl' % ((len(val_data) // args.conf_per_mol) // 1000)), "wb") as fout:
        pickle.dump(val_data, fout)
    print('save val %dk done' % ((len(val_data) // args.conf_per_mol) // 1000))
    del test_data

    # filter test data
    # cata_folder_path = os.path.join(args.base_path, 'test')
    # test_data, name_list = dataset.get_CATA_testset(cata_folder_path, tot_mol_size=args.test_mol_size, seed=2021)
    # with open(os.path.join(processed_data_path, 'test_data_%d.pkl' % len(test_data)), "wb") as fout:
    #     pickle.dump(test_data, fout)
    # with open(os.path.join(processed_data_path, 'name_list_%d.pkl' % len(test_data)), "wb") as fout2:
    #     pickle.dump(name_list, fout2)
    # print('save test %d done' % len(test_data))



