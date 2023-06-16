import os
import pickle
import copy
import json
import joblib
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
import random

import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.transforms import Compose
from torch_geometric.utils import to_networkx, to_undirected, to_dense_adj, dense_to_sparse
from torch_scatter import scatter
from torch_sparse import coalesce
#from torch.utils.data import Dataset

import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import Mol, HybridizationType, BondType
from rdkit import RDLogger

from ase.io import read

import networkx as nx
from tqdm import tqdm
RDLogger.DisableLog('rdApp.*')

from confgf import utils


@torch.no_grad()
# extend the edge on the fly, second order: angle, third order: dihedral
def extend_graph(data: Data, order=3):
    def binarize(x):
        return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))

    def get_higher_order_adj_matrix(adj, order, ac_target):
        """
        Args:
            adj:        (N, N)
            type_mat:   (N, N)
        """
        adj_mats = [torch.eye(adj.size(0), dtype=torch.long, device=adj.device), \
                    binarize(adj + torch.eye(adj.size(0), dtype=torch.long, device=adj.device))]

        for i in range(2, order + 1):
            mask = torch.matmul(ac_target.unsqueeze(-1), ac_target.unsqueeze(0))
            adj_mats.append(binarize(torch.where(mask == 1, adj_mats[i - 1] @ adj_mats[1], adj_mats[1])))
        order_mat = torch.zeros_like(adj)

        for i in range(1, order + 1):
            order_mat += (adj_mats[i] - adj_mats[i - 1]) * i

        return order_mat

    num_types = len(utils.BOND_TYPES)

    N = data.num_nodes
    adj = to_dense_adj(data.edge_index).squeeze(0)
    ac_target = data.ac_target
    adj_order = get_higher_order_adj_matrix(adj, order, ac_target)  # (N, N)

    type_mat = to_dense_adj(data.edge_index, edge_attr=data.edge_type).squeeze(0)  # (N, N)
    type_highorder = torch.where(adj_order > 1, num_types + adj_order - 1, torch.zeros_like(adj_order))
    assert (type_mat * type_highorder == 0).all()
    type_new = type_mat + type_highorder

    new_edge_index, new_edge_type = dense_to_sparse(type_new)
    _, edge_order = dense_to_sparse(adj_order)

    data.bond_edge_index = data.edge_index  # Save original edges
    data.edge_index, data.edge_type = coalesce(new_edge_index, new_edge_type.long(), N, N)  # modify data
    edge_index_1, data.edge_order = coalesce(new_edge_index, edge_order.long(), N, N)  # modify data
    data.is_bond = (data.edge_type < num_types)
    assert (data.edge_index == edge_index_1).all()

    return data


def df_to_data(node, edge, node_feature_names, molecule):
    pos = torch.tensor(node[['x', 'y', 'z']].values, dtype=torch.float32)
    edge_index = torch.tensor([edge['source'].tolist(),
                               edge['target'].tolist()], dtype=torch.long)
    x = torch.tensor(node[node_feature_names].values, dtype=torch.float)
    edge_type = torch.tensor(edge['bond_type'].values, dtype=torch.int)
    ac_target = torch.tensor(node['ac_target'].values, dtype=torch.float32)
    data = Data(x=x, pos=pos, edge_index=edge_index, edge_type=edge_type, ac_target=ac_target)
    data.edge_index, data.edge_type = to_undirected(data.edge_index, edge_attr=edge_type, reduce='add')
    G = to_networkx(data, to_undirected=True)
    if len(list(nx.connected_components(G))) > 1:
        return None
    data = extend_graph(data, order=2)
    data.edge_length = torch.tensor(molecule.get_distances(data.edge_index[0], data.edge_index[1], mic=True),
                                    dtype=torch.float32).unsqueeze(-1) # (num_edge, 1)
    return data


def rdmol_to_data(mol:Mol, smiles=None):
    assert mol.GetNumConformers() == 1
    N = mol.GetNumAtoms()

    pos = torch.tensor(mol.GetConformer(0).GetPositions(), dtype=torch.float32)

    atomic_number = []
    aromatic = []
    sp = []
    sp2 = []
    sp3 = []
    num_hs = []
    for atom in mol.GetAtoms():
        atomic_number.append(atom.GetAtomicNum())
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        hybridization = atom.GetHybridization()
        sp.append(1 if hybridization == HybridizationType.SP else 0)
        sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
        sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

    z = torch.tensor(atomic_number, dtype=torch.long)

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [utils.BOND_TYPES[bond.GetBondType()]]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type)

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]

    row, col = edge_index
    hs = (z == 1).to(torch.float32)

    num_hs = scatter(hs[row], col, dim_size=N, reduce='sum').tolist()

    if smiles is None:
        smiles = Chem.MolToSmiles(mol)

    data = Data(atom_type=z, pos=pos, edge_index=edge_index, edge_type=edge_type,
                rdmol=copy.deepcopy(mol), smiles=smiles)
    #data.nx = to_networkx(data, to_undirected=True)

    return data

def smiles_to_data(smiles):
    """
    Convert a SMILES to a pyg object that can be fed into ConfGF for generation
    """
    try:    
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    except:
        return None
        
    N = mol.GetNumAtoms()
    pos = torch.rand((N, 3), dtype=torch.float32)

    atomic_number = []
    aromatic = []

    for atom in mol.GetAtoms():
        atomic_number.append(atom.GetAtomicNum())
        aromatic.append(1 if atom.GetIsAromatic() else 0)

    z = torch.tensor(atomic_number, dtype=torch.long)

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [utils.BOND_TYPES[bond.GetBondType()]]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type)

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]

    row, col = edge_index

    data = Data(atom_type=z, pos=pos, edge_index=edge_index, edge_type=edge_type,
                rdmol=copy.deepcopy(mol), smiles=smiles)
    
    transform = Compose([
        utils.AddHigherOrderEdges(order=3),
        utils.AddEdgeLength(),
        utils.AddPlaceHolder(),
        utils.AddEdgeName()
    ])
    
    return transform(data)


def preprocess_iso17_dataset(base_path):
    train_path = os.path.join(base_path, 'iso17_split-0_train.pkl')
    test_path = os.path.join(base_path, 'iso17_split-0_test.pkl')
    with open(train_path, 'rb') as fin:
        raw_train = pickle.load(fin)
    with open(test_path, 'rb') as fin:
        raw_test = pickle.load(fin)

    smiles_list_train = [utils.mol_to_smiles(mol) for mol in raw_train]
    smiles_set_train = list(set(smiles_list_train))
    smiles_list_test = [utils.mol_to_smiles(mol) for mol in raw_test]
    smiles_set_test = list(set(smiles_list_test))

    print('preprocess train...')
    all_train = []
    for i in tqdm(range(len(raw_train))):
        smiles = smiles_list_train[i]
        data = rdmol_to_data(raw_train[i], smiles=smiles)
        all_train.append(data)

    print('Train | find %d molecules with %d confs' % (len(smiles_set_train), len(all_train)))    
    
    print('preprocess test...')
    all_test = []
    for i in tqdm(range(len(raw_test))):
        smiles = smiles_list_test[i]
        data = rdmol_to_data(raw_test[i], smiles=smiles)
        all_test.append(data)

    print('Test | find %d molecules with %d confs' % (len(smiles_set_test), len(all_test)))  

    return all_train, all_test


def preprocess_CATA_dataset(base_path, train_size=0.8, val_size=0.2, tot_mol_size=5000, seed=None):
    """
    base_path: directory that contains GEOM dataset
    conf_per_mol: keep mol that has at least conf_per_mol confs, and sampling the most probable conf_per_mol confs
    train_size ratio, val = test = (1-train_size) / 2
    seed: rand seed for RNG
    """

    # set random seed
    if seed is None:
        seed = 2021
    np.random.seed(seed)
    random.seed(seed)

    # read summary file
    summary_path = os.path.join(base_path, 'summary', 'summary.csv')
    summ = pd.read_csv(summary_path)
    summ = summ[summ['error'] == 1]
    summ['id'] = summ['id'].astype(int)
    summ['extxyz_id'] = summ['extxyz_id'].astype(int)
    summ['data_id'] = summ['data_id'].astype(int)
    summ = summ.set_index('id', drop=True).head(4)

    # filter valid pickle path
    pickle_path_list = summ.index.tolist()
    random.shuffle(pickle_path_list)
    print('pre-filter: find %d confs' % len(pickle_path_list))
    pickle_path_list = pickle_path_list[:tot_mol_size]
    print('but use %d confs' % len(pickle_path_list))

    # 1. select the most probable 'conf_per_mol' confs of each 2D molecule
    # 2. split the dataset based on 2D structure, i.e., test on unseen graphs
    train_data, val_data, test_data = [], [], []

    # generate train, val, test split indexes
    split_indexes = list(range(tot_mol_size))
    random.shuffle(split_indexes)
    index2split = {}
    for i in range(0, int(tot_mol_size * train_size)):
        index2split[split_indexes[i]] = 'train'
    for i in range(int(tot_mol_size * train_size), int(tot_mol_size * (train_size + val_size))):
        index2split[split_indexes[i]] = 'val'
    for i in range(int(tot_mol_size * (train_size + val_size)), tot_mol_size):
        index2split[split_indexes[i]] = 'test'

    num_mols = np.zeros(4, dtype=int)  # (tot, train, val, test)
    num_confs = np.zeros(4, dtype=int)  # (tot, train, val, test)

    bad_case = 0

    for i in tqdm(range(len(pickle_path_list))):
        file_id = summ.loc[pickle_path_list[i], 'extxyz_id']
        molecule_id = summ.loc[pickle_path_list[i], 'data_id']
        file_path = str(file_id) + '.extxyz'
        molecule_path = str(file_id) + '-' + str(molecule_id) + '.csv'
        molecule = read(os.path.join(base_path, file_path), molecule_id)
        node = pd.read_csv(os.path.join(base_path, 'node', molecule_path))
        edge = pd.read_csv(os.path.join(base_path, 'edge', molecule_path))
        edge['id'] = edge['id'].astype(int)
        edge['node1'] = edge['node1'].astype(int)
        edge['node2'] = edge['node2'].astype(int)
        edge['bond_type'] = (edge['bond_type'] + 1).astype(int)
        node['id'] = node['id'].astype(int)
        node['label'] = node['label'].astype(int)
        node['ac_target'] = node['ac_target'].astype(float)
        edge = edge.rename(columns={'id': 'edge_id', 'node1': 'source', 'node2': 'target'})
        node = node.rename(columns={'id': 'node_id'})

        if node['node_id'].unique().shape[0] != node.shape[0]:
            print('node ids are not unique')
            bad_case += 1
            continue
        if edge[['source', 'target']].duplicated().sum() > 0:
            print('edges are not unique')
            bad_case += 1
            continue

        bin_dict = {'puling_en': {'min': 0.5, 'max': 4, 'num': 10},
                    'ionization_eng_lg': {'min': 0.5, 'max': 1.4, 'num': 9},
                    'covalent_radius': {'min': 0.25, 'max': 2.5, 'num': 10}}
        cate_dict = {'puling_en': [list(range(10))],
                     'ionization_eng_lg': [list(range(9))],
                     'covalent_radius': [list(range(10))],
                     'unpaired_elec': [list(range(9))],
                     'valence_elec': [list(range(1, 18))],
                     'block': [list(range(4))]}
        node_feature_names = list(cate_dict.keys())
        expected_features_num = 10 + 9 + 10 + 9 + 17 + 4
        final_node_feature_names = []
        for col in node_feature_names:
            col_bin = col
            if col in bin_dict:
                col_bin = col + '_bin'
                est = KBinsDiscretizer(n_bins=bin_dict[col]['num'], encode='ordinal', strategy='uniform')
                est.fit([[bin_dict[col]['min']], [bin_dict[col]['max']]])
                node[col_bin] = est.transform(node[[col]].values)
            enc = OneHotEncoder(handle_unknown='ignore', categories=cate_dict[col])
            enc.fit(node[[col_bin]])
            new_names = [col + '_' + str(i) for i in range(len(cate_dict[col][0]))]
            node[new_names] = enc.transform(node[[col_bin]]).toarray()
            final_node_feature_names += new_names
        assert len(final_node_feature_names) == expected_features_num

        data = df_to_data(node, edge, final_node_feature_names, molecule)
        if data is None:
            print('unconnected graph')
            bad_case += 1
            continue
        data['idx'] = torch.tensor([i], dtype=torch.long)
        datas = [data]

        if index2split[i] == 'train':
            train_data.extend(datas)
            num_mols += [1, 1, 0, 0]
            num_confs += [len(datas), len(datas), 0, 0]
        elif index2split[i] == 'val':
            val_data.extend(datas)
            num_mols += [1, 0, 1, 0]
            num_confs += [len(datas), 0, len(datas), 0]
        elif index2split[i] == 'test':
            test_data.extend(datas)
            num_mols += [1, 0, 0, 1]
            num_confs += [len(datas), 0, 0, len(datas)]
        else:
            raise ValueError('unknown index2split value.')

    print('post-filter: find %d molecules with %d confs' % (num_mols[0], num_confs[0]))
    print('train size: %d molecules with %d confs' % (num_mols[1], num_confs[1]))
    print('val size: %d molecules with %d confs' % (num_mols[2], num_confs[2]))
    print('test size: %d molecules with %d confs' % (num_mols[3], num_confs[3]))
    print('bad case: %d' % bad_case)
    print('done!')

    return train_data, val_data, test_data, index2split


def get_CATA_testset(base_path, tot_mol_size=5000, seed=None):
    """
    base_path: directory that contains GEOM dataset
    conf_per_mol: keep mol that has at least conf_per_mol confs, and sampling the most probable conf_per_mol confs
    train_size ratio, val = test = (1-train_size) / 2
    seed: rand seed for RNG
    """

    # set random seed
    if seed is None:
        seed = 2021
    np.random.seed(seed)
    random.seed(seed)

    # read summary file
    summary_path = os.path.join(base_path, 'summary', 'summary.csv')
    summ = pd.read_csv(summary_path)
    summ = summ[summ['error'] == 1]
    summ['id'] = summ['id'].astype(int)
    summ['extxyz_id'] = summ['extxyz_id'].astype(int)
    summ['data_id'] = summ['data_id'].astype(int)
    summ = summ.set_index('id', drop=True).head(4)

    # filter valid pickle path
    pickle_path_list = summ.index.tolist()
    random.shuffle(pickle_path_list)
    print('pre-filter: find %d confs' % len(pickle_path_list))
    pickle_path_list = pickle_path_list[:tot_mol_size]
    print('but use %d confs' % len(pickle_path_list))

    test_data = []

    bad_case = 0

    for i in tqdm(range(len(pickle_path_list))):
        file_id = summ.loc[pickle_path_list[i], 'extxyz_id']
        molecule_id = summ.loc[pickle_path_list[i], 'data_id']
        file_path = str(file_id) + '.extxyz'
        molecule_path = str(file_id) + '-' + str(molecule_id) + '.csv'
        molecule = read(os.path.join(base_path, file_path), molecule_id)
        node = pd.read_csv(os.path.join(base_path, 'init_node', molecule_path))
        edge = pd.read_csv(os.path.join(base_path, 'init_edge', molecule_path))

        edge['id'] = edge['id'].astype(int)
        edge['node1'] = edge['node1'].astype(int)
        edge['node2'] = edge['node2'].astype(int)
        edge['bond_type'] = (edge['bond_type'] + 1).astype(int)
        node['id'] = node['id'].astype(int)
        node['label'] = node['label'].astype(int)
        node['ac_target'] = node['ac_target'].astype(float)
        edge = edge.rename(columns={'id': 'edge_id', 'node1': 'source', 'node2': 'target'})
        node = node.rename(columns={'id': 'node_id'})

        if (node['node_id'].unique().shape[0] != node.shape[0]):
            bad_case += 1
            continue
        if (edge[['source', 'target']].duplicated().sum() > 0):
            bad_case += 1
            continue

        bin_dict = {'puling_en': {'min': 0.5, 'max': 4, 'num': 10},
                    'ionization_eng_lg': {'min': 0.5, 'max': 1.4, 'num': 9},
                    'covalent_radius': {'min': 0.25, 'max': 2.5, 'num': 10}}
        cate_dict = {'puling_en': [list(range(10))],
                     'ionization_eng_lg': [list(range(9))],
                     'covalent_radius': [list(range(10))],
                     'unpaired_elec': [list(range(9))],
                     'valence_elec': [list(range(1, 18))],
                     'block': [list(range(4))]}
        node_feature_names = list(cate_dict.keys())
        expected_features_num = 10 + 9 + 10 + 9 + 17 + 4
        final_node_feature_names = []
        for col in node_feature_names:
            col_bin = col
            if col in bin_dict:
                col_bin = col + '_bin'
                est = KBinsDiscretizer(n_bins=bin_dict[col]['num'], encode='ordinal', strategy='uniform')
                est.fit([[bin_dict[col]['min']], [bin_dict[col]['max']]])
                node[col_bin] = est.transform(node[[col]].values)
            enc = OneHotEncoder(handle_unknown='ignore', categories=cate_dict[col])
            enc.fit(node[[col_bin]])
            new_names = [col + '_' + str(i) for i in range(len(cate_dict[col][0]))]
            node[new_names] = enc.transform(node[[col_bin]]).toarray()
            final_node_feature_names += new_names
        assert len(final_node_feature_names) == expected_features_num

        data = df_to_data(node, edge, final_node_feature_names, molecule)
        data['idx'] = torch.tensor([i], dtype=torch.long)
        datas = [data]
        test_data.extend(datas)

    print('test size: %d confs' % len(test_data))
    print('bad case: %d' % bad_case)
    print('done!')

    return test_data


def preprocess_GEOM_dataset(base_path, dataset_name, conf_per_mol=5, train_size=0.8, tot_mol_size=50000, seed=None):
    """
    base_path: directory that contains GEOM dataset
    dataset_name: dataset name in [qm9, drugs]
    conf_per_mol: keep mol that has at least conf_per_mol confs, and sampling the most probable conf_per_mol confs
    train_size ratio, val = test = (1-train_size) / 2
    tot_mol_size: max num of mols. The total number of final confs should be tot_mol_size * conf_per_mol
    seed: rand seed for RNG
    """

    # set random seed
    if seed is None:
        seed = 2021
    np.random.seed(seed)
    random.seed(seed)
    

    # read summary file
    assert dataset_name in ['qm9', 'drugs']
    summary_path = os.path.join(base_path, 'summary_%s.json' % dataset_name)
    with open(summary_path, 'r') as f:
        summ = json.load(f)

    # filter valid pickle path
    smiles_list = []
    pickle_path_list = []
    num_mols = 0    
    num_confs = 0    
    for smiles, meta_mol in tqdm(summ.items()):
        u_conf = meta_mol.get('uniqueconfs')
        if u_conf is None:
            continue
        pickle_path = meta_mol.get('pickle_path')
        if pickle_path is None:
            continue
        if u_conf < conf_per_mol:
            continue
        num_mols += 1
        num_confs += conf_per_mol
        smiles_list.append(smiles)
        pickle_path_list.append(pickle_path)

    random.shuffle(pickle_path_list)
    assert len(pickle_path_list) >= tot_mol_size, 'the length of all available mols is %d, which is smaller than tot mol size %d' % (len(pickle_path_list), tot_mol_size)

    pickle_path_list = pickle_path_list[:tot_mol_size]

    print('pre-filter: find %d molecules with %d confs, use %d molecules with %d confs' % (num_mols, num_confs, tot_mol_size, tot_mol_size*conf_per_mol))


    # 1. select the most probable 'conf_per_mol' confs of each 2D molecule
    # 2. split the dataset based on 2D structure, i.e., test on unseen graphs
    train_data, val_data, test_data = [], [], []
    val_size = test_size = (1. - train_size) / 2

    # generate train, val, test split indexes
    split_indexes = list(range(tot_mol_size))
    random.shuffle(split_indexes)
    index2split = {}
    for i in range(0, int(len(split_indexes) * train_size)):
        index2split[split_indexes[i]] = 'train'
    for i in range(int(len(split_indexes) * train_size), int(len(split_indexes) * (train_size + val_size))):
        index2split[split_indexes[i]] = 'val'
    for i in range(int(len(split_indexes) * (train_size + val_size)), len(split_indexes)):
        index2split[split_indexes[i]] = 'test'        


    num_mols = np.zeros(4, dtype=int) # (tot, train, val, test)
    num_confs = np.zeros(4, dtype=int) # (tot, train, val, test)


    bad_case = 0

    for i in tqdm(range(len(pickle_path_list))):
        
        with open(os.path.join(base_path, pickle_path_list[i]), 'rb') as fin:
            mol = pickle.load(fin)
        
        if mol.get('uniqueconfs') > len(mol.get('conformers')):
            bad_case += 1
            continue
        if mol.get('uniqueconfs') <= 0:
            bad_case += 1
            continue

        datas = []
        smiles = mol.get('smiles')

        if mol.get('uniqueconfs') == conf_per_mol:
            # use all confs
            conf_ids = np.arange(mol.get('uniqueconfs'))
        else:
            # filter the most probable 'conf_per_mol' confs
            all_weights = np.array([_.get('boltzmannweight', -1.) for _ in mol.get('conformers')])
            descend_conf_id = (-all_weights).argsort()
            conf_ids = descend_conf_id[:conf_per_mol]

        for conf_id in conf_ids:
            conf_meta = mol.get('conformers')[conf_id]
            data = rdmol_to_data(conf_meta.get('rd_mol'), smiles=smiles)
            labels = {
                'totalenergy': conf_meta['totalenergy'],
                'boltzmannweight': conf_meta['boltzmannweight'],
            }
            for k, v in labels.items():
                data[k] = torch.tensor([v], dtype=torch.float32)
            data['idx'] = torch.tensor([i], dtype=torch.long)
            datas.append(data)
        assert len(datas) == conf_per_mol

        if index2split[i] == 'train':
            train_data.extend(datas)
            num_mols += [1, 1, 0, 0]
            num_confs += [len(datas), len(datas), 0, 0]
        elif index2split[i] == 'val':    
            val_data.extend(datas)
            num_mols += [1, 0, 1, 0]
            num_confs += [len(datas), 0, len(datas), 0]
        elif index2split[i] == 'test': 
            test_data.extend(datas)
            num_mols += [1, 0, 0, 1]
            num_confs += [len(datas), 0, 0, len(datas)] 
        else:
            raise ValueError('unknown index2split value.')                         

    print('post-filter: find %d molecules with %d confs' % (num_mols[0], num_confs[0]))    
    print('train size: %d molecules with %d confs' % (num_mols[1], num_confs[1]))    
    print('val size: %d molecules with %d confs' % (num_mols[2], num_confs[2]))    
    print('test size: %d molecules with %d confs' % (num_mols[3], num_confs[3]))    
    print('bad case: %d' % bad_case)
    print('done!')

    return train_data, val_data, test_data, index2split


def get_GEOM_testset(base_path, dataset_name, block, tot_mol_size=200, seed=None, confmin=50, confmax=500):
    """
    base_path: directory that contains GEOM dataset
    dataset_name: dataset name, should be in [qm9, drugs]
    block: block the training and validation set
    tot_mol_size: size of the test set
    seed: rand seed for RNG
    confmin and confmax: range of the number of conformations
    """

    #block smiles in train / val 
    block_smiles = defaultdict(int)
    for block_ in block:
        for i in range(len(block_)):
            block_smiles[block_[i].smiles] = 1

    # set random seed
    if seed is None:
        seed = 2021
    np.random.seed(seed)
    random.seed(seed)
    

    # read summary file
    assert dataset_name in ['qm9', 'drugs']
    summary_path = os.path.join(base_path, 'summary_%s.json' % dataset_name)
    with open(summary_path, 'r') as f:
        summ = json.load(f)

    # filter valid pickle path
    smiles_list = []
    pickle_path_list = []
    num_mols = 0    
    num_confs = 0    
    for smiles, meta_mol in tqdm(summ.items()):
        u_conf = meta_mol.get('uniqueconfs')
        if u_conf is None:
            continue
        pickle_path = meta_mol.get('pickle_path')
        if pickle_path is None:
            continue
        if u_conf < confmin or u_conf > confmax:
            continue
        if block_smiles[smiles] == 1:
            continue

        num_mols += 1
        num_confs += u_conf
        smiles_list.append(smiles)
        pickle_path_list.append(pickle_path)


    random.shuffle(pickle_path_list)
    assert len(pickle_path_list) >= tot_mol_size, 'the length of all available mols is %d, which is smaller than tot mol size %d' % (len(pickle_path_list), tot_mol_size)

    pickle_path_list = pickle_path_list[:tot_mol_size]

    print('pre-filter: find %d molecules with %d confs' % (num_mols, num_confs))


    bad_case = 0
    all_test_data = []
    num_valid_mol = 0
    num_valid_conf = 0

    for i in tqdm(range(len(pickle_path_list))):
        
        with open(os.path.join(base_path, pickle_path_list[i]), 'rb') as fin:
            mol = pickle.load(fin)
        
        if mol.get('uniqueconfs') > len(mol.get('conformers')):
            bad_case += 1
            continue
        if mol.get('uniqueconfs') <= 0:
            bad_case += 1
            continue

        datas = []
        smiles = mol.get('smiles')

        conf_ids = np.arange(mol.get('uniqueconfs'))
      
        for conf_id in conf_ids:
            conf_meta = mol.get('conformers')[conf_id]
            data = rdmol_to_data(conf_meta.get('rd_mol'), smiles=smiles)
            labels = {
                'totalenergy': conf_meta['totalenergy'],
                'boltzmannweight': conf_meta['boltzmannweight'],
            }
            for k, v in labels.items():
                data[k] = torch.tensor([v], dtype=torch.float32)
            data['idx'] = torch.tensor([i], dtype=torch.long)
            datas.append(data)

      
        all_test_data.extend(datas)
        num_valid_mol += 1
        num_valid_conf += len(datas)

    print('poster-filter: find %d molecules with %d confs' % (num_valid_mol, num_valid_conf))


    return all_test_data


class CATADataset(Dataset):

    def __init__(self, data=None, transform=None):
        super().__init__()
        self.data = data
        self.transform = transform
        self.edge_types = self._edge_types()

    def __getitem__(self, idx):

        data = self.data[idx].clone()
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __len__(self):
        return len(self.data)

    def _edge_types(self):
        """All edge types."""
        edge_types = set()
        for graph in self.data:
            edge_types.update(graph.edge_type.tolist())
        return sorted(edge_types)


class GEOMDataset(Dataset):

    def __init__(self, data=None, transform=None):
        super().__init__()
        self.data = data
        self.transform = transform
        self.atom_types = self._atom_types()
        self.edge_types = self._edge_types()

    def __getitem__(self, idx):

        data = self.data[idx].clone()
        if self.transform is not None:
            data = self.transform(data)        
        return data

    def __len__(self):
        return len(self.data)

        
    def _atom_types(self):
        """All atom types."""
        atom_types = set()
        for graph in self.data:
            atom_types.update(graph.atom_type.tolist())
        return sorted(atom_types)

    def _edge_types(self):
        """All edge types."""
        edge_types = set()
        for graph in self.data:
            edge_types.update(graph.edge_type.tolist())
        return sorted(edge_types)




class GEOMDataset_PackedConf(GEOMDataset):

    def __init__(self, data=None, transform=None):
        super(GEOMDataset_PackedConf, self).__init__(data, transform)
        self._pack_data_by_mol()

    def _pack_data_by_mol(self):
        """
        pack confs with same mol into a single data object
        """
        self._packed_data = defaultdict(list)
        if hasattr(self.data, 'idx'):
            for i in range(len(self.data)):
                self._packed_data[self.data[i].idx.item()].append(self.data[i])
        else:
            for i in range(len(self.data)):
                self._packed_data[self.data[i].smiles].append(self.data[i])
        print('got %d molecules with %d confs' % (len(self._packed_data), len(self.data)))

        new_data = []
        # logic
        # save graph structure for each mol once, but store all confs 
        cnt = 0
        for k, v in self._packed_data.items():
            data = copy.deepcopy(v[0])
            all_pos = []
            for i in range(len(v)):
                all_pos.append(v[i].pos)
            data.pos_ref = torch.cat(all_pos, 0) # (num_conf*num_node, 3)
            data.num_pos_ref = torch.tensor([len(all_pos)], dtype=torch.long)
            #del data.pos

            if hasattr(data, 'totalenergy'):
                del data.totalenergy
            if hasattr(data, 'boltzmannweight'):
                del data.boltzmannweight
            new_data.append(data)
        self.new_data = new_data
        

    def __getitem__(self, idx):

        data = self.new_data[idx].clone()
        if self.transform is not None:
            data = self.transform(data)        
        return data

    def __len__(self):
        return len(self.new_data)

               

if __name__ == '__main__':
    pass