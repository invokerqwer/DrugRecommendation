import os
import torch
import numpy as np
import dill
import pickle as pkl
import argparse
from torch.optim import Adam
from modules import MedRecCGIBModel
from modules.gnn import graph_batch_from_smile
from util import buildPrjSmiles
from training import Test, Train
import random

def setup_seed(seed):
      torch.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)
      np.random.seed(seed)
      random.seed(seed)
      torch.backends.cudnn.deterministic = True

def get_model_name(args):
    model_name = [
        f'dim_{args.dim}',  f'lr_{args.lr}', f'coef_{args.coef}', f'mu_{args.mu}',f'gamma_{args.gamma}',
        f'beta_{args.beta}',f'cbsize_{args.codebook_size}',
        f'dp_{args.dp}', f'ddi_{args.target_ddi}'
    ]
    if args.embedding:
        model_name.append('embedding')
    return '-'.join(model_name)


def parse_args():
    parser = argparse.ArgumentParser('Experiment For DrugRec')
    parser.add_argument('--Test', action='store_true', help="evaluating mode")
    parser.add_argument('--dim', default=64, type=int, help='model dimension')
    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
    parser.add_argument('--dp', default=0.7, type=float, help='dropout ratio')
    parser.add_argument(
        '--model_name', type=str,
        help="the model name for training, if it's left blank,"
        " the code will generate a new model name on it own"
    )
    parser.add_argument(
        '--resume_path', type=str,
        help='path of well trained model, only for evaluating the model'
    )
    parser.add_argument(
        '--device', type=int, default=0,
        help='gpu id to run on, negative for cpu'
    )
    parser.add_argument(
        '--target_ddi', type=float, default=0.06,
        help='expected ddi for training'
    )
    parser.add_argument(
        '--coef', default=2.5, type=float,
        help='coefficient for DDI Loss Weight Annealing'
    )
    parser.add_argument(
        '--epochs', default=50, type=int,
        help='the epochs for training'
    )
    parser.add_argument(
        '--weighted_bce', action='store_true',
        help='weighted_bce'
    )
    parser.add_argument(
        '--weighted_rnn', action='store_true',
        help='weighted_rnn'
    )
    parser.add_argument(
        '--mu', default=1e-4, type=float,
        help='coefficient for patient_pred_loss and kl_loss in cgib Loss '
    )
    parser.add_argument(
        '--gamma', default=2e-4, type=float,
        help='coefficient for vq Loss '
    )
    parser.add_argument(
        '--alpha', default=1, type=float,
        help='coefficient for commitment Loss in vq loss'
    )
    parser.add_argument(
        '--codebook_size', default=128, type=int,
        help='codebook_size'
    )
    args = parser.parse_args()
    # args.weighted_rnn = True
    if args.Test and args.resume_path is None:
        raise FileNotFoundError('Can\'t Load Model Weight From Empty Dir')
    if args.model_name is None:
        args.model_name = get_model_name(args)

    return args


if __name__ == '__main__':
    setup_seed(3407)
    args = parse_args()
    print(args)
    if not torch.cuda.is_available() or args.device < 0:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.device}')

    data_path = './data/records_final.pkl'
    voc_path = './data/voc_final.pkl'
    ddi_adj_path = './data/ddi_A_final.pkl'
    ddi_mask_path = './data/ddi_mask_H.pkl'
    molecule_path = './data/idx2SMILES.pkl'
    with open(ddi_adj_path, 'rb') as Fin:
        ddi_adj = torch.from_numpy(dill.load(Fin)).to(device)
    with open(ddi_mask_path, 'rb') as Fin:
        ddi_mask_H = torch.from_numpy(dill.load(Fin)).to(device)
    with open(data_path, 'rb') as Fin:
        data = dill.load(Fin)
    with open(molecule_path, 'rb') as Fin:
        molecule = dill.load(Fin)
    with open(voc_path, 'rb') as Fin:
        voc = dill.load(Fin)

    diag_voc, pro_voc, med_voc = \
        voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    voc_size = (
        len(diag_voc.idx2word),
        len(pro_voc.idx2word),
        len(med_voc.idx2word)
    )

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point + eval_len:]

    average_projection, smiles_list = \
        buildPrjSmiles(molecule, med_voc.idx2word)
    average_projection = average_projection.to(device)

    molecule_graphs = graph_batch_from_smile(smiles_list)
    molecule_forward = {'batched_data': molecule_graphs.to(device)}
    molecule_para = {
        'num_layer': 4, 'emb_dim': args.dim, 'graph_pooling': 'mean',
        'drop_ratio': args.dp, 'gnn_type': 'gin', 'virtual_node': False
    }

    model = MedRecCGIBModel(
        global_para=molecule_para, 
        emb_dim=args.dim, 
        voc_size=voc_size,
        ddi_adj=ddi_adj,
        weighted_rnn=args.weighted_rnn,device=device, dropout=args.dp
        ,codebook_size=args.codebook_size,alpha=args.alpha
    ).to(device)

    drug_data = {
        'mol_data': molecule_forward,
        'tensor_ddi_adj': ddi_adj,
        'average_projection': average_projection
    }

    if args.Test:
        Test(model, args.resume_path, device, data_test, voc_size, drug_data)
    else:
        if not os.path.exists(os.path.join('./saved', args.model_name)):
            os.makedirs(os.path.join('./saved', args.model_name))
        log_dir = os.path.join('./saved', args.model_name)
        optimizer = Adam(model.parameters(), lr=args.lr)
        Train(
            model, device, data_train, data_eval, voc_size, drug_data,
            optimizer, log_dir, args.coef, args.target_ddi, args.weighted_bce,EPOCH=args.epochs,mu=args.mu,gamma=args.gamma
        )
