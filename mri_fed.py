import copy
import numpy as np
import pandas as pd
import torch
import os
from monai.networks.nets import UNet
from torch.utils.data import Subset
from sklearn.model_selection import KFold
from pathlib import Path

from utils.options import args_parser
from utils.mri_utils import NiiDataset
from models.Update import LocalMriUpdate, MriTest

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    use_cache = args.use_cache.lower() == 'true'

    base_dir = './save/{}/{}_le{}/{}/'.format(
        args.dataset, args.model, args.local_ep, args.results_save)
    if not os.path.exists(os.path.join(base_dir, 'mri_fed')):
        os.makedirs(os.path.join(base_dir, 'mri_fed'), exist_ok=True)
    save_path = Path(base_dir)

    datasets = []

    centres = args.centres
    target_shape = args.target_shape
    kf = KFold(n_splits=args.k)
    for centre in centres:
        dataset = NiiDataset(centre, centres, target_shape, use_cache=use_cache)
        train_idxs, test_idxs = next(kf.split(np.arange(len(dataset))))
        train_subset, test_subset = Subset(dataset, train_idxs), Subset(dataset, test_idxs)
        datasets.append((train_subset, test_subset))

    # build model
    model_glob = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )
    print(model_glob)
    if (save_path / 'mri_fed' / 'model.pt').exists():
        model_glob.load_state_dict(torch.load(save_path / 'mri_fed' / 'model.pt'))
    model_glob.train()

    # training

    lr = args.lr
    number_path = save_path / 'mri_fed' / 'round_number.pt'
    round_number = 0
    if number_path.exists():
        round_number = torch.load(number_path)
    new_round_number = 0

    for iter in range(args.rounds):
        
        w_glob = None
        loss_locals = []
        print("Round {}, lr: {}".format(iter, lr))

        for idx, centre in enumerate(centres):

            local = LocalMriUpdate(args=args, dataset=datasets[idx][0])
            model_local = copy.deepcopy(model_glob)

            w_local, loss = local.train(model=model_local.to(args.device), lr=lr)
            loss_locals.append(copy.deepcopy(loss))

            if w_glob is None:
                w_glob = copy.deepcopy(w_local)
            else:
                for k in w_glob.keys():
                    w_glob[k] += w_local[k]

        lr *= args.lr_decay

        # update global weights
        for k in w_glob.keys():
            w_glob[k] = torch.div(w_glob[k], len(centres))

        # copy weight to model_glob
        model_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print(f"\tAvg Loss: {loss_avg}")

        torch.save(model_glob.state_dict(), (save_path / 'mri_fed' / 'model.pt'))
        new_round_number = round_number + iter + 1
        torch.save(new_round_number, (save_path / 'mri_fed' / 'round_number.pt'))

    # Testing:
    results = {}
    for idx, centre in enumerate(centres):
        tester = MriTest(args, centre, datasets[idx][1])
        results[str(centre)] = tester.test(model=model_glob.to(args.device))

    total_samples = sum(len(dataset[1]) for dataset in datasets)
    results['average'] = {
        'dice_loss': sum(results[str(centre)]['dice_loss']*len(dataset[1]) for centre, dataset in zip(centres, datasets)) / total_samples,
        'iou': sum(results[str(centre)]['iou']*len(dataset[1]) for centre, dataset in zip(centres, datasets)) / total_samples,
        'accuracy': sum(results[str(centre)]['correct'] for centre in centres) / sum(results[str(centre)]['total'] for centre in centres),
        'correct': sum(results[str(centre)]['correct'] for centre in centres),
        'total': sum(results[str(centre)]['total'] for centre in centres)
    }

    torch.save(results, (save_path / 'mri_fed' / f'results{new_round_number}.pt'))