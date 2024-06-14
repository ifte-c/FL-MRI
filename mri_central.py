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
    if not os.path.exists(os.path.join(base_dir, 'central')):
        os.makedirs(os.path.join(base_dir, 'central'), exist_ok=True)
    save_path = Path(base_dir)

    centres = args.centres
    target_shape = args.target_shape
    kf = KFold(n_splits=args.k)

    dataset = NiiDataset(None, centres, target_shape, use_cache=use_cache)
    train_idxs, test_idxs = next(kf.split(np.arange(len(dataset))))
    train_subset, test_subset = Subset(dataset, train_idxs), Subset(dataset, test_idxs)

    # build model
    model_glob = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )
    if (save_path / 'central' / 'model.pt').exists():
        model_glob.load_state_dict(torch.load(save_path / 'central' / 'model.pt'))
    model_glob.train()

    lr = args.lr
    number_path = save_path / 'central' / 'round_number.pt'
    round_number = 0
    if number_path.exists():
        round_number = torch.load(number_path)
    new_round_number = 0

    for iter in range(args.rounds):

        print("Round {}, lr: {}".format(iter, lr))

        local = LocalMriUpdate(args=args, dataset=train_subset)

        w_local, loss = local.train(model=model_glob.to(args.device), lr=lr)

        lr *= args.lr_decay

        # copy weight to model_glob
        model_glob.load_state_dict(w_local)

        # print loss
        print(f"\tAvg Loss: {loss}")

    torch.save(model_glob.state_dict(), (save_path / 'central' / 'model.pt'))
    new_round_number = round_number + iter + 1
    torch.save(new_round_number, (save_path / 'central' / 'round_number.pt'))

    # Testing:
    results = {}
    tester = MriTest(args, centre='central', dataset=test_subset)
    results['central'] = tester.test(model=model_glob.to(args.device))
    # TODO: TEST ACROSS ALL CENTRES

    # total_samples = len(test_subset)
    # results['average'] = {
    #     'dice_loss': sum(results[str(centre)]['dice_loss']*len(dataset[1]) for centre, dataset in zip(centres, datasets)) / total_samples,
    #     'iou': sum(results[str(centre)]['iou']*len(dataset[1]) for centre, dataset in zip(centres, datasets)) / total_samples,
    #     'accuracy': sum(results[str(centre)]['correct'] for centre in centres) / sum(results[str(centre)]['total'] for centre in centres),
    #     'correct': sum(results[str(centre)]['correct'] for centre in centres),
    #     'total': sum(results[str(centre)]['total'] for centre in centres)
    # }

    torch.save(results, (save_path / 'central' / f'results{new_round_number}.pt'))