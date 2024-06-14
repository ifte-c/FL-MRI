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
from utils.mri_utils import NiiDataset, get_lg_layer_keys, freeze_layers
from models.Update import LocalMriUpdate, MriTest

if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    use_cache = args.use_cache.lower() == 'true'

    base_dir = './save/{}/{}_le{}/{}/'.format(
        args.dataset, args.model, args.local_ep, args.results_save)
    if not os.path.exists(os.path.join(base_dir, 'mri_lgfed')):
        os.makedirs(os.path.join(base_dir, 'mri_lgfed'), exist_ok=True)
    save_path = Path(base_dir)

    # build model
    model_glob = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )
    if (save_path / 'mri_lgfed' / 'model.pt').exists():
        model_glob.load_state_dict(torch.load(save_path / 'mri_lgfed' / 'model.pt'))
    model_glob.train()

    datasets = []
    local_states = []
    if (save_path / 'mri_lgfed' / 'local_states.pt').exists():
        local_states = torch.load(save_path / 'mri_lgfed' / 'local_states.pt')
    centres = args.centres
    target_shape = args.target_shape
    kf = KFold(n_splits=args.k)
    for centre in centres:
        dataset = NiiDataset(centre, centres, target_shape, use_cache=use_cache)
        train_idxs, test_idxs = next(kf.split(np.arange(len(dataset))))
        train_subset, test_subset = Subset(dataset, train_idxs), Subset(dataset, test_idxs)
        datasets.append((train_subset, test_subset))
        local_states.append(copy.deepcopy(model_glob.state_dict()))

    # Get lists of state keys for global & local representation layers
    state_dict = model_glob.state_dict()
    global_layer_keys, local_layer_keys = get_lg_layer_keys(state_dict)

    # Display percentage of parameters being shared globally
    num_param_glob = 0
    num_param_local = 0
    for key, param in state_dict.items():
        if key in global_layer_keys:
            num_param_glob += param.numel()
        else:
            num_param_local += param.numel()
    percentage_global = 100 * float(num_param_glob) / (num_param_glob + num_param_local)
    percentage_local = 100 * float(num_param_local) / (num_param_glob + num_param_local)
    print(f'# Params: {num_param_local} (local), {num_param_glob} (global); Global percentage {percentage_global}')

    lr = args.lr
    number_path = save_path / 'mri_lgfed' / 'round_number.pt'
    round_number = 0
    if number_path.exists():
        round_number = torch.load(number_path)
    new_round_number = 0

    local_layer_epochs = 5
    global_layer_epochs = 1
    for iter in range(args.rounds):
        
        w_glob = None
        loss_global_layers = []
        loss_local_layers = []
        print("Round {}, lr: {}".format(iter, lr))

        for idx, centre in enumerate(centres):
            
            local = LocalMriUpdate(args=args, dataset=datasets[idx][0])
            model_local = copy.deepcopy(model_glob)
            w_model = model_local.state_dict()

            # Retrieve local layers for this centre
            w_local = local_states[idx]
            for k in local_layer_keys:
                w_model[k] = w_local[k]
            model_local.load_state_dict(w_model)

            # Freeze global layers for local layer training
            freeze_layers(model_local, global_layer_keys)
            w_local, loss = local.train(model=model_local.to(args.device), local_ep=local_layer_epochs, lr=lr)
            loss_local_layers.append(copy.deepcopy(loss))

            local_states[idx] = w_local
            model_local.load_state_dict(w_local)

            # Freeze local layers for global layer training
            freeze_layers(model_local, local_layer_keys)
            w_local, loss = local.train(model=model_local.to(args.device), local_ep=global_layer_epochs, lr=lr)
            loss_global_layers.append(copy.deepcopy(loss))

            if w_glob is None:
                w_glob = copy.deepcopy(w_local)
            else:
                for k in global_layer_keys:
                    w_glob[k] += w_local[k]

        lr *= args.lr_decay

        # NOTE: Track loss averages here using tensorflow

        # update global weights
        for k in global_layer_keys:
            w_glob[k] = torch.div(w_glob[k], len(centres))

        # copy weight to model_glob
        model_glob.load_state_dict(w_glob)
        torch.save(model_glob.state_dict(), (save_path / 'mri_lgfed' / 'model.pt'))
        torch.save(local_states, (save_path / 'mri_lgfed' / 'local_states.pt'))
        new_round_number = round_number + iter + 1
        torch.save(new_round_number, (save_path / 'mri_lgfed' / 'round_number.pt'))

        # print loss
        local_loss_avg = sum(loss_local_layers) / len(loss_local_layers)
        global_loss_avg = sum(loss_global_layers) / len(loss_global_layers)
        print(f"\tAvg Local Loss: {local_loss_avg}")
        print(f"\tAvg Global Loss: {global_loss_avg}")

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

    torch.save(results, (save_path / 'mri_lgfed' / f'results{new_round_number}.pt'))