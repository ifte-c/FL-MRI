import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import math
import numpy as np

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.pretrain = pretrain

    def train(self, net, idx=-1, lr=0.1):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.5)

        epoch_loss = []
        if self.pretrain:
            local_eps = self.args.local_ep_pretrain
        else:
            local_eps = self.args.local_ep
        for iter in range(local_eps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images) # images input into model

                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


class LocalUpdateMTL(object):
    def __init__(self, args, dataset=None, idxs=None, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.pretrain = pretrain

    def train(self, net, lr=0.1, omega=None, W_glob=None, idx=None, w_glob_keys=None):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.5)

        epoch_loss = []
        if self.pretrain:
            local_eps = self.args.local_ep_pretrain
        else:
            local_eps = self.args.local_ep

        for iter in range(local_eps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)

                loss = self.loss_func(log_probs, labels)

                W = W_glob.clone()

                W_local = [net.state_dict(keep_vars=True)[key].flatten() for key in w_glob_keys]
                W_local = torch.cat(W_local)
                W[:, idx] = W_local

                loss_regularizer = 0
                loss_regularizer += W.norm() ** 2

                k = 4000
                for i in range(W.shape[0] // k):
                    x = W[i * k:(i+1) * k, :]
                    loss_regularizer += x.mm(omega).mm(x.T).trace()
                f = (int)(math.log10(W.shape[0])+1) + 1
                loss_regularizer *= 10 ** (-f)

                loss = loss + loss_regularizer
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Flatten each sample in the batch
        inputs = inputs.contiguous().view(inputs.size(0), -1)
        targets = targets.contiguous().view(targets.size(0), -1)
        
        # Calculate intersection for each sample
        intersection = (inputs * targets).sum(dim=1)
        
        # Calculate Dice coefficient for each sample
        dice = (2. * intersection + self.smooth) / (inputs.sum(dim=1) + targets.sum(dim=1) + self.smooth)
        
        # Dice loss is 1 - Dice coefficient
        loss = 1 - dice
        
        # Return the mean Dice loss over the batch
        return loss.mean()
    
def iou(pred, target, smooth=1e-6):
    pred = pred.contiguous().view(pred.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)
    
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()

class LocalMriUpdate(object):
    def __init__(self, args, dataset=None):
        self.args = args
        self.loss_fn = DiceLoss()
        self.loader_train = DataLoader(dataset, batch_size=int(np.ceil(len(dataset)/8))) # TODO: adjust batch size

    def train(self, model, local_ep=1, lr=0.1):
        model.train()
        # train and update
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.5)
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr) # Consider experimenting with Adam

        epoch_loss = []
        for epoch in range(local_ep):
            batch_loss = []
            for batch_idx, (data, target) in enumerate(self.loader_train):
                data, target = data.to(self.args.device), target.to(self.args.device)
                model.zero_grad()
                print(f'\t{data.shape}', flush=True)
                pred = model(data) # images input into model

                loss = self.loss_fn(pred, target)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
class MriTest(object):
    def __init__(self, args, centre, dataset=None):
        self.args = args
        self.loss_fn = DiceLoss()
        self.loader_test = DataLoader(dataset, batch_size=int(np.ceil(len(dataset)/8)))
        self.centre = centre

    def test(self, model):
        model.eval()
        total, correct = 0, 0
        targets, preds = [], []

        with torch.no_grad():
            batch_loss = []
            iou_scores = []
            for batch_idx, (data, target) in enumerate(self.loader_test):
                data, target = data.to(self.args.device), target.to(self.args.device)
                pred = model(data)

                print(pred, target, flush=True)

                comparison = (pred.round().float() == target.float())
                total += comparison.numel()
                correct += comparison.sum()
                targets.append(target)
                preds.append(pred)

                dice_loss = self.loss_fn(pred.round(), target)
                iou_score = iou(pred.round(), target)
                iou_scores.append(iou_score.item())
                batch_loss.append(dice_loss.item())

        test_loss = sum(batch_loss)/len(batch_loss)
        test_iou = sum(iou_scores)/len(iou_scores)
        test_accuracy = 100. * correct / total

        print(f"\nCentre {self.centre}:")
        print(f'Test set: Average dice loss: {test_loss}, Average IoU: {test_iou}, Accuracy: {correct}/{total} ({test_accuracy}%)')

        return {
            'dice_loss': test_loss,
            'iou': test_iou,
            'accuracy': test_accuracy,
            'correct': correct,
            'total': total
        }