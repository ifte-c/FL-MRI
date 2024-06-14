from torchio.transforms import Compose, HistogramStandardization, RescaleIntensity, Resample, CropOrPad
from torch.utils.data import Dataset, DataLoader
import torch
import torchio as tio
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
import inspect
import os

# from monai.networks.nets import UNet

class NiiDataset(Dataset):
    def __init__(self, centre: int | None, centres: list[int], target_shape: list, use_cache = False):
        self.subjects = []
        self.standardization_dir = datasets_dir / 'Standardization' / f'{centre if centre != None else "centralised"}'
        self.processed_dir = datasets_dir / 'Processed' / f'{centre if centre != None else "centralised"}'

        self.standardization_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        if use_cache and any(self.processed_dir.iterdir()):
            for subject_path in self.processed_dir.glob('*.pt'):
                self.subjects.append(torch.load(subject_path))
        else:
            self._process_subjects(centre, centres, target_shape)

    def _process_subjects(self, centre: int | None, centres: list[int], target_shape: list):
        # centre: None=Centralised, 0=ACDC, 1=Vall d'Hebron, 2=Sagrada Familia, 4=SantPau
        target_shape = np.append(target_shape, 1)
        patients = []

        # Empty standardization and processed folders for this centre
        for file in self.standardization_dir.glob('*.nii.gz'):
            os.remove(file)
        for file in self.processed_dir.glob('*.nii.gz'):
            os.remove(file)

        # Get patients only from specific centres
        if centre == 0 or centre == None:
            directory = acdc_training
            patients += list(directory.glob(f'*/*.nii.gz'))
        if centre != 0 or centre == None:
            directory = mnm_training
            mnmInfoPath = list(directory.resolve().parent.parent.glob('*.csv'))
            assert len(mnmInfoPath) == 1, f'{len(mnmInfoPath)} .csv files found'
            mnmInfo = pd.read_csv(mnmInfoPath[0])
            validCentres = mnmInfo.loc[(mnmInfo['Centre'].isin(centres))]
            codes = validCentres['External code'].tolist()
            for child in directory.iterdir():
                if child.stem in codes:
                    patients += list(directory.glob(f'{child.stem}/*.nii.gz'))

        # Pair data with ground truths
        gt_dict = {}
        patients_zip_raw = []
        for patient in patients:
            if patient.match('*_gt.nii.gz'):
                name = patient.stem.split('.')[0].rsplit('_gt', 1)[0]
                gt_dict[name] = patient
        for patient in patients:
            name = patient.stem.split('.')[0]
            if not patient.match('*_gt.nii.gz') and gt_dict.get(name):
                patients_zip_raw.append((patient, gt_dict[name]))
        assert len(patients_zip_raw) == len(patients) / 2, "Not all images have ground truths"

        # Process images into transformed slices
        resampled_subjects = []
        for data_path, truth_path in patients_zip_raw:
            data, truth = nib.load(data_path).get_fdata(), nib.load(truth_path).get_fdata()
            assert data.shape == truth.shape, f"data and truth have different shapes: {data.shape} vs {truth.shape}"

            # Remove duplicate slices
            slice_idxs = []
            for i in range(data.shape[2]):
                if i == 0 or not np.array_equal(data[:, :, i], data[:, :, i-1]):
                    slice_idxs.append(i)

            # Resample and store new slices in standardization folder
            unique_data, unique_truth =  np.take(data, slice_idxs, axis=2), np.take(truth, slice_idxs, axis=2)
            for slice_idx in range(unique_data.shape[2]):
                data_slice, truth_slice = np.expand_dims(unique_data[:, :, slice_idx], (0, 3)), np.expand_dims(unique_truth[:, :, slice_idx], (0, 3))
                
                subject = tio.Subject(data=tio.ScalarImage(tensor=data_slice), truth=tio.LabelMap(tensor=truth_slice))
                new_spacing = np.array(subject["data"].spacing) * (np.array(subject["data"].shape[1:]) / target_shape) # TODO: Check spacing
                transform = Compose([Resample(new_spacing, image_interpolation='bspline', label_interpolation='nearest'), CropOrPad(target_shape)])

                resampled_slice = transform(subject)
                new_name = f'{data_path.stem.split(".")[0]}_{slice_idx}'
                resampled_subjects.append((resampled_slice, new_name))
                resampled_slice["data"].save(self.standardization_dir / (f'{new_name}.nii.gz'))

        # Train landmarks for histogram standardization
        resampled_paths = list(self.standardization_dir.glob('*.nii.gz'))
        landmarks = HistogramStandardization.train(resampled_paths)
        transform = Compose([HistogramStandardization({'data': landmarks}), RescaleIntensity((0,1))])

        for subject, name in resampled_subjects:
            subject_processed = transform(subject)
            torch.save(subject_processed, (self.processed_dir / f'{name}.pt'))
            self.subjects.append(subject_processed)

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject = self.subjects[idx]
        data = np.squeeze(subject['data']['data'], axis=3)
        truth = np.squeeze(subject['truth']['data'], axis=3)

        return data, truth
    
def get_lg_layer_keys(state_dict):
    layer_names = list(state_dict.keys())
    num_layers = len(layer_names)
    middle_layer_start = num_layers // 3
    middle_layer_end = 2 * num_layers // 3
    upper_layers = layer_names[:middle_layer_start] + layer_names[middle_layer_end:]
    middle_layers = layer_names[middle_layer_start:middle_layer_end]
    return upper_layers, middle_layers

def freeze_layers(model, keys_to_freeze):
    for name, param in model.named_parameters():
        if name in keys_to_freeze:
            param.requires_grad = False
        else:
            param.requires_grad = True

script_dir = Path(inspect.getfile(inspect.currentframe())).parent.parent
datasets_dir = (script_dir / 'data' / 'Datasets').resolve()
acdc_training = (datasets_dir / 'ACDC' / 'Processed' / 'Training').resolve()
mnm_training = (datasets_dir / 'MnM' / 'Processed' / 'Training').resolve()

# Local testing:
# centres = [0, 1, 2]
# target_shape = [176, 176]
# dataset = NiiDataset(0, centres, target_shape, use_cache=False)
# dataloader = iter(DataLoader(dataset, batch_size=int(np.ceil(len(dataset)/8))))
# data, truth = next(dataloader)
# print(data.shape)

# model = UNet(
#     spatial_dims=2,
#     in_channels=1,
#     out_channels=1,
#     channels=(16, 32, 64, 128, 256),
#     strides=(2, 2, 2, 2),
#     num_res_units=2,
# )

# state_dict = model.state_dict()
# global_layer_keys, local_layer_keys = get_lg_layer_keys(state_dict)

# freeze_layers(model, global_layer_keys)
# frozen = 0
# for name, param in model.named_parameters():
#     if param.requires_grad == False:
#         frozen += 1
# print("Global frozen:", frozen)

# freeze_layers(model, local_layer_keys)
# frozen = 0
# for name, param in model.named_parameters():
#     if param.requires_grad == False:
#         frozen += 1
# print("Local frozen:", frozen)
print(torch.load('./round_number.pt'))