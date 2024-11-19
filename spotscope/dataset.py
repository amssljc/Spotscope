import torch
import numpy as np
import torchvision.transforms.functional as TF
import random
from PIL import Image
import scanpy as sc
from torch.utils.data import DataLoader, random_split, ConcatDataset, WeightedRandomSampler
from .utils import find_values_from_mult_idict
from scanpy import AnnData

# update

class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, adata_st, annotation=True, transform=True, patch_size = 224):
        
        self.annotation = annotation
        self.transform = transform
        if isinstance(adata_st, str):
            adata_st = sc.read_h5ad(adata_st)
        elif isinstance(adata_st, AnnData):
            pass
            
        self.patch_size = patch_size
        self.whole_image = find_values_from_mult_idict(adata_st.uns, 'hires')
        self.h, self.w = self.whole_image.shape[:2]
        self.spatial_pos = adata_st.obsm['spatial']
        self.scale_factor = find_values_from_mult_idict(adata_st.uns, 'tissue_hires_scalef')
        if annotation:
            self.annotations = adata_st.obsm['annotations']
        
        # print("Finished loading all files")

    def transform_img(self, image):
        image = Image.fromarray(image)
        # Random flipping and rotations
        if random.random() > 0.5:
            image = TF.hflip(image)
        if random.random() > 0.5:
            image = TF.vflip(image)
        angle = random.choice([180, 90, 0, -90])
        image = TF.rotate(image, angle)
        return np.asarray(image)

    def __getitem__(self, idx):
        item = {}
        y_coord, x_coord = self.spatial_pos[idx, :]
        x_coord = int(x_coord * self.scale_factor)
        y_coord = int(y_coord * self.scale_factor)
        image = self.whole_image[(x_coord- self.patch_size//2):(x_coord+self.patch_size//2),(y_coord-self.patch_size//2):(y_coord+self.patch_size//2)]
        if self.transform:
            image = self.transform_img(image)
        
        item['image'] = torch.tensor(image).permute(2, 0, 1).float() #color channel first, then XY
        item['spatial_coords'] = [x_coord, y_coord]
        item['relative_coords'] = torch.tensor([x_coord / self.h, y_coord / self.w]).float()  
        if self.annotation:
            item['annotations'] = torch.tensor(self.annotations[idx, :]).float()  

        return item


    def __len__(self):
        return len(self.spatial_pos)
    
    
 
def build_train_loaders(batch_size, dataset_paths= [
        "/mnt/zj-gpfs/home/lengjiacheng/project/BLEEP_my/data/mouse_bolb/Rep1_MOB_test.h5ad",
        "/mnt/zj-gpfs/home/lengjiacheng/project/BLEEP_my/data/mouse_bolb/Rep2_MOB_test.h5ad",
        "/mnt/zj-gpfs/home/lengjiacheng/project/BLEEP_my/data/mouse_bolb/Rep3_MOB_test.h5ad",
        "/mnt/zj-gpfs/home/lengjiacheng/project/BLEEP_my/data/mouse_bolb/Rep4_MOB_test.h5ad",
        "/mnt/zj-gpfs/home/lengjiacheng/project/BLEEP_my/data/mouse_bolb/Rep5_MOB_test.h5ad",
        "/mnt/zj-gpfs/home/lengjiacheng/project/BLEEP_my/data/mouse_bolb/Rep6_MOB_test.h5ad",
    ], train_ratio=0.9, task='discrete', pin_memory=True,num_workers=5
):
    if task not in ['discrete', 'continuous']:
        raise ValueError("Parameter 'task' must be either 'discrete' or 'continuous'")
    
    print("Building loaders")
    datasets = [CLIPDataset(path, annotation=True, transform=True) for path in dataset_paths]
    dataset = ConcatDataset(datasets)

    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    print(len(train_dataset), len(test_dataset), "train/test split completed")

    if task == 'discrete':
        # Collect annotations and compute weights for one-hot encoded classes
        annotations = torch.vstack([data['annotations'] for data in train_dataset])
        class_counts = torch.sum(annotations, axis=0)
        class_weights = 1. / class_counts
        sample_weights = annotations @ class_weights
        
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, pin_memory=pin_memory, drop_last=False,num_workers=num_workers)
    elif task == 'continuous':
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, drop_last=False,num_workers=num_workers)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, drop_last=False,num_workers=num_workers)
    print("Finished building loaders")
    return train_loader, test_loader


def build_loaders_references(dataset_paths= [
        "/mnt/zj-gpfs/home/lengjiacheng/project/BLEEP_my/data/mouse_bolb/Rep1_MOB_test.h5ad",
        "/mnt/zj-gpfs/home/lengjiacheng/project/BLEEP_my/data/mouse_bolb/Rep2_MOB_test.h5ad",
        "/mnt/zj-gpfs/home/lengjiacheng/project/BLEEP_my/data/mouse_bolb/Rep3_MOB_test.h5ad",
        "/mnt/zj-gpfs/home/lengjiacheng/project/BLEEP_my/data/mouse_bolb/Rep4_MOB_test.h5ad",
        "/mnt/zj-gpfs/home/lengjiacheng/project/BLEEP_my/data/mouse_bolb/Rep5_MOB_test.h5ad",
        "/mnt/zj-gpfs/home/lengjiacheng/project/BLEEP_my/data/mouse_bolb/Rep6_MOB_test.h5ad",
    ],num_workers=5):
    print("Building reference loaders")
    # get annotation_list
    adata_ = sc.read_h5ad(dataset_paths[0])
    annotation_list = adata_.uns['annotation_list']
    
    datasets = [CLIPDataset(path, annotation=True, transform=False) for path in dataset_paths]
    data_size = [len(i) for i in datasets]
    dataset = torch.utils.data.ConcatDataset(datasets)
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False, pin_memory=True, drop_last=False,num_workers=num_workers)
    print("Finished building reference loaders")
    return test_loader, dataset, data_size, annotation_list

def build_loaders_querys(adata,num_workers=5):
    print("Building query loaders")
    datasets = [CLIPDataset(adata, annotation=False, transform=False)]
    data_size = [len(i) for i in datasets]
    dataset = torch.utils.data.ConcatDataset(datasets)
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False, pin_memory=True, drop_last=False,num_workers=num_workers)
    print("Finished building query loaders")
    return test_loader,dataset, data_size

def load_reference_datasets(adata, dataset_paths):
    if dataset_paths is None:
        assert 'annotation_list' in adata.uns.keys() and len(adata.uns['annotation_list'])>0, "Please input annotation list."
    dataloader, dataset, data_size, annotation_list = build_loaders_references(dataset_paths)
    adata.uns['reference_dataloaders'] = dataloader
    adata.uns['reference_dataset'] = dataset
    adata.uns['reference_data_size'] = data_size
    adata.uns['annotation_list'] = annotation_list
        
def load_query_datasets(adata,):
    dataloader,dataset, data_size = build_loaders_querys(adata)
    adata.uns['query_dataloaders'] = dataloader
    adata.uns['query_dataset'] = dataset
    adata.uns['query_data_size'] = data_size