# Derived from mahmoodlab/CLAM; distributed under GPL-3.0.
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import h5py


class Whole_Slide_Bag_FP(Dataset):

    def __init__(self, file_path, wsi=None, wsi_path=None, img_transforms=None):
        """
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			img_transforms (callable, optional): Optional transform to be applied on a sample
		"""
        if wsi is None and wsi_path is None:
            raise ValueError('Either wsi or wsi_path must be provided')
        self.wsi = wsi
        self.wsi_path = wsi_path
        self.roi_transforms = img_transforms
        self.file_path = file_path
        with h5py.File(self.file_path, 'r') as f:
            dset = f['coords']
            attrs = f['coords'].attrs
            self.coords = dset[:]
            self.patch_level = int(attrs.get('source_level', attrs['patch_level']))
            self.patch_size = int(attrs.get('source_read_size', attrs['patch_size']))
            self.output_patch_size = int(attrs.get('output_patch_size', attrs['patch_size']))
            self.length = len(dset)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.wsi is None:
            from wsi_core.wsi_reader import open_wsi
            self.wsi = open_wsi(self.wsi_path)
        coord = self.coords[idx]
        img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
        if self.patch_size != self.output_patch_size:
            img = img.resize((self.output_patch_size, self.output_patch_size), Image.Resampling.BILINEAR)
        img = self.roi_transforms(img)
        return {'img': img, 'coord': coord}

    def __getstate__(self):
        state = self.__dict__.copy()
        state['wsi'] = None
        return state

    def close(self):
        if self.wsi is not None:
            self.wsi.close()
            self.wsi = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
