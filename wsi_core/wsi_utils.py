# Derived from mahmoodlab/CLAM; distributed under GPL-3.0.
import math
import h5py
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm


def save_hdf5(output_path, asset_dict, attr_dict=None, mode='a'):
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1,) + data_shape[1:]
            maxshape = (None,) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val
    file.close()
    return output_path

def DrawGrid(img, coord, shape, thickness=2, color=(0, 0, 0, 255)):
    cv2.rectangle(img, tuple(np.maximum([0, 0], coord - thickness // 2)), tuple(coord - thickness // 2 + np.array(shape)), (0, 0, 0, 255), thickness=thickness)
    return img

def DrawMapFromCoords(canvas, wsi_object, coords, patch_size, vis_level, indices=None, draw_grid=True):
    downsamples = wsi_object.wsi.level_downsamples[vis_level]
    if indices is None:
        indices = np.arange(len(coords))
    total = len(indices)
    patch_size = tuple(np.ceil(np.array(patch_size) / np.array(downsamples)).astype(np.int32))
    print('downscaled patch size: {}x{}'.format(patch_size[0], patch_size[1]))
    for idx in tqdm(range(total)):
        patch_id = indices[idx]
        coord = coords[patch_id]
        patch = np.array(wsi_object.wsi.read_region(tuple(coord), vis_level, patch_size).convert('RGB'))
        coord = np.ceil(coord / downsamples).astype(np.int32)
        canvas_crop_shape = canvas[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0], :3].shape[:2]
        canvas[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0], :3] = patch[:canvas_crop_shape[0], :canvas_crop_shape[1], :]
        if draw_grid:
            DrawGrid(canvas, coord, patch_size)
    return Image.fromarray(canvas)

def StitchCoords(hdf5_file_path, wsi_object, downscale=16, draw_grid=False, bg_color=(0, 0, 0), alpha=-1):
    wsi = wsi_object.getOpenSlide()
    w, h = wsi.level_dimensions[0]
    print('original size: {} x {}'.format(w, h))
    vis_level = wsi.get_best_level_for_downsample(downscale)
    w, h = wsi.level_dimensions[vis_level]
    print('downscaled size for stiching: {} x {}'.format(w, h))
    with h5py.File(hdf5_file_path, 'r') as file:
        dset = file['coords']
        coords = dset[:]
        print('start stitching {}'.format(dset.attrs['name']))
        patch_size = dset.attrs['patch_size']
        patch_level = dset.attrs['patch_level']
        level0_patch_size = dset.attrs.get('level0_patch_size', None)
    print(f'number of patches: {len(coords)}')
    print(f'patch size: {patch_size} x {patch_size} patch level: {patch_level}')
    if level0_patch_size is not None:
        patch_size = (int(level0_patch_size), int(level0_patch_size))
    else:
        patch_size = tuple((np.array((patch_size, patch_size)) * wsi.level_downsamples[patch_level]).astype(np.int32))
    print(f'ref patch size: {patch_size} x {patch_size}')
    if w * h > Image.MAX_IMAGE_PIXELS:
        raise Image.DecompressionBombError('Visualization Downscale %d is too large' % downscale)
    if alpha < 0 or alpha == -1:
        heatmap = Image.new(size=(w, h), mode='RGB', color=bg_color)
    else:
        heatmap = Image.new(size=(w, h), mode='RGBA', color=bg_color + (int(255 * alpha),))
    heatmap = np.array(heatmap)
    heatmap = DrawMapFromCoords(heatmap, wsi_object, coords, patch_size, vis_level, indices=None, draw_grid=draw_grid)
    return heatmap
