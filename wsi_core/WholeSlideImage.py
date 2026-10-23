# Derived from mahmoodlab/CLAM; distributed under GPL-3.0.
import math
import os
import itertools
import cv2
import numpy as np
from PIL import Image
from wsi_core.wsi_reader import open_wsi
from wsi_core.wsi_utils import save_hdf5
from wsi_core.util_classes import (isInContourV1, isInContourV2,
    isInContourV3_Easy, isInContourV3_Hard, Contour_Checking_fn)
from utils.file_utils import save_pkl
Image.MAX_IMAGE_PIXELS = 933120000


class WholeSlideImage(object):

    def __init__(self, path):
        """
        Args:
            path (str): fullpath to WSI file
        """
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.wsi = open_wsi(path)
        self.level_downsamples = self._assertLevelDownsamples()
        self.level_dim = self.wsi.level_dimensions
        self.contours_tissue = None
        self.contours_tumor = None

    def getOpenSlide(self):
        return self.wsi

    def close(self):
        self.wsi.close()

    def saveSegmentation(self, mask_file):
        asset_dict = {'holes': self.holes_tissue, 'tissue': self.contours_tissue}
        save_pkl(mask_file, asset_dict)

    def segmentTissue(self, seg_level=0, sthresh=20, sthresh_up=255, mthresh=7,
                      close=0, use_otsu=False, filter_params=None,
                      ref_patch_size=512, exclude_ids=(), keep_ids=()):
        """
            Segment the tissue via HSV -> Median thresholding -> Binary threshold
        """

        def _filter_contours(contours, hierarchy, filter_params):
            """
                Filter contours by: area.
            """
            filtered = []
            hierarchy_1 = np.flatnonzero(hierarchy[:, 1] == -1)
            all_holes = []
            for cont_idx in hierarchy_1:
                cont = contours[cont_idx]
                holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)
                a = cv2.contourArea(cont)
                hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in holes]
                a = a - np.array(hole_areas).sum()
                if a == 0:
                    continue
                if tuple((filter_params['a_t'],)) < tuple((a,)):
                    filtered.append(cont_idx)
                    all_holes.append(holes)
            foreground_contours = [contours[cont_idx] for cont_idx in filtered]
            hole_contours = []
            for hole_ids in all_holes:
                unfiltered_holes = [contours[idx] for idx in hole_ids]
                unfilered_holes = sorted(unfiltered_holes, key=cv2.contourArea, reverse=True)
                unfilered_holes = unfilered_holes[:filter_params['max_n_holes']]
                filtered_holes = []
                for hole in unfilered_holes:
                    if cv2.contourArea(hole) > filter_params['a_h']:
                        filtered_holes.append(hole)
                hole_contours.append(filtered_holes)
            return (foreground_contours, hole_contours)
        img = np.array(self.wsi.read_region((0, 0), seg_level, self.level_dim[seg_level]))
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img_med = cv2.medianBlur(img_hsv[:, :, 1], mthresh)
        if use_otsu:
            _, img_otsu = cv2.threshold(img_med, 0, sthresh_up, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        else:
            _, img_otsu = cv2.threshold(img_med, sthresh, sthresh_up, cv2.THRESH_BINARY)
        if close > 0:
            kernel = np.ones((close, close), np.uint8)
            img_otsu = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel)
        scale = self.level_downsamples[seg_level]
        scaled_ref_patch_area = int(ref_patch_size ** 2 / (scale[0] * scale[1]))
        filter_params = (filter_params or {'a_t': 100, 'a_h': 16, 'max_n_holes': 8}).copy()
        filter_params['a_t'] = filter_params['a_t'] * scaled_ref_patch_area
        filter_params['a_h'] = filter_params['a_h'] * scaled_ref_patch_area
        contours, hierarchy = cv2.findContours(img_otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if hierarchy is None:
            self.contours_tissue, self.holes_tissue = [], []
            return
        hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]
        if filter_params:
            foreground_contours, hole_contours = _filter_contours(contours, hierarchy, filter_params)
        self.contours_tissue = self.scaleContourDim(foreground_contours, scale)
        self.holes_tissue = self.scaleHolesDim(hole_contours, scale)
        if len(keep_ids) > 0:
            contour_ids = set(keep_ids) - set(exclude_ids)
        else:
            contour_ids = set(np.arange(len(self.contours_tissue))) - set(exclude_ids)
        self.contours_tissue = [self.contours_tissue[i] for i in contour_ids]
        self.holes_tissue = [self.holes_tissue[i] for i in contour_ids]

    def visWSI(self, vis_level=0, color=(0, 255, 0), hole_color=(0, 0, 255), annot_color=(255, 0, 0), line_thickness=250, max_size=None, top_left=None, bot_right=None, custom_downsample=1, view_slide_only=False, number_contours=False, seg_display=True, annot_display=True):
        downsample = self.level_downsamples[vis_level]
        scale = [1 / downsample[0], 1 / downsample[1]]
        if top_left is not None and bot_right is not None:
            top_left = tuple(top_left)
            bot_right = tuple(bot_right)
            w, h = tuple((np.array(bot_right) * scale).astype(int) - (np.array(top_left) * scale).astype(int))
            region_size = (w, h)
        else:
            top_left = (0, 0)
            region_size = self.level_dim[vis_level]
        img = np.array(self.wsi.read_region(top_left, vis_level, region_size).convert('RGB'))
        if not view_slide_only:
            offset = tuple(-(np.array(top_left) * scale).astype(int))
            line_thickness = int(line_thickness * math.sqrt(scale[0] * scale[1]))
            if self.contours_tissue is not None and seg_display:
                if not number_contours:
                    cv2.drawContours(img, self.scaleContourDim(self.contours_tissue, scale), -1, color, line_thickness, lineType=cv2.LINE_8, offset=offset)
                else:
                    for idx, cont in enumerate(self.contours_tissue):
                        contour = np.array(self.scaleContourDim(cont, scale))
                        M = cv2.moments(contour)
                        cX = int(M['m10'] / (M['m00'] + 1e-09))
                        cY = int(M['m01'] / (M['m00'] + 1e-09))
                        cv2.drawContours(img, [contour], -1, color, line_thickness, lineType=cv2.LINE_8, offset=offset)
                        cv2.putText(img, '{}'.format(idx), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 10)
                for holes in self.holes_tissue:
                    cv2.drawContours(img, self.scaleContourDim(holes, scale), -1, hole_color, line_thickness, lineType=cv2.LINE_8)
            if self.contours_tumor is not None and annot_display:
                cv2.drawContours(img, self.scaleContourDim(self.contours_tumor, scale), -1, annot_color, line_thickness, lineType=cv2.LINE_8, offset=offset)
        img = Image.fromarray(img)
        w, h = img.size
        if custom_downsample > 1:
            img = img.resize((int(w / custom_downsample), int(h / custom_downsample)))
        if max_size is not None and (w > max_size or h > max_size):
            resizeFactor = max_size / w if w > h else max_size / h
            img = img.resize((int(w * resizeFactor), int(h * resizeFactor)))
        return img

    @staticmethod
    def isInHoles(holes, pt, patch_size):
        for hole in holes:
            if cv2.pointPolygonTest(hole, (pt[0] + patch_size / 2, pt[1] + patch_size / 2), False) > 0:
                return 1
        return 0

    @staticmethod
    def isInContours(cont_check_fn, pt, holes=None, patch_size=256):
        if cont_check_fn(pt):
            if holes is not None:
                return not WholeSlideImage.isInHoles(holes, pt, patch_size)
            else:
                return 1
        return 0

    @staticmethod
    def scaleContourDim(contours, scale):
        return [np.array(cont * scale, dtype='int32') for cont in contours]

    @staticmethod
    def scaleHolesDim(contours, scale):
        return [[np.array(hole * scale, dtype='int32') for hole in holes] for holes in contours]

    def _assertLevelDownsamples(self):
        level_downsamples = []
        dim_0 = self.wsi.level_dimensions[0]
        for downsample, dim in zip(self.wsi.level_downsamples, self.wsi.level_dimensions):
            estimated_downsample = (dim_0[0] / float(dim[0]), dim_0[1] / float(dim[1]))
            level_downsamples.append(estimated_downsample) if estimated_downsample != (downsample, downsample) else level_downsamples.append((downsample, downsample))
        return level_downsamples

    def process_contours(self, save_path, patch_level=0, patch_size=256, step_size=256, **kwargs):
        save_path_hdf5 = os.path.join(save_path, str(self.name) + '.h5')
        print('Creating patches for: ', self.name, '...')
        n_contours = len(self.contours_tissue)
        print('Total number of contours to process: ', n_contours)
        fp_chunk_size = math.ceil(n_contours * 0.05)
        init = True
        for idx, cont in enumerate(self.contours_tissue):
            if (idx + 1) % fp_chunk_size == fp_chunk_size:
                print('Processing contour {}/{}'.format(idx, n_contours))
            asset_dict, attr_dict = self.process_contour(cont, self.holes_tissue[idx], patch_level, save_path, patch_size, step_size, **kwargs)
            if len(asset_dict) > 0:
                if init:
                    save_hdf5(save_path_hdf5, asset_dict, attr_dict, mode='w')
                    init = False
                else:
                    save_hdf5(save_path_hdf5, asset_dict, mode='a')
        return save_path_hdf5 if not init else None

    def process_contour(self, cont, contour_holes, patch_level, save_path,
                        patch_size=256, step_size=256, contour_fn='four_pt',
                        use_padding=True, top_left=None, bot_right=None, target_mpp=None):
        start_x, start_y, w, h = cv2.boundingRect(cont) if cont is not None else (0, 0, self.level_dim[patch_level][0], self.level_dim[patch_level][1])
        if target_mpp is not None:
            base_mpp = getattr(self.wsi, 'base_mpp', None)
            if base_mpp is None or not np.isfinite(base_mpp) or base_mpp <= 0:
                raise ValueError('target_mpp requires valid level-0 MPP metadata')
            target_mpp = float(target_mpp)
            if not np.isfinite(target_mpp) or target_mpp <= 0:
                raise ValueError('target_mpp must be positive and finite')
            desired_downsample = target_mpp / float(base_mpp)
            patch_level = self.wsi.get_best_level_for_downsample(desired_downsample)
            native_mpp = float(base_mpp) * float(self.level_downsamples[patch_level][0])
            source_read_size = max(1, int(round(patch_size * target_mpp / native_mpp)))
            ref_patch_edge = max(1, int(round(patch_size * target_mpp / float(base_mpp))))
            ref_step_edge = max(1, int(round(step_size * target_mpp / float(base_mpp))))
            ref_patch_size = (ref_patch_edge, ref_patch_edge)
        else:
            patch_downsample = (int(self.level_downsamples[patch_level][0]), int(self.level_downsamples[patch_level][1]))
            ref_patch_size = (patch_size * patch_downsample[0], patch_size * patch_downsample[1])
            ref_step_edge = step_size * patch_downsample[0]
            source_read_size = patch_size
            base_mpp = np.nan
            native_mpp = np.nan
        img_w, img_h = self.level_dim[0]
        if use_padding:
            stop_y = start_y + h
            stop_x = start_x + w
        else:
            stop_y = min(start_y + h, img_h - ref_patch_size[1] + 1)
            stop_x = min(start_x + w, img_w - ref_patch_size[0] + 1)
        print('Bounding Box:', start_x, start_y, w, h)
        print('Contour Area:', cv2.contourArea(cont))
        if bot_right is not None:
            stop_y = min(bot_right[1], stop_y)
            stop_x = min(bot_right[0], stop_x)
        if top_left is not None:
            start_y = max(top_left[1], start_y)
            start_x = max(top_left[0], start_x)
        if bot_right is not None or top_left is not None:
            w, h = (stop_x - start_x, stop_y - start_y)
            if w <= 0 or h <= 0:
                print('Contour is not in specified ROI, skip')
                return ({}, {})
            else:
                print('Adjusted Bounding Box:', start_x, start_y, w, h)
        if isinstance(contour_fn, str):
            if contour_fn == 'four_pt':
                cont_check_fn = isInContourV3_Easy(contour=cont, patch_size=ref_patch_size[0], center_shift=0.5)
            elif contour_fn == 'four_pt_hard':
                cont_check_fn = isInContourV3_Hard(contour=cont, patch_size=ref_patch_size[0], center_shift=0.5)
            elif contour_fn == 'center':
                cont_check_fn = isInContourV2(contour=cont, patch_size=ref_patch_size[0])
            elif contour_fn == 'basic':
                cont_check_fn = isInContourV1(contour=cont)
            else:
                raise NotImplementedError
        else:
            assert isinstance(contour_fn, Contour_Checking_fn)
            cont_check_fn = contour_fn
        step_size_x = ref_step_edge
        step_size_y = ref_step_edge
        x_range = np.arange(start_x, stop_x, step=step_size_x)
        y_range = np.arange(start_y, stop_y, step=step_size_y)
        x_coords, y_coords = np.meshgrid(x_range, y_range, indexing='ij')
        coord_candidates = np.array([x_coords.flatten(), y_coords.flatten()]).transpose()
        iterable = [(coord, contour_holes, ref_patch_size[0], cont_check_fn) for coord in coord_candidates]
        results = itertools.starmap(WholeSlideImage.process_coord_candidate, iterable)
        results = np.array([result for result in results if result is not None])
        print('Extracted {} coordinates'.format(len(results)))
        if len(results) > 0:
            asset_dict = {'coords': results}
            attr = {
                'patch_size': patch_size,
                'patch_level': patch_level,
                'source_level': patch_level,
                'source_read_size': source_read_size,
                'output_patch_size': patch_size,
                'level0_patch_size': ref_patch_size[0],
                'level0_step_size': ref_step_edge,
                'base_mpp': base_mpp,
                'target_mpp': float(target_mpp) if target_mpp is not None else np.nan,
                'native_level_mpp': native_mpp,
                'downsample': self.level_downsamples[patch_level],
                'downsampled_level_dim': self.level_dim[patch_level],
                'level_dim': self.level_dim[patch_level],
                'name': self.name,
                'save_path': save_path,
            }
            attr_dict = {'coords': attr}
            return (asset_dict, attr_dict)
        else:
            return ({}, {})

    @staticmethod
    def process_coord_candidate(coord, contour_holes, ref_patch_size, cont_check_fn):
        if WholeSlideImage.isInContours(cont_check_fn, coord, contour_holes, ref_patch_size):
            return coord
        else:
            return None
