#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from typing import List, Set, Dict, Union, Tuple, Optional

import os
import dataclasses
import PIL
import PIL.IcnsImagePlugin
import skimage
import openslide
import numpy as np
import json

from tqdm import tqdm

@dataclasses.dataclass
class WSIPatchExtract:
    """Extract image patches for either the H&E or mIF WSI."""

    def __init__(
        self,
        thumbnail_width: int = 500,
        patch_level: int = 0,
        max_patch_size: int = 1024,
        tissue_area_threshold: float = 0.0
    ) -> None:
        super(WSIPatchExtract).__init__()
        self.thumbnail_width = thumbnail_width
        self.patch_level = patch_level
        self.max_patch_size = max_patch_size
        self.tissue_area_threshold = tissue_area_threshold

    def _get_wsi_thumbnail(
        self,
        wsi_dir: str,
    ) -> Union[openslide.OpenSlide, List[int], PIL.Image.Image]:
        """Create the thumbnail image for WSI.

        Parameters
        ----------
        wsi_dir : str
            directory of input WSI.

        Returns
        -------
        Union[openslide.OpenSlide, list[int], PIL.Image.Image]
            return the openslide WSI object, top_level with patch width,
            height, and the ratio for the target image width.
        """
        print(wsi_dir)
        wsi = openslide.OpenSlide(filename=wsi_dir)

        # Get the ratio for the target image width.
        divisor = int(wsi.level_dimensions[0][0] / self.thumbnail_width)
        # Get the height and width of the thumbnail using the ratio.
        patch_size_x = int(wsi.level_dimensions[0][0] / divisor)
        patch_size_y = int(wsi.level_dimensions[0][1] / divisor)
        top_level = [patch_size_x, patch_size_y, divisor]
        # Extract the thumbnail.
        thumbnail = wsi.get_thumbnail(size=(patch_size_x, patch_size_y))
        return wsi, top_level, thumbnail

    def _get_ihc_thumbnail(
        self,
        ihc_dir: str,
        wsi_downsample: int
    ) -> Union[openslide.OpenSlide, List[int], PIL.Image.Image]:
        """Create the thumbnail image for WSI.

        Parameters
        ----------
        wsi_dir : str
            directory of input WSI.

        Returns
        -------
        Union[openslide.OpenSlide, list[int], PIL.Image.Image]
            return the openslide WSI object, top_level with patch width,
            height, and the ratio for the target image width.
        """
        wsi = openslide.OpenSlide(filename=ihc_dir)
        for i in range(len(wsi.level_downsamples)):
            if wsi.level_downsamples[i] == wsi_downsample:
                self.ihc_patch_level = i
                print("IHC Corresponding Patch Level", self.ihc_patch_level)
        # Get the ratio for the target image width.
        divisor = int(wsi.level_dimensions[0][0] / self.thumbnail_width)
        # Get the height and width of the thumbnail using the ratio.
        patch_size_x = int(wsi.level_dimensions[0][0] / divisor)
        patch_size_y = int(wsi.level_dimensions[0][1] / divisor)
        top_level = [patch_size_x, patch_size_y, divisor]
        # Extract the thumbnail.
        thumbnail = wsi.get_thumbnail(size=(patch_size_x, patch_size_y))
        return wsi, top_level, thumbnail
        
    def _get_otsu_binary_img(self, thumbnail: PIL.Image.Image) -> np.ndarray:
        """Create binary mask image for tissue detection using OTSU method.

        Parameters
        ----------
        thumbnail : PIL.Image.Image
            The thumbnail image of WSI.

        Returns
        -------
        np.ndarray
            binary mask image in numpy array.
        """
        # Convert to grey scale image.
        gs_thumbnail = np.array(thumbnail.convert("L"))
        # Get the otsu threshold value.
        thresh = skimage.filters.threshold_otsu(image=gs_thumbnail)
        # Convert to binary mask.
        binary_img = gs_thumbnail < thresh
        binary_img = binary_img.astype(int)
        return binary_img

    def _locate_tissue_regions(self, binary_img: np.ndarray) -> Set[Tuple[int]]:
        """Locate the x- and y- coords from binary image mask with tissue pixel values.

        Parameters
        ----------
        binary_img : np.ndarray
            binary mask image array.

        Returns
        -------
        set[Tuple[int]]
            set of x- and y- coords for tissue regions.
        """
        idx = np.sum(binary_img)
        idx = np.where(binary_img == 1)
        tissue_region_index = []
        for i in range(0, len(idx[0]), 1):
            x = idx[1][i]
            y = idx[0][i]
            tissue_region_index.append((x, y))

        return set(tissue_region_index)

    def _get_patch_coords(
        self,
        wsi: openslide.OpenSlide,
        divisor: int,
        tissue_region_index: Set[Tuple[int]],
    ) -> Set[Tuple[int]]:
        """Get the x- and y- coords for patches that needs to be extracted.

        Parameters
        ----------
        wsi : openslide.OpenSlide
            openslide WSI object.
        divisor : int
            ratio for the target image width.
        tissue_region_index : set[Tuple[int]]
            set of x- and y- coords for tissue regions.

        Returns
        -------
        Set[Tuple[int]]
            set of tuple of x- and y- coords for patches that needs to be extracted.
        """
        assert (
            self.patch_level <= len(wsi.level_dimensions) - 1
        ), f"level {str(self.patch_level)} exceeds {str(len(wsi.level_dimensions)-1)}"

        patch_start_xy_coords = []

        # creating sub patches
        # Iterating through x coordinate
        wsi_level_dims = wsi.level_dimensions[self.patch_level]
        wsi_level_downsamples = wsi.level_downsamples[self.patch_level]
       
        start_x = 0
        while start_x + self.max_patch_size < wsi_level_dims[0]:
            # Iterating through y coordinate
            start_y = 0
            while start_y + self.max_patch_size < wsi_level_dims[1]:
                current_x = int(start_x * wsi_level_downsamples / divisor)
                current_y = int(start_y * wsi_level_downsamples / divisor)
                current_x_stop = int(
                    ((start_x + self.max_patch_size) * wsi_level_downsamples) / divisor
                )
                current_y_stop = int(
                    ((start_y + self.max_patch_size) * wsi_level_downsamples) / divisor
                )

                tissue_pixels = sum(
                    (i, j) in tissue_region_index
                    for i in range(current_x, current_x_stop + 1)
                    for j in range(current_y, current_y_stop + 1)
                )

                if (
                    tissue_pixels
                    / (
                        (current_y_stop + 1 - current_y)
                        * (current_x_stop + 1 - current_x)
                    )
                ) > self.tissue_area_threshold:
                    patch_start_xy_coords.append((start_x, start_y))
                start_y += self.max_patch_size
            start_x += self.max_patch_size
        return set(patch_start_xy_coords)

    def _extract_patches(
        self,
        wsi_dir: str,
        wsi: openslide.OpenSlide,
        er: openslide.OpenSlide,
        her2: openslide.OpenSlide,
        ki67: openslide.OpenSlide,
        pgr: openslide.OpenSlide,

        patch_start_xy_coords: Set[Tuple[int]],
    ) -> Union[str, Dict[str, np.ndarray]]:
        """Extract image patches from WSI.

        Parameters
        ----------
        wsi_dir : str
            directory of input WSI.
        wsi : openslide.OpenSlide
            openslide WSI object.
        patch_start_xy_coords : Set[Tuple[int]]
            set of tuple of x- and y- coords for patches that needs to be extracted.

        Returns
        -------
        Union[str, dict[str, np.ndarray]]
            return the unique identifier for the de-identified WSI, and the
            dictionary with key be patch name, value be the patch object in np.ndarray.
        """
        wsi_uuid = wsi_dir.split("/")[-1].split(".")[0]
        patch_names = []
        patch_objs = []

        for coords in tqdm(patch_start_xy_coords):
            start_x_coord = int(coords[0] * wsi.level_downsamples[self.patch_level])
            start_y_coord = int(coords[-1] * wsi.level_downsamples[self.patch_level])
            patch_obj = wsi.read_region(
                location=(start_x_coord, start_y_coord),
                level=self.patch_level,
                size=(self.max_patch_size, self.max_patch_size),
            )
            
            if er is not None:
                er_patch_obj = er.read_region(
                    location=(start_x_coord, start_y_coord),
                    level=self.ihc_patch_level,
                    size=(self.max_patch_size, self.max_patch_size),
                )

            if her2 is not None:
                her2_patch_obj = her2.read_region(
                    location=(start_x_coord, start_y_coord),
                    level=self.ihc_patch_level,
                    size=(self.max_patch_size, self.max_patch_size),
                )


            if ki67 is not None:
                ki67_patch_obj = ki67.read_region(
                    location=(start_x_coord, start_y_coord),
                    level=self.ihc_patch_level,
                    size=(self.max_patch_size, self.max_patch_size),
                )

            if pgr is not None:
                pgr_patch_obj = pgr.read_region(
                    location=(start_x_coord, start_y_coord),
                    level=self.ihc_patch_level,
                    size=(self.max_patch_size, self.max_patch_size),
                )

            patch_name = f"{wsi_uuid}_level_{self.patch_level}_x_{start_x_coord}_y_{start_y_coord}"

            patch_names.append(patch_name)

            patch_obs = {}
            patch_obs["he"] = np.asarray(patch_obj)
            if er is not None:
                patch_obs["er"] = np.asarray(er_patch_obj)

            if her2 is not None:
                patch_obs["her2"] = np.asarray(her2_patch_obj)


            if ki67 is not None:
                patch_obs["ki67"] = np.asarray(ki67_patch_obj)

            if pgr is not None:
                patch_obs["pgr"] = np.asarray(pgr_patch_obj)

            patch_objs.append({
                "he": np.asarray(patch_obj),
                "er": np.asarray(er_patch_obj),
                "her2": np.asarray(her2_patch_obj),
                "ki67": np.asarray(ki67_patch_obj),
                "pgr": np.asarray(pgr_patch_obj)
            })
            
        return wsi_uuid, dict(zip(patch_names, patch_objs))

    def _save_patches(
        self, wsi_uuid: str, patch_dicts: Dict[str, np.ndarray], patch_dir: str
    ) -> None:
        """Save extracted image patches.

        Parameters
        ----------
        wsi_uuid: str
            unique identifier for the de-identified WSI.
        patch_dicts : dict[str, np.ndarray]
            dictionary with key be patch name, value be the patch object in np.ndarray.
        patch_dir : str
            directory of output image patches extracted from WSI.
        """

        for name, patch in tqdm(patch_dicts.items()):
            np.save(f"{patch_dir}/{wsi_uuid}/he/{name}.npy", patch["he"])

            if patch["er"]:
                np.save(f"{patch_dir}/{wsi_uuid}/er/{name}.npy", patch["er"])

            if patch["her2"]:
                np.save(f"{patch_dir}/{wsi_uuid}/her2/{name}.npy", patch["her2"])

            if patch["ki67"]:
                np.save(f"{patch_dir}/{wsi_uuid}/ki67/{name}.npy", patch["ki67"])

            if patch["pgr"]:
                np.save(f"{patch_dir}/{wsi_uuid}/pgr/{name}.npy", patch["pgr"])


    def forward(
        self, input_dir, he: str, er:str, her2:str, ki67:str, pgr:str, output_dir: str
    ) -> Optional[Dict[str, np.ndarray]]:
        """Main function to execute the patch extraction from WSIPatchExtract class.

        Returns
        -------
        Optional[dict[str, np.ndarray]]
            dictionary with key be patch name, value be the patch object in np.ndarray.
        """
        print(os.path.exists(er), os.path.exists(her2), os.path.exists(ki67), os.path.exists(pgr))

        wsi, wsi_top_level, wsi_thumbnail = self._get_wsi_thumbnail(he)
        wsi_uuid = input_dir.split("/")[-1].split(".")[0]
        os.makedirs(f"{output_dir}/{wsi_uuid}/he", exist_ok=True)

        if os.path.exists(er):
            er_ihc, er_top_level, er_thumbnail = self._get_ihc_thumbnail(er, wsi.level_downsamples[self.patch_level])
            os.makedirs(f"{output_dir}/{wsi_uuid}/er", exist_ok=True)
        else:
            er_ihc = None
        if os.path.exists(her2):
            her2_ihc, her2_top_level, her2_thumbnail = self._get_ihc_thumbnail(her2, wsi.level_downsamples[self.patch_level])
            os.makedirs(f"{output_dir}/{wsi_uuid}/her2", exist_ok=True)
        else:
            her2_ihc = None
            
        if os.path.exists(ki67):
            ki67_ihc, ki67_top_level, ki67_thumbnail = self._get_ihc_thumbnail(ki67, wsi.level_downsamples[self.patch_level])
            os.makedirs(f"{output_dir}/{wsi_uuid}/ki67", exist_ok=True)
        else:
            ki67_ihc = None
        
        if os.path.exists(pgr):
            pgr_ihc, pgr_top_level, pgr_thumbnail = self._get_ihc_thumbnail(pgr, wsi.level_downsamples[self.patch_level])
            os.makedirs(f"{output_dir}/{wsi_uuid}/pgr", exist_ok=True)
        else:
            pgr_ihc = None
            
        binary_img = self._get_otsu_binary_img(wsi_thumbnail)
        tissue_region_index = self._locate_tissue_regions(binary_img)
        patch_start_xy_coords = self._get_patch_coords(
            wsi, wsi_top_level[-1], tissue_region_index
        )
        wsi_uuid, patch_dicts = self._extract_patches(
            input_dir, wsi, er_ihc, her2_ihc, ki67_ihc, pgr_ihc, patch_start_xy_coords
        )
        
        if output_dir is None:
            return patch_dicts
        self._save_patches(wsi_uuid, patch_dicts, output_dir)
        return None


# In[ ]:


wsi_root = "/ix1/qgu/ngl18/ACROBAT_Valis_Batch2/"
wsi_list = sos.listdir(wsi_root)


# In[ ]:


# os.makedirs("/ix1/qgu/ngl18/ACROBAT_VirtualStaining_Valis_10x")


# In[ ]:


extractor = WSIPatchExtract(max_patch_size=256, tissue_area_threshold=0.9, patch_level=0)


# In[ ]:





# In[ ]:


for wsi in wsi_list:
    # Check to ensure all four IHCs exist
    he_file =  wsi_root + f"{wsi}/{wsi}_HE_train.ome.tiff"
    er_file = wsi_root + f"{wsi}/{wsi}_ER_train.ome.tiff"
    her2_file = wsi_root + f"{wsi}/{wsi}_HER2_train.ome.tiff"
    ki67_file = wsi_root + f"{wsi}/{wsi}_KI67_train.ome.tiff"
    pgr_file = wsi_root + f"{wsi}/{wsi}_PGR_train.ome.tiff" 

    extractor.forward(wsi_root + f"{wsi}", he_file, er_file, her2_file, ki67_file, pgr_file, 
                          "/ix1/qgu/ngl18/ACROBAT_VirtualStaining_Valis_10x")


# In[ ]:




