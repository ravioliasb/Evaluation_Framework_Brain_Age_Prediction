import monai
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss, DiceCELoss
from monai.inferers import sliding_window_inference, SimpleInferer
from monai.data import CacheDataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import glob
import numpy as np
from monai.utils import first, set_determinism
from monai.data import Dataset, ArrayDataset, DataLoader, load_decathlon_datalist
from monai.transforms import (Transform,AsDiscrete,Activations, Activationsd, Compose, LoadImaged,
                              Transposed, ScaleIntensityd, RandAxisFlipd, RandRotated, RandAxisFlipd,
                              RandBiasFieldd, ScaleIntensityRangePercentilesd, RandAdjustContrastd,
                              RandHistogramShiftd, DivisiblePadd, Orientationd, RandGibbsNoised, Spacingd,
                              RandRicianNoised, AsChannelLastd, RandSpatialCropd,ToNumpyd,EnsureChannelFirstd,
                              RandSpatialCropSamplesd, RandCropByPosNegLabeld, ThresholdIntensityd,)
from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import UNet, BasicUNet
from monai.data.utils import pad_list_data_collate
import pandas as pd
import os

from monai.transforms import MaskIntensityd

test_transforms = Compose(
    [
        # Load both the image and the mask
        LoadImaged(keys=["img", "mask"]),
        EnsureChannelFirstd(keys=["img", "mask"]),
        
        # Apply the mask to the image using the 'mask' key
        
        # Now scale the masked image
        ScaleIntensityd(
            keys=["img"],
            minv=0.0,
            maxv=1.0
        ),

        MaskIntensityd(
            keys=["img"], 
            mask_key="mask"
        ),
        
        # Clean up: remove the mask from the dictionary if you don't need it for training
        # This keeps the batch data light
        # RemoveDictKeysd(keys=["mask"]), 
        
        DivisiblePadd(["img"], 16),
    ]
)


def load_data_test(t1w_csv, seg_mask_csv, age_csv_path, brain_mask_csv, batch, root_dir):

    #testing on CC359
    
    file_name = "pediatric_test.csv"
    files = os.path.join(root_dir, file_name)
    shuff_data = pd.read_csv(files)
    imgs_list = list(shuff_data['imgs'])
    #masks_list = [path.replace(".nii.gz", "_mask_mask.nii.gz") for path in imgs_list]
    age_labels = list(shuff_data['age'])
    masks_list = list(shuff_data['mask'])

    # #only for camcan
    # length = len(imgs_list)
    # print(length)
    # test = int(0.85*length)

    # imgs_list = imgs_list[test:]
    # age_labels = age_labels[test:]

    filenames_test = [{"img": x, "mask": m, "age_label": z} for (x, m, z) in
                  zip(imgs_list, masks_list, age_labels)]

    # print('filenames train', filenames_train)
    ds_test = monai.data.Dataset(filenames_test, test_transforms)
    test_loader = DataLoader(ds_test, batch_size=1, shuffle=True, num_workers=2, pin_memory=True,
                             collate_fn=pad_list_data_collate)

    return ds_test, test_loader
