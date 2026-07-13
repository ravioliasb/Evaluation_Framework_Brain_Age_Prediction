from monai.losses import DiceLoss
import torch
import matplotlib.pylab as plt
import numpy as np
from monai.metrics import DiceMetric
from monai.transforms import (
    AsDiscrete,
    Compose,
    EnsureType)
from monai.data.utils import pad_list_data_collate
from monai.data import decollate_batch
import wandb
from monai.inferers import sliding_window_inference, SimpleInferer, SlidingWindowInferer
from utils import *
from loss import *
import torch.nn.functional as F

def create_pad_mask(pred, target, brain_mask):
    for i in range(len(target.cpu().numpy())):

            ground_truth = torch.full_like(pred[i], target.cpu().numpy()[i]) #creates a tensor of size pred_shape and fills it with values target
            #noise = torch.randint_like(ground_truth, low=-2, high=3)
            ground_truth = torch.mul(ground_truth, brain_mask[i]) #tensor with target age values only in brain region

            prediction = torch.mul(pred[i], brain_mask[i]) #to ensure prediction tensor has non-zero values only in brain region
            loss_img = torch.sum(torch.abs(torch.sub(prediction,ground_truth, alpha=1))) / torch.sum(brain_mask[i]) #mean absolute error voxel wise
            pad_mask = torch.sub(prediction,ground_truth, alpha=1)
    return pad_mask, loss_img



def test_check1(test_loader, model, optimizer, scheduler, root_dir, path_to_chkpt):

    def model_fun1(input_img):
        pred_seg, pred_age, pred_vox_age = model(input_img)

        bg_ch = pred_seg[:, 0:1, :, :, :]
        f_ch = pred_seg[:, 1:2, :, :, :]
        s_ch = pred_seg[:, 2:3, :, :, :]
        t_ch = pred_seg[:, 3:4, :, :, :]

        pred_glob_age = torch.full_like(bg_ch, 0, dtype=torch.float32)

        for i in range(pred_age.size(0)):
            pred_glob_age[i,:,:,:,:] = pred_age[i].item()

        return bg_ch , f_ch, s_ch, t_ch, pred_glob_age, pred_vox_age

    state = torch.load(path_to_chkpt)
    model.load_state_dict(state['state_dict'])

    dir_name = str((path_to_chkpt.split('/')[-1]).split('.')[0])
    dir_name = 'camcan_1_' +(dir_name)
    path = os.path.join(root_dir,  dir_name)
    print(path)
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print('dir created')

    with torch.no_grad():
        mae_loss = 0.0
        voxel_error = []
        metric_dice = []
        for step, batch in enumerate(test_loader):
            img, age = (batch["img"].cuda(), batch["age_label"].cuda())
            full_input_path = batch["img"].meta["filename_or_obj"][0]
            
            # --- 1. CONSISTENT PATH LOGIC ---
            # Get filename (e.g., "subject_01.nii.gz")
            name = os.path.basename(full_input_path)
            # Get category folder (the parent directory of the image)
            parent_folder = os.path.basename(os.path.dirname(full_input_path))
            
            # Create: root_dir/experiment_name/category/subject_01.nii.gz/
            sub_path = os.path.join(path, parent_folder, name)
            os.makedirs(sub_path, exist_ok=True)

            expected_files = [
            f"age_{name}",
            f"seg_{name}",
            f"vox_age_raw_{name}",
            f"vox_age_pad_{name}",
            f"orig_{name}"
        ]

        # --- Check if every single file exists in the sub-path ---
            if all(os.path.exists(os.path.join(sub_path, f)) for f in expected_files):
                print(f"Skipping {name}: All output components already exist.")
                continue
            
            # --- 2. INFERENCE ---
            brain_img = img
            bg_ch, f_ch, s_ch, t_ch, pred_glob_age, pred_vox_age = sliding_window_inference(
                inputs=brain_img, roi_size=(128,128,128), sw_batch_size=2, 
                predictor=model_fun1, overlap=0.9, progress=False
            )

            pred_seg = torch.cat((bg_ch, f_ch, s_ch, t_ch), dim=1)
            pred_seg = torch.argmax(pred_seg, dim=1).unsqueeze(0)

            # Load header/affine from original image
            img1 = nib.load(full_input_path)

            # --- 3. SAVING ALL COMPONENTS ---
            
            # Save Global Age
            age_output = pred_glob_age.squeeze(0).squeeze(0).cpu().numpy()
            nib.save(nib.Nifti1Image(age_output, img1.affine, img1.header), 
                     os.path.join(sub_path, f"age_{name}"))

            # Save Segmentation
            seg_output = pred_seg.squeeze(0).squeeze(0).cpu().numpy()
            nib.save(nib.Nifti1Image(seg_output, img1.affine, img1.header), 
                     os.path.join(sub_path, f"seg_{name}"))

            # Save Raw Voxel Age (Masked)
            brain_mask = torch.where(brain_img > 0, 1, 0)
            raw_vox_output = torch.mul(pred_vox_age.squeeze(0), brain_mask.squeeze(0))
            raw_vox_output = raw_vox_output.squeeze(0).cpu().numpy()
            nib.save(nib.Nifti1Image(raw_vox_output, img1.affine, img1.header), 
                     os.path.join(sub_path, f"vox_age_raw_{name}"))

            # Save Voxel Age Pad/Error Mask
            brain_pad_mask, mae_voxel = create_pad_mask(pred_vox_age, age, brain_mask)
            voxel_error.append(mae_voxel.item())
            vox_age_output = brain_pad_mask.squeeze(0).squeeze(0).cpu().numpy()
            nib.save(nib.Nifti1Image(vox_age_output, img1.affine, img1.header), 
                     os.path.join(sub_path, f"vox_age_pad_{name}"))

            # Save Original Image (for reference)
            orig_output = brain_img.squeeze(0).squeeze(0).cpu().numpy()
            nib.save(nib.Nifti1Image(orig_output, img1.affine, img1.header), 
                     os.path.join(sub_path, f"orig_{name}"))

            print(f"Saved results for {name} in {parent_folder}")











