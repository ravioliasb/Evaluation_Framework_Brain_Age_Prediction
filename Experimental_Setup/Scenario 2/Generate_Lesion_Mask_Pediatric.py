import numpy as np
import nibabel as nib
import os
import sys

# --- Configuration (Must match your generation script exactly) ---
ROOT_OUTPUT_FOLDER = '/home/ravi.bullock/Test_File_Info/SPIEExtension_Lesioned/Pediatric'
MASK_OUTPUT_BASE = '/home/ravi.bullock/Test_File_Info/SPIEExtension_Lesioned/Pediatric_Masks'
MASK_BASE_DIR = '/work/wilms_lab/ravi/SPIE_Extension_Samples/PediatricSynthetic/Masks'

FIXED_SLICE_Z = 95
SMALL, MEDIUM, LARGE = 20, 30, 50

def generate_binary_lesion_mask(shape, lesion_center_coords, lesion_diameter_mm, voxel_dims):
    """Generates a 1/0 binary mask based on the ellipsoid math."""
    mask = np.zeros(shape, dtype=np.uint8)
    center_x, center_y, center_z = lesion_center_coords

    # Replicating your ellipsoid axis logic
    a_vox = (lesion_diameter_mm / 4.0) / voxel_dims[0]
    b_vox = (lesion_diameter_mm / 2.0) / voxel_dims[1]
    c_vox = (lesion_diameter_mm / 4.0) / voxel_dims[2]

    # Bounding box
    min_x, max_x = int(max(0, center_x - a_vox)), int(min(shape[0], center_x + a_vox))
    min_y, max_y = int(max(0, center_y - b_vox)), int(min(shape[1], center_y + b_vox))
    min_z, max_z = int(max(0, center_z - c_vox)), int(min(shape[2], center_z + c_vox))

    for z in range(min_z, max_z + 1):
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                d_sq = ((x - center_x)**2 / a_vox**2) + \
                       ((y - center_y)**2 / b_vox**2) + \
                       ((z - center_z)**2 / c_vox**2)
                if d_sq <= 1.0:
                    mask[x, y, z] = 1
    return mask

def process_masks(full_t1_path):
    filename = os.path.basename(full_t1_path)
    subj_id = filename.split('_')[0]
    
    # We need the original brain mask to recalculate the dynamic centroid
    brain_mask_path = os.path.join(MASK_BASE_DIR, f"{subj_id}_T1w.nii.gz")

    if not os.path.exists(brain_mask_path):
        print(f"Skipping {subj_id}: Original brain mask not found.")
        return

    # Load metadata
    t1_img = nib.load(full_t1_path)
    brain_mask_data = nib.load(brain_mask_path).get_fdata()
    voxel_dims = t1_img.header.get_zooms()[:3]
    
    # --- DYNAMIC CENTROID LOGIC (Replicating your placement) ---
    brain_coords = np.argwhere(brain_mask_data > 0)
    min_x, min_y, _ = brain_coords.min(axis=0)
    max_x, max_y, _ = brain_coords.max(axis=0)
    mid_x = (min_x + max_x) // 2
    dynamic_mid_y = (min_y + max_y) // 2
    v_off = lambda mm: int(mm / voxel_dims[0])

    placements = {
        "Medium_Size/Close_Midline": (MEDIUM, (mid_x + v_off(15), dynamic_mid_y, FIXED_SLICE_Z)),
        "Medium_Size/Middle_Midline": (MEDIUM, (mid_x + v_off(35), dynamic_mid_y, FIXED_SLICE_Z)),
        "Medium_Size/Far_Midline": (MEDIUM, (mid_x + v_off(55), dynamic_mid_y, FIXED_SLICE_Z)),
        "Small_Size/Close_Midline": (SMALL, (mid_x + v_off(15), dynamic_mid_y, FIXED_SLICE_Z)),
        "Large_Size/Close_Midline": (LARGE, (mid_x + v_off(15), dynamic_mid_y, FIXED_SLICE_Z))
    }

    for case_path, (diam, center) in placements.items():
        size_dir, loc_dir = case_path.split('/')
        final_mask_dir = os.path.join(MASK_OUTPUT_BASE, size_dir, loc_dir)
        os.makedirs(final_mask_dir, exist_ok=True)
        
        # Generate the mask
        lesion_mask = generate_binary_lesion_mask(t1_img.shape, center, diam, voxel_dims)
        
        # Save mask
        mask_nifti = nib.Nifti1Image(lesion_mask, t1_img.affine, t1_img.header)
        # Setting to ubyte for space efficiency
        mask_nifti.header.set_data_dtype(np.uint8) 
        
        out_name = os.path.join(final_mask_dir, f"{subj_id}_lesion_mask.nii.gz")
        nib.save(mask_nifti, out_name)
        print(f"Saved: {out_name}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        process_masks(sys.argv[1])
    else:
        print("Usage: python generate_masks.py <path_to_one_original_t1>")