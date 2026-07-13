import numpy as np
import nibabel as nib
import os
import sys

# --- Configuration ---
ROOT_OUTPUT_FOLDER = '/home/ravi.bullock/Test_File_Info/SPIEExtension_Lesioned/Adult'
SEG_BASE_DIR = '/work/wilms_lab/ravi/SPIE_Extension_Samples/AdultSynthetic/Segmentations'
MASK_BASE_DIR = '/work/wilms_lab/ravi/SPIE_Extension_Samples/AdultSynthetic/Masks'

# Constants
FIXED_SLICE_Z = 95
SMALL, MEDIUM, LARGE = 30, 50, 70

def add_lesion_to_volume(volume, lesion_center_coords, lesion_diameter_mm,
                         lesion_intensity, voxel_dims, lesion_shape='oval', blend_factor=0.7):
    modified_volume = np.copy(volume).astype(np.float32)
    center_x, center_y, center_z = lesion_center_coords

    a_mm = (lesion_diameter_mm / 2.0) / 2.0
    b_mm = lesion_diameter_mm / 2.0
    c_mm = (lesion_diameter_mm / 2.0) / 2.0
    
    a_vox, b_vox, c_vox = a_mm / voxel_dims[0], b_mm / voxel_dims[1], c_mm / voxel_dims[2]

    min_x, max_x = int(max(0, center_x - a_vox)), int(min(volume.shape[0], center_x + a_vox))
    min_y, max_y = int(max(0, center_y - b_vox)), int(min(volume.shape[1], center_y + b_vox))
    min_z, max_z = int(max(0, center_z - c_vox)), int(min(volume.shape[2], center_z + c_vox))

    core_radius_factor = np.clip(blend_factor, 0.0, 1.0)
    
    for z in range(min_z, max_z):
        for y in range(min_y, max_y):
            for x in range(min_x, max_x):
                d_sq = ((x - center_x)**2 / a_vox**2) + ((y - center_y)**2 / b_vox**2) + ((z - center_z)**2 / c_vox**2)
                if d_sq <= 1.0:
                    d_norm = np.sqrt(d_sq)
                    if d_norm <= core_radius_factor:
                        modified_volume[x, y, z] = lesion_intensity
                    else:
                        blend_w = (d_norm - core_radius_factor) / (1.0 - core_radius_factor)
                        modified_volume[x, y, z] = (1 - blend_w) * lesion_intensity + blend_w * volume[x, y, z]
    return modified_volume

def process_image(full_t1_path):
    filename = os.path.basename(full_t1_path)
    subj_id = filename.split('_')[0]
    
    seg_path = os.path.join(SEG_BASE_DIR, f"{subj_id}_T1w.nii.gz")
    mask_path = os.path.join(MASK_BASE_DIR, f"{subj_id}_T1w.nii.gz")

    if not all(os.path.exists(p) for p in [full_t1_path, seg_path, mask_path]):
        return

    t1_img = nib.load(full_t1_path)
    t1_data = t1_img.get_fdata()
    seg_data = nib.load(seg_path).get_fdata()
    mask_data = nib.load(mask_path).get_fdata()
    voxel_dims = t1_img.header.get_zooms()[:3]
    
    # Intensity: 10th percentile of the MASK
    lesion_intensity = np.percentile(t1_data[mask_data > 0], 10)
    
    # --- DYNAMIC CENTROID LOGIC ---
    brain_coords = np.argwhere(mask_data > 0)
    min_x, min_y, _ = brain_coords.min(axis=0)
    max_x, max_y, _ = brain_coords.max(axis=0)
    mid_x = (min_x + max_x) // 2
    dynamic_mid_y = (min_y + max_y) // 2

    v_off = lambda mm: int(mm / voxel_dims[0])

    # --- UPDATED PLACEMENT MATRIX ---
    # Experiment 1: Fixed Size (Medium), Variable Location (Close, Middle, Far)
    # Experiment 2: Variable Size (Small, Medium, Large), Fixed Location (Close)
    placements = {
        "Medium_Size/Close_Midline": (MEDIUM, (mid_x + v_off(15), dynamic_mid_y, FIXED_SLICE_Z)),
        "Medium_Size/Middle_Midline": (MEDIUM, (mid_x + v_off(35), dynamic_mid_y, FIXED_SLICE_Z)),
        "Medium_Size/Far_Midline": (MEDIUM, (mid_x + v_off(55), dynamic_mid_y, FIXED_SLICE_Z)),
        
        "Small_Size/Close_Midline": (SMALL, (mid_x + v_off(15), dynamic_mid_y, FIXED_SLICE_Z)),
        "Large_Size/Close_Midline": (LARGE, (mid_x + v_off(15), dynamic_mid_y, FIXED_SLICE_Z))
    }

    for case_path, (diam, center) in placements.items():
        size_dir, loc_dir = case_path.split('/')
        final_dir = os.path.join(ROOT_OUTPUT_FOLDER, size_dir, loc_dir)
        os.makedirs(final_dir, exist_ok=True)
        
        lesioned_data = add_lesion_to_volume(t1_data, center, diam, lesion_intensity, voxel_dims)
        out_name = os.path.join(final_dir, f"{subj_id}.nii.gz")
        nib.save(nib.Nifti1Image(lesioned_data, t1_img.affine, t1_img.header), out_name)

if __name__ == "__main__":
    process_image(sys.argv[1])