import ants
import numpy as np
import SimpleITK as sitk
import os
import sys

# --- Paths & Config ---
base_data_dir = "/home/ravi.bullock/"
output_transformations_dir = os.path.join(base_data_dir, "Output_Transformations_Pediatric")
synthetic_data_t1_dir = "/home/ravi.bullock/SPIE_Extension_Pediatric_12"
save_base_dir = "/work/wilms_lab/ravi/FullField_AgeModified_Pediatric"

baseline_age = 12
zero_origin = (0.0, 0.0, 0.0)

def load_image(fpath):
    itk_img = sitk.ReadImage(fpath)
    itk_img.SetOrigin(zero_origin)
    return itk_img

def match_image_geometry(moving_img, reference_img):
    moving_img.SetOrigin(reference_img.GetOrigin())
    moving_img.SetSpacing(reference_img.GetSpacing())
    moving_img.SetDirection(reference_img.GetDirection())
    return moving_img

def resample_to_reference(moving, reference):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(sitk.sitkBSpline) 
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(moving)

def svf_scaling_and_squaring(velo_field_itk, accuracy=16, compute_inverse=True):
    velo_field_np = np.float32(sitk.GetArrayFromImage(-velo_field_itk if compute_inverse else velo_field_itk)/(2**accuracy))
    velo = sitk.GetImageFromArray(velo_field_np, isVector=True)
    velo.CopyInformation(velo_field_itk)
    
    warper = sitk.WarpImageFilter()
    warper.SetInterpolator(sitk.sitkBSpline)
    warper.SetOutputParameteresFromImage(velo_field_itk)
    
    for i in range(accuracy):
        temp = warper.Execute(velo, velo)
        velo = velo + temp
    return velo

def process_single_subject(subject_id_str):
    print(f"\n--- Full-Field Processing: {subject_id_str} ---")
    subject_save_dir = os.path.join(save_base_dir, f"sub-{subject_id_str}")
    os.makedirs(subject_save_dir, exist_ok=True)

    t1_path = os.path.join(synthetic_data_t1_dir, f"sub-{subject_id_str}{baseline_age}_T1w.nii.gz")
    if not os.path.exists(t1_path):
        print(f"Error: Baseline image not found for {subject_id_str}")
        return
    itk_img_12 = load_image(t1_path)

    def process_age_shift(target_age, prefix):
        vf_folder = f"sub-{subject_id_str}_{baseline_age}_to_{target_age}"
        vf_filename = f"sub-{subject_id_str}{baseline_age}_T1w_velo_{baseline_age}_to_{target_age}.nii.gz"
        vf_path = os.path.join(output_transformations_dir, vf_folder, vf_filename)
        
        if not os.path.exists(vf_path):
            print(f"    Warning: Velocity field not found for {prefix} (Age {target_age})")
            return

        velo_field = sitk.ReadImage(vf_path)
        

        vx = sitk.VectorIndexSelectionCast(velo_field, 0)
        vy = sitk.VectorIndexSelectionCast(velo_field, 1)
        vz = sitk.VectorIndexSelectionCast(velo_field, 2)


        velo_field = sitk.Compose(vx, vy, vz)

        velo_field = match_image_geometry(resample_to_reference(velo_field, itk_img_12), itk_img_12)
        
        disp_field = svf_scaling_and_squaring(velo_field, accuracy=16, compute_inverse=False)

        warper = sitk.WarpImageFilter()
        warper.SetInterpolator(sitk.sitkBSpline)
        warper.SetOutputParameteresFromImage(itk_img_12)
        warped_img = warper.Execute(itk_img_12, disp_field)

        out_folder = os.path.join(subject_save_dir, f"{prefix}_FullField")
        os.makedirs(out_folder, exist_ok=True)
        sitk.WriteImage(warped_img, os.path.join(out_folder, f"warped_{prefix}_Age{target_age}.nii.gz"))
        print(f"    Successfully generated {prefix} Full-Field image (Axis-wise Scaling Applied).")

    process_age_shift(8, "Younger")
    process_age_shift(16, "Older")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <subject_id>")
        sys.exit(1)
    process_single_subject(sys.argv[1])