import ants
import numpy as np
import SimpleITK as sitk
import os
import random
import sys

random.seed(6)

def load_image(fpath):
    itk_img = sitk.ReadImage(fpath)
    img_array = sitk.GetArrayFromImage(itk_img).astype(float)
    img_array = np.swapaxes(img_array, 0, 2)
    img = ants.from_numpy(img_array)
    img.set_origin(itk_img.GetOrigin())
    return itk_img, img

def get_roi_bspline_params(df, region_val, label_atlas):
    mask = get_mask(label_atlas, region_val)
    np_df = sitk.GetArrayFromImage(df)
    np_mask = sitk.GetArrayFromImage(mask)
    mask_nonzero = np.nonzero(np_mask)
    
    n_coords = len(mask_nonzero[0])
    if n_coords == 0:
        return np.array([]), np.array([])
        
    parametric_data_roi = np.zeros((n_coords, 3))
    scattered_data_roi = np.zeros((n_coords, 3))

    parametric_data_roi[:,0] = mask_nonzero[2] 
    parametric_data_roi[:,1] = mask_nonzero[1] 
    parametric_data_roi[:,2] = mask_nonzero[0] 

    for i in range(n_coords):
        z, y, x = mask_nonzero[0][i], mask_nonzero[1][i], mask_nonzero[2][i]
        scattered_data_roi[i,:] = np_df[z, y, x, :]

    return parametric_data_roi, scattered_data_roi

def get_bspline_disp_field(ref_img, parametric_data, scattered_data, meshSize=32, NFittingLevels=3):
    bspline_ants = ants.fit_bspline_object_to_scattered_data(
        scattered_data, parametric_data,
        parametric_domain_origin=[0.0, 0.0, 0.0],
        parametric_domain_spacing=[1.0, 1.0, 1.0],
        parametric_domain_size = ref_img.shape,
        number_of_fitting_levels=NFittingLevels, mesh_size=meshSize)

    bspline_arr = bspline_ants.numpy().swapaxes(0, 2)
    bspline_itk = sitk.GetImageFromArray(bspline_arr, isVector=True)
    
    flag, jd = diffeomorphic_check(bspline_itk)
    return bspline_itk, flag, jd

def diffeomorphic_check(disp_field):
    flag = 0
    jd = sitk.DisplacementFieldJacobianDeterminant(disp_field)
    jd_arr = sitk.GetArrayViewFromImage(jd)
    if jd_arr.min() < 0:
        flag = 1
    return flag, jd

def get_mask(label_atlas, region_val):
    return sitk.BinaryThreshold(label_atlas, region_val, region_val, 1, 0)

def svf_scaling_and_squaring(velo_field_itk, accuracy=16, compute_inverse=False):

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

def resample_to_reference(moving, reference):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(sitk.sitkBSpline)
    return resampler.Execute(moving)

def match_image_geometry(moving_img, reference_img):
    moving_img.SetOrigin(reference_img.GetOrigin())
    moving_img.SetSpacing(reference_img.GetSpacing())
    moving_img.SetDirection(reference_img.GetDirection())
    return moving_img

base_data_dir = "/home/ravi.bullock/"
output_transformations_dir = os.path.join(base_data_dir, "Output_Transformations_Pediatric")
synthetic_data_t1_dir = "/home/ravi.bullock/SPIE_Extension_Pediatric_12"
synthetic_data_segmentations_dir = "/home/ravi.bullock/SPIE_Extension_Pediatric_12_Segmentations"
save_base_dir = "/work/wilms_lab/ravi/MergedOutput_AgeModified_Pediatric"
baseline_age = 12
zero_origin = (0.0, 0.0, 0.0)

regions_of_interest = {
    "VentralDiencephalon": (26, 77), 
    "ThirdVentricle": (29, 80) 
}

def process_single_subject(subject_id_str):
    print(f"\n--- Starting Subject: {subject_id_str} ---")
    subject_save_dir = os.path.join(save_base_dir, f"Sub{subject_id_str}")
    os.makedirs(subject_save_dir, exist_ok=True)

    t1_path = os.path.join(synthetic_data_t1_dir, f"sub-{subject_id_str}{baseline_age}_T1w.nii.gz")
    if not os.path.exists(t1_path): return
    itk_img_12, ants_img_12 = load_image(t1_path)
    itk_img_12.SetOrigin(zero_origin)

    seg_path = os.path.join(synthetic_data_segmentations_dir, f"sub-{subject_id_str}{baseline_age}_T1w.nii.gz")
    if not os.path.exists(seg_path): return
    label_sitk = sitk.ReadImage(seg_path)
    label_sitk.SetOrigin(zero_origin)
    label_sitk = match_image_geometry(resample_to_reference(label_sitk, itk_img_12), itk_img_12)

    def load_vf(target_age):
        path = os.path.join(output_transformations_dir, f"sub-{subject_id_str}_{baseline_age}_to_{target_age}", f"sub-{subject_id_str}{baseline_age}_T1w_velo_{baseline_age}_to_{target_age}.nii.gz")
        if not os.path.exists(path): return None
        vf = sitk.ReadImage(path)
        vf = match_image_geometry(resample_to_reference(vf, itk_img_12), itk_img_12)
        vf.SetOrigin(zero_origin)
        return vf

    vf_8 = load_vf(8); vf_16 = load_vf(16)

    warper = sitk.WarpImageFilter()
    warper.SetInterpolator(sitk.sitkBSpline) 
    warper.SetOutputParameteresFromImage(itk_img_12)

    exps = [("Younger", vf_8), ("Older", vf_16)]
    for prefix, target_vf in exps:
        if target_vf is None: continue
        for region_name, label_pair in regions_of_interest.items():
            print(f"  Processing {prefix} {region_name}...")
            out_dir = os.path.join(subject_save_dir, f"{prefix}_{region_name}")
            os.makedirs(out_dir, exist_ok=True)

            roi_p, roi_s = [], []
            mask_union_sitk = sitk.Image(itk_img_12.GetSize(), sitk.sitkUInt8)
            mask_union_sitk.CopyInformation(itk_img_12)

            for val in label_pair:
                p, s = get_roi_bspline_params(target_vf, val, label_sitk)
                if p.size > 0:
                    roi_p.append(p); roi_s.append(s)
                    mask_union_sitk = sitk.Or(mask_union_sitk, get_mask(label_sitk, val))
            
            if not roi_p: continue
            
            context_mask_np = sitk.GetArrayFromImage(sitk.Not(mask_union_sitk))
            bg_coords = np.array(np.nonzero(context_mask_np)).T
            
            if len(bg_coords) > 50000:
                bg_coords = bg_coords[np.random.choice(len(bg_coords), 50000, replace=False)]
            
            ctx_p = np.zeros_like(bg_coords, dtype=np.float32)
            ctx_s = np.zeros((len(bg_coords), 3), dtype=np.float32)
            ctx_p[:,0], ctx_p[:,1], ctx_p[:,2] = bg_coords[:,2], bg_coords[:,1], bg_coords[:,0]

            all_p, all_s = np.vstack(roi_p + [ctx_p]), np.vstack(roi_s + [ctx_s])

            current_mesh = 32
            is_valid = False
            while not is_valid and current_mesh >= 8:
                vf_final, flag, _ = get_bspline_disp_field(ants_img_12, all_p, all_s, meshSize=current_mesh)
                if flag == 0:
                    is_valid = True
                else:
                    current_mesh -= 8
            
            vf_final = match_image_geometry(vf_final, itk_img_12)
            df_final = svf_scaling_and_squaring(vf_final, accuracy=16, compute_inverse=False)
            sitk.WriteImage(warper.Execute(itk_img_12, df_final), os.path.join(out_dir, f"warped_{prefix}_{region_name}.nii.gz"))

    mixed_cases = [(vf_16, vf_8, "OlderVentralDiencephalon_YoungerThirdVentricle"), 
                   (vf_8, vf_16, "YoungerVentralDiencephalon_OlderThirdVentricle")]
    
    for p_vf, v_vf, mixed_name in mixed_cases:
        if p_vf is None or v_vf is None: continue
        print(f"  Processing Mixed Case: {mixed_name}...")
        out_dir = os.path.join(subject_save_dir, mixed_name)
        os.makedirs(out_dir, exist_ok=True)

        m_roi_p, m_roi_s = [], []
        mask_m_union = sitk.Image(itk_img_12.GetSize(), sitk.sitkUInt8)
        mask_m_union.CopyInformation(itk_img_12)

        for val in regions_of_interest["VentralDiencephalon"]:
            p, s = get_roi_bspline_params(p_vf, val, label_sitk)
            if p.size > 0: 
                m_roi_p.append(p); m_roi_s.append(s)
                mask_m_union = sitk.Or(mask_m_union, get_mask(label_sitk, val))
        for val in regions_of_interest["ThirdVentricle"]:
            p, s = get_roi_bspline_params(v_vf, val, label_sitk)
            if p.size > 0: 
                m_roi_p.append(p); m_roi_s.append(s)
                mask_m_union = sitk.Or(mask_m_union, get_mask(label_sitk, val))

        ctx_m_np = sitk.GetArrayFromImage(sitk.Not(mask_m_union))
        bg_m_coords = np.array(np.nonzero(ctx_m_np)).T
        if len(bg_m_coords) > 50000:
            bg_m_coords = bg_m_coords[np.random.choice(len(bg_m_coords), 50000, replace=False)]

        ctx_m_p = np.zeros_like(bg_m_coords, dtype=np.float32)
        ctx_m_s = np.zeros((len(bg_m_coords), 3), dtype=np.float32)
        ctx_m_p[:,0], ctx_m_p[:,1], ctx_m_p[:,2] = bg_m_coords[:,2], bg_m_coords[:,1], bg_m_coords[:,0]

        all_m_p, all_m_s = np.vstack(m_roi_p + [ctx_m_p]), np.vstack(m_roi_s + [ctx_m_s])

        cur_m_mesh = 32
        is_m_valid = False
        while not is_m_valid and cur_m_mesh >= 8:
            vf_m, flag, _ = get_bspline_disp_field(ants_img_12, all_m_p, all_m_s, meshSize=cur_m_mesh)
            if flag == 0: is_m_valid = True
            else: cur_m_mesh -= 8

        vf_m = match_image_geometry(vf_m, itk_img_12)
        df_m = svf_scaling_and_squaring(vf_m, accuracy=16, compute_inverse=False)
        sitk.WriteImage(warper.Execute(itk_img_12, df_m), os.path.join(out_dir, f"{mixed_name}_final.nii.gz"))

if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit(1)
    process_single_subject(sys.argv[1])