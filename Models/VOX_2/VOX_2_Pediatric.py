import numpy as np
import utils
import sklearn
import bids
import sklearn.model_selection
import monai.transforms
import os
import torch
from monai.networks.nets import UNet
from torch import nn
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import re
import pickle

print('CUDA:', torch.cuda.is_available())
print('DEVICE:', torch.cuda.get_device_name(0))

LAMBDA_TV = 0            
LAMBDA_REGRESSION = 1.0     

#Static Definitions
space="rigid"
pix_spacing=1.
writer = monai.data.ITKWriter() 
lr=1e-3 
num_workers=os.cpu_count()//2 
pin_memory=torch.cuda.is_available() if num_workers > 0 else False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
amp_mode="amp"
grad_scaler=True if amp_mode=="amp" else False
num_of_files=-1

#Operations
train_model = False
test_model = True

#Batch Sizes
batch_size_train = 2
batch_size_test = batch_size_train

#Path Extractor
def extract_substring(input_str):
    pattern = r"sub-[^\.]+" 
    
    match = re.search(pattern, input_str)
    if match:
        return match.group(0)  
    return None 


#Save Locations
validation_save_dir = "/work/wilms_lab/ravi/BrainAgeVoxelLevelPredictions_SPIEExtension/Pediatric_Real/Validation/"
test_save_dir = "/work/wilms_lab/ravi/BrainAgeVoxelLevelPredictions_SPIEExtension/Pediatric_Real/Test_Real/"
bias_save_dir = "/work/wilms_lab/ravi/BrainAgeVoxelLevelPredictions_SPIEExtension/Pediatric_Real/Bias/"
model_epoch_save_dir = "/work/wilms_lab/ravi/BrainAgeVoxelLevelPredictions_SPIEExtension/Pediatric_Real/Models/"

model_best_save_path = "/work/wilms_lab/ravi/UNET_Models_Real/Pediatric/translation_model.pth"
model_best_save_dir = "/work/wilms_lab/ravi/BrainAgeVoxelLevelPredictions_SPIEExtension/Pediatric_Real/BestModel/"


best_model_metrics_path = "/work/wilms_lab/ravi/BrainAgeVoxelLevelPredictions_SPIEExtension/Pediatric_Real/BestModel/BestModel.txt"
model_load_path = "/work/wilms_lab/ravi/BrainAgeVoxelLevelPredictions_SPIEExtension/Pediatric_Real/BestModel/translation_model.pth"
test_mae_save_path = "/work/wilms_lab/ravi/BrainAgeVoxelLevelPredictions_SPIEExtension/Pediatric_Real/Test_Real/MAE.txt"


#checkpoint_path = "/work/wilms_lab/ravi/BrainAgeVoxelLevelPredictions_SPIEExtension/Pediatric_Long/Models/translation_model32.pth"  #Resume from last to save time
checkpoint_path = ""

#Epochs
train_epochs = 101

#Load Data

# #Synthetic Test
# with open("synthetic_splits/synthetic_pediatric_splits_longitudinal.pkl", "rb") as f:
#     splits = pickle.load(f)

# image_train, age_train = splits["train"]
# image_val, age_val = splits["val"]
# image_test, age_test = splits["test"]

with open("/home/ravi.bullock/pediatric_splits.pkl", "rb") as f:
    splits = pickle.load(f)

image_train, age_train = splits["train"]
image_val, age_val = splits["val"]
image_test, age_test = splits["test"]

def fix_path(img_path):
    if img_path.startswith("pediatric/data"):
        return img_path.replace("pediatric/data", "/work/wilms_lab/ravi/pediatric/data")
    return img_path

image_train = [fix_path(p) for p in image_train]
image_test = [fix_path(p) for p in image_test]
image_val = [fix_path(p) for p in image_val]

age_train=np.float32((np.array(age_train)-0.)/1.0)
age_test=np.float32((np.array(age_test)-0.)/1.0)
age_val=np.float32((np.array(age_val)-0.)/1.0)

#Dictionaries
train_files = [{"image": img, "age": label, "path": img} for img, label in zip(image_train, age_train)]
test_files = [{"image": img, "age": label, "path": img} for img, label in zip(image_test, age_test)]
val_files = [{"image": img, "age": label, "path": img} for img, label in zip(image_val, age_val)]


print("Train Images:", len(train_files),"("+str(np.mean(age_train))+"/"+str(np.std(age_train))+")")
print("Test Images:", len(test_files),"("+str(np.mean(age_test))+"/"+str(np.std(age_test))+")")
print("Validation Images:", len(val_files),"("+str(np.mean(age_val))+"/"+str(np.std(age_val))+")")

#Define Transforms
train_transforms = monai.transforms.Compose(
    [monai.transforms.LoadImaged(keys=["image"], ensure_channel_first=True,reader="ITKReader"),
     monai.transforms.Spacingd(keys=["image"],pixdim=(pix_spacing,pix_spacing,pix_spacing)),
     monai.transforms.DivisiblePadd(keys=["image"],k=64),
     monai.transforms.CenterSpatialCropd(keys=["image"],roi_size=[192,224,192]),
     monai.transforms.ScaleIntensityRangePercentilesd(keys="image", lower=1, upper=99, b_min=-1, b_max=1,clip=True),
     monai.transforms.ToTensord(keys=["image", "age"],track_meta=False), 
    ])

# train_transforms = monai.transforms.Compose([
#     monai.transforms.LoadImaged(keys=["image"], ensure_channel_first=True, reader="ITKReader"),
#     monai.transforms.Spacingd(keys=["image"], pixdim=(pix_spacing, pix_spacing, pix_spacing)),
#     monai.transforms.DivisiblePadd(keys=["image"], k=64),
#     monai.transforms.CenterSpatialCropd(keys=["image"], roi_size=[192, 224, 192]),
#     monai.transforms.ScaleIntensityRangePercentilesd(keys="image", lower=1, upper=99, b_min=-1, b_max=1, clip=True),
#     monai.transforms.RandAdjustContrastd(keys=["image"], prob=0.5, gamma=(0.5, 2.0)),
#     monai.transforms.RandGaussianNoised(keys=["image"], prob=0.5, mean=0.0, std=0.15),
#     monai.transforms.RandBiasFieldd(keys=["image"], prob=0.5),
#     monai.transforms.ToTensord(keys=["image", "age"], track_meta=False), 
# ])


val_test_transforms = monai.transforms.Compose(
    [monai.transforms.LoadImaged(keys=["image"], ensure_channel_first=True,reader="ITKReader"),
     monai.transforms.Spacingd(keys=["image"],pixdim=(pix_spacing,pix_spacing,pix_spacing)),
     monai.transforms.DivisiblePadd(keys=["image"],k=64),
     monai.transforms.CenterSpatialCropd(keys=["image"],roi_size=[192,224,192]),
     monai.transforms.ScaleIntensityRangePercentilesd(keys="image", lower=1, upper=99, b_min=-1, b_max=1,clip=True),
     monai.transforms.ToTensord(keys=["image", "age"],track_meta=False), 
    ])

#Loader
train_ds = monai.data.Dataset(data=train_files,transform=train_transforms)
train_loader = monai.data.DataLoader(train_ds, shuffle=True, batch_size=batch_size_train, num_workers=num_workers, pin_memory=True)

val_ds = monai.data.Dataset(data=val_files,transform=val_test_transforms)
val_loader = monai.data.DataLoader(val_ds, shuffle=False, batch_size=batch_size_test, num_workers=num_workers, pin_memory=False)

test_ds = monai.data.Dataset(data=test_files,transform=val_test_transforms)
test_loader = monai.data.DataLoader(test_ds, shuffle=False, batch_size=batch_size_test, num_workers=num_workers, pin_memory=False)


# check_ds = monai.data.Dataset(data=train_files[0:1],transform=train_transforms)
# check_loader = monai.data.DataLoader(check_ds, shuffle=False,batch_size=1, num_workers=0, pin_memory=False)
# im_dict = monai.utils.misc.first(check_loader)
# #print(im_dict['image'].shape, im_dict['age'])
# writer = monai.data.ITKWriter()  # subclass of ImageWriter
# writer.set_data_array(torch.squeeze(im_dict['image'][0].data),channel_dim=None)
# writer.write("./test_unet_pediatric.nii.gz")

import nibabel as nib
import nibabel as nib
import numpy as np

# 1. Get the exact batch
check_ds = monai.data.Dataset(data=train_files[0:1], transform=train_transforms)
check_loader = monai.data.DataLoader(check_ds, shuffle=False, batch_size=1)
im_dict = monai.utils.misc.first(check_loader)

# 2. Extract the raw numpy array (Model View)
model_input_array = im_dict['image'][0][0].detach().cpu().numpy()

# 3. Create a NIfTI object with an Identity Matrix (since meta is tracked as False)
# This mimics how the writer behaves when metadata is missing
input_as_nifti = nib.Nifti1Image(model_input_array, np.eye(4))
nib.save(input_as_nifti, "./test_unet_pediatric.nii.gz")

# 4. Print the Dimensions and Origin
print(f"--- UNET INPUT DATA ---")
print(f"Dimensions: {input_as_nifti.shape}")
print(f"Voxel Spacing: {input_as_nifti.header.get_zooms()}")
print(f"Origin (Affine Translation): {input_as_nifti.affine[:3, 3]}")
print(f"Intensity Range: {model_input_array.min():.4f} to {model_input_array.max():.4f}")


# --- ADDED: Custom Loss Functions ---
def total_variation_loss(pred):
    """3D Total Variation Loss (L1 norm of gradients in x, y, z)"""
    tv_loss = 0.0
    # L1 norm of the gradient in x-direction
    tv_loss += torch.sum(torch.abs(pred[:, :, 1:, :-1, :-1] - pred[:, :, :-1, :-1, :-1]))
    # L1 norm of the gradient in y-direction
    tv_loss += torch.sum(torch.abs(pred[:, :, :-1, 1:, :-1] - pred[:, :, :-1, :-1, :-1]))
    # L1 norm of the gradient in z-direction
    tv_loss += torch.sum(torch.abs(pred[:, :, :-1, :-1, 1:] - pred[:, :, :-1, :-1, :-1]))
    return tv_loss

# # Modified TV loss to focus only on the brain tissue
# def total_variation_loss(pred, mask):
#     # Apply mask so we don't penalize the sharp edge at the brain boundary
#     masked_pred = pred * mask
    
#     # L1 of gradients
#     tv_x = torch.abs(masked_pred[:, :, 1:, :, :] - masked_pred[:, :, :-1, :, :])
#     tv_y = torch.abs(masked_pred[:, :, :, 1:, :] - masked_pred[:, :, :, :-1, :])
#     tv_z = torch.abs(masked_pred[:, :, :, :, 1:] - masked_pred[:, :, :, :, :-1])
    
#     # Return mean to keep values stable regardless of volume size
#     return (tv_x.mean() + tv_y.mean() + tv_z.mean())


#Model
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=[8, 16, 32, 64], 
    strides=(1,1,1), 
    kernel_size=3,
    act = ("leakyrelu", {"negative_slope":0.2}),
    norm="instance",
    dropout=0.0,
    bias=True
).to(device)

mae_loss_function = nn.L1Loss()
optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
scaler = torch.amp.GradScaler('cuda', enabled=grad_scaler)

if train_model:
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming training from epoch {start_epoch}")
    else:
        print("No checkpoint found, starting from scratch.")
        start_epoch = 0  

    print("TRAIN")
    if not os.path.exists(validation_save_dir):
        os.makedirs(validation_save_dir)
    if not os.path.exists(model_epoch_save_dir):
        os.makedirs(model_epoch_save_dir)
    if not os.path.exists(model_best_save_dir):
        os.makedirs(model_best_save_dir)
    
    best_metric = float("inf")

    for epoch in range(start_epoch, train_epochs):
        model.train()
        epoch_loss = 0.0
        step = 0

        for batch_data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_epochs} (Training)", unit="batch"):
            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data["age"].to(device).float()
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=grad_scaler):
                outputs = model(inputs)
                mask = inputs > -0.9
                target = labels.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(inputs)
                noise = (torch.empty_like(target).uniform_(-0.5, 0.5))
                noised_target = torch.add(target, noise)

                # # --- MODIFIED: Combined Loss ---
                loss_regression = mae_loss_function((outputs*mask).squeeze(0).squeeze(0), (noised_target*mask).squeeze(0).squeeze(0))
                loss_tv = total_variation_loss(outputs)
                loss = (LAMBDA_REGRESSION * loss_regression) + (LAMBDA_TV * loss_tv)

                # loss_regression = mae_loss_function(outputs * mask, noised_target * mask)
                # loss_tv = total_variation_loss(outputs, mask)
                # loss = (LAMBDA_REGRESSION * loss_regression) + (LAMBDA_TV * loss_tv)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        epoch_loss /= step
        print(f"Epoch {epoch + 1}/{train_epochs}, Loss: {epoch_loss:.4f}")

        #Save Model Each Epoch
        save_path = f"{model_epoch_save_dir}translation_model{epoch+1}.pth"
        torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, save_path)
        print(f"Model Saved to {save_path}")
        loss_filename = f"{model_epoch_save_dir}training_loss.txt"
        with open(loss_filename, "a") as f:
            f.write(f"Epoch {epoch+1} Loss: {loss:.4f}\n")
            print(f"Loss appended to {loss_filename}")

        #Validation
        if epoch % 10 == 0:
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for val_data in tqdm(val_loader, desc=f"Epoch {epoch+1}/{train_epochs} (Validation)", unit="batch"):
                    val_inputs, val_labels = val_data["image"].to(device), val_data["age"].to(device).float()
                    with torch.amp.autocast('cuda',enabled=grad_scaler):
                        val_outputs = model(val_inputs)
                        val_mask = val_inputs > -0.9

                        val_target = val_labels.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(val_inputs)
                        val_noise = (torch.empty_like(val_target).uniform_(-0.5, 0.5))
                        val_noised_target= torch.add(val_target, val_noise)

                        # --- MODIFIED: Combined Loss ---
                        val_loss_reg = mae_loss_function((val_outputs*val_mask).squeeze(0).squeeze(0), (val_noised_target*val_mask).squeeze(0).squeeze(0))
                        val_loss_tv = total_variation_loss(val_outputs)
                        loss = (LAMBDA_REGRESSION * val_loss_reg) + (LAMBDA_TV * val_loss_tv)

                    val_loss += loss.item()
            val_loss /= len(val_loader)
            print(f"Validation Loss: {val_loss:.4f}")
            val_loss_filename = f"{validation_save_dir}val_loss.txt"
            with open(val_loss_filename, "a") as f:
                f.write(f"Epoch {epoch+1} Loss: {val_loss:.4f}\n")
                print(f"Val Loss appended to {val_loss_filename}")
            scheduler.step(val_loss)


            if val_loss < best_metric:
                best_metric = val_loss
                best_metric_epoch = epoch + 1
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, model_best_save_path)
                print(f"Best Model Saved to {model_best_save_path}")
                with open(os.path.join(best_model_metrics_path), "w") as file:
                    file.write(f"Epoch: {epoch+1}, Train_Loss: {epoch_loss}, Val_Loss: {val_loss}")

            with torch.no_grad():
                sample_iter = next(iter(val_loader))
                sample_input, sample_label, sample_paths = sample_iter["image"].to(device), sample_iter["age"].to(device).float(), sample_iter["path"]
                with torch.amp.autocast('cuda',enabled=grad_scaler):
                    sample_output = model(sample_input)
                    epoch_dir = os.path.join(validation_save_dir, f"{epoch}")
                    if not os.path.exists(epoch_dir):
                        os.makedirs(epoch_dir, exist_ok = True)

                    # Iterate through each image in the batch
                    for i in range(sample_input.shape[0]):
                        # Extract a unique identifier for the filename
                        unique_id = extract_substring(sample_paths[i]).replace('_register', '')
                        if not unique_id:
                            unique_id = f"sample_{i}" # Fallback if path extraction fails

                        # Select the i-th image from the batch and remove the channel dimension
                        writer.set_data_array(sample_input[i], channel_dims=0)
                        writer.write(os.path.join(epoch_dir, f"{unique_id}_val_image.nii.gz"))

                        sample_mask = sample_input > -0.9
                        writer.set_data_array(sample_mask[i], channel_dims=0)
                        writer.write(os.path.join(epoch_dir, f"{unique_id}_val_mask.nii.gz"))

                        sample_target = sample_label.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(sample_input)
                        sample_noise = (torch.empty_like(sample_target).uniform_(-0.5, 0.5))
                        sample_noised_target = torch.add(sample_target, sample_noise)

                        writer.set_data_array(((sample_noised_target * sample_mask))[i], channel_dims=0)
                        writer.write(os.path.join(epoch_dir, f"{unique_id}_val_target.nii.gz"))

                        masked_output = (sample_output)*sample_mask
                        writer.set_data_array(masked_output[i], channel_dims=0)
                        writer.write(os.path.join(epoch_dir, f"{unique_id}_val_output.nii.gz"))

                        global_brain_age = np.mean((sample_output[i]).squeeze(0).cpu().numpy()[sample_mask[i].squeeze(0).cpu().numpy()])
                        with open(os.path.join(epoch_dir, f"{unique_id}_GlobalBrainAge.txt"), "w") as file:
                                file.write(f"Predicted Global Brain Age: {global_brain_age}, Actual Global Brain Age: {sample_iter['age'][i]}")

if test_model:
    print("TEST")
    checkpoint = torch.load(model_best_save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    if not os.path.exists(test_save_dir):
        os.makedirs(test_save_dir)

    test_ages = []
    predicted_ages = []
    voxel_wise_mae_all_samples = []

    for test_data in tqdm(test_loader, total=len(test_loader), desc="Processing Test Samples"):
        test_inputs, test_labels, test_paths = test_data["image"].to(device), test_data["age"], test_data["path"]

        with torch.no_grad():
            test_outputs = model(test_inputs)

        # Iterate through each image in the batch for saving
        for i in range(test_inputs.shape[0]):
            test_input_i = test_inputs[i]
            test_output_i = test_outputs[i]
            test_label_i = test_labels[i]
            test_path_i = test_paths[i]

            unique_id = extract_substring(test_path_i).replace('_register', '')
            if not unique_id:
                unique_id = f"test_sample_{i}" # Fallback if path extraction fails

            test_dir = os.path.join(test_save_dir, f"{unique_id}")
            if not os.path.exists(test_dir):
                os.makedirs(test_dir)

            writer.set_data_array(test_input_i.cpu().detach().numpy(), channel_dims=0)
            writer.write(os.path.join(test_dir, f"{unique_id}_test_image.nii.gz"))

            mask_i = test_input_i > -0.9
            writer.set_data_array((test_output_i.cpu().detach().numpy()) * mask_i.squeeze(0).cpu().detach().numpy(), channel_dims=0)
            writer.write(os.path.join(test_dir, f"{unique_id}_test_output.nii.gz"))

            chronological_arr = np.zeros_like(test_output_i.cpu().numpy()) + (test_label_i.item())

            brain_age_gap = (test_output_i.squeeze(0).cpu().detach().numpy()) - chronological_arr

            writer.set_data_array(brain_age_gap * mask_i.cpu().detach().numpy(), channel_dims=0)
            writer.write(os.path.join(test_dir, f"{unique_id}_test_brain_age_gap.nii.gz"))

            writer.set_data_array(mask_i, channel_dims=0)
            writer.write(os.path.join(test_dir, f"{unique_id}_test_mask.nii.gz"))

            masked_values = (test_output_i.squeeze(0))[mask_i.squeeze(0).detach().cpu().numpy()]
            masked_values = masked_values.detach().cpu().numpy()
            global_brain_age = np.mean(masked_values)

            with open(os.path.join(test_dir, f"{unique_id}_GlobalBrainAge.txt"), "w") as file:
                file.write(f"Predicted Global Brain Age: {global_brain_age}, Actual Global Brain Age: {test_label_i}")

            test_ages.append(test_label_i.item())
            predicted_ages.append(global_brain_age)
            noised_target = test_label_i.item() + np.random.uniform(-0.5, 0.5, size=masked_values.shape)
            voxel_wise_mae = np.abs(masked_values - noised_target)
            voxel_wise_mae_all_samples.append(voxel_wise_mae)