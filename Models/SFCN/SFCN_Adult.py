import numpy as np
import torch
import os

import monai
from monai.data import ImageDataset, decollate_batch, DataLoader
from monai.transforms import EnsureChannelFirst, Compose, RandRotate90, Resize, ScaleIntensity, DivisiblePad, RandGibbsNoise
from monai.handlers import StatsHandler, TensorBoardStatsHandler, stopping_fn_from_metric
from monai.utils import get_torch_version_tuple, set_determinism

import ignite
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.metrics import MeanAbsoluteError, RunningAverage, Loss
from ignite.contrib.handlers import ProgressBar

import SFCN
import utils
import sklearn.model_selection
#import xai_regression
#import captum
import bids
import random
import captum
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
torch.multiprocessing.set_sharing_strategy('file_system')
from ignite.handlers import Checkpoint, global_step_from_engine
#monai.config.print_config()
torch.manual_seed(8)
set_determinism(8)
random.seed(8)
np.random.seed(8)
torch.set_float32_matmul_precision('high')
import torch
import pickle


print('CUDA:', torch.cuda.is_available())
print('DEVICE:', torch.cuda.get_device_name(0))


# def generate_xai(net, test_item,output_dir='./'):
#     net.eval()
#     img=torch.unsqueeze(test_item['image'],0).to(device)
#     subject_path = str(test_item["path"])
#     subject_id = os.path.splitext(os.path.splitext(os.path.basename(subject_path))[0])[0]
#     subject_id = subject_id.replace('_T1w', '').replace('_register', '')
#     cap_sal=captum.attr.Saliency(net)
#     cap_nt = captum.attr.NoiseTunnel(cap_sal)
#     cap_map = cap_nt.attribute(img, nt_type='smoothgrad',nt_samples=25,nt_samples_batch_size=1,stdevs=0.15*(img.max()-img.min()).item())
#     new_dict={"image":test_item['image'] ,"grad":cap_map[0].detach().cpu()}
#     savers=monai.transforms.Compose([
#             monai.transforms.ShiftIntensityd(keys=["image"],offset=.5),
#             monai.transforms.SaveImaged(keys=["image"],separate_folder="False",output_dir=os.path.join(output_dir, subject_id),output_postfix="img",resample=True,mode='bilinear',padding_mode='zeros',writer="ITKWriter"),monai.transforms.ScaleIntensityd(keys=["grad"]),monai.transforms.SaveImaged(keys=["grad"],separate_folder="False",output_dir=os.path.join(output_dir, subject_id),output_postfix="grad",resample=True,mode='nearest',padding_mode='zeros',writer="ITKWriter")])
#     _=savers(new_dict)

#params
lr=5e-4
lr_scheduler=True
num_workers=os.cpu_count()//2
pin_memory=torch.cuda.is_available() if num_workers > 0 else False
batch_size_train=1#20#1#3
batch_size_test=batch_size_train#2#8#5
train_epochs = 200#20#200
gradient_accumulation_steps=2
pix_spacing=1.0#1.5#1.5
evaluate_every_x_epochs=10
amp_mode="amp"
grad_scaler=True if amp_mode=="amp" else False
wd=0#1e-2
eps=1e-6
pediatric_data=True
data_string="abcd"
stratify=True
num_of_files=-1
num_of_train_files=-1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#checkpoint_filename=None
checkpoint_filename="/work/wilms_lab/ravi/SFCN_Models_Real/Adult/SFCN_Adult_output.pt"

train_only=False
evaluate_only=True
use_bb=False
model_string="SFCN"
#model_string="MedNeXt"
#space="aff"
space="rigid"

#Load Data

# #Synthetic Test
with open("/home/ravi.bullock/synthetic_splits/synthetic_adult_splits_crosssectional.pkl", "rb") as f:
    splits = pickle.load(f)

image_train, age_train = splits["train"]
image_val, age_val = splits["val"]
image_test, age_test = splits["test"]

#Real Test
# with open("/home/ravi.bullock/ukbb_splits.pkl", "rb") as f:
#     splits = pickle.load(f)

    

# image_train, age_train = splits["train"]
# image_val, age_val = splits["val"]
# image_test, age_test = splits["test"]

# def fix_path(img_path):
#     if img_path.startswith("adult/data"):
#         return img_path.replace("adult/data", "/work/wilms_lab/ravi/adult/data")
#     return img_path

# image_train = [fix_path(p) for p in image_train]
# image_test = [fix_path(p) for p in image_test]
# image_val = [fix_path(p) for p in image_val]

#Correct ages to start with 0
age_train=np.float32((np.array(age_train)-0.)/1.0)
age_test=np.float32((np.array(age_test)-0.)/1.0)
age_val=np.float32((np.array(age_val)-0.)/1.0)

#Dictionaries
train_files = [{"image": img, "age": label, "path": img} for img, label in zip(image_train, age_train)]
test_files = [{"image": img, "age": label, "path": img, "test": 'True'} for img, label in zip(image_test, age_test)]
val_files = [{"image": img, "age": label, "path": img} for img, label in zip(image_val, age_val)]


print("Train Images:", len(train_files),"("+str(np.mean(age_train))+"/"+str(np.std(age_train))+")")
print("Test Images:", len(test_files),"("+str(np.mean(age_test))+"/"+str(np.std(age_test))+")")
print("Validation Images:", len(val_files),"("+str(np.mean(age_val))+"/"+str(np.std(age_val))+")")

#get initial bounding box
img=monai.transforms.LoadImage(reader="ITKReader",image_only=False,ensure_channel_first=True)(image_train[0])
if use_bb:
    bounding_box=monai.transforms.utils.generate_spatial_bounding_box(img[0],lambda x: x > 0,margin=10)

train_transforms = monai.transforms.Compose([monai.transforms.LoadImaged(keys=["image"], ensure_channel_first=True,reader="ITKReader"),
                                             monai.transforms.Spacingd(keys=["image"],pixdim=(pix_spacing,pix_spacing,pix_spacing),mode='bilinear',lazy=True),
                                             monai.transforms.DivisiblePadd(keys=["image"],k=64),
                                             monai.transforms.CenterSpatialCropd(keys=["image"],roi_size=[192,256,192]),
                                             monai.transforms.ScaleIntensityRangePercentilesd(keys="image", lower=1, upper=99, b_min=-1, b_max=1,clip=True),
                                             monai.transforms.ToTensord(keys=["image"],track_meta=False),],lazy=False) #monai.transforms.NormalizeIntensityd(keys=["img"],nonzero=True)

test_transforms = monai.transforms.Compose([monai.transforms.LoadImaged(keys=["image"], ensure_channel_first=True,reader="ITKReader"),
                                            monai.transforms.Spacingd(keys=["image"],pixdim=(pix_spacing,pix_spacing,pix_spacing)),
                                            monai.transforms.DivisiblePadd(keys=["image"],k=64),
                                            monai.transforms.CenterSpatialCropd(keys=["image"],roi_size=[192,256,192]),
                                            monai.transforms.ScaleIntensityRangePercentilesd(keys="image", lower=1, upper=99, b_min=-1, b_max=1,clip=True),
                                            monai.transforms.ToTensord(keys=["image"],track_meta=False)]) #monai.transforms.CenterSpatialCropd(keys=["img"],roi_size=[96,96,96])

check_ds = monai.data.Dataset(data=train_files[0:1],transform=train_transforms)
check_loader = monai.data.DataLoader(check_ds, shuffle=False,batch_size=1, num_workers=0, pin_memory=False)
im_dict = monai.utils.misc.first(check_loader)
#print(im_dict['image'].shape, im_dict['age'])
writer = monai.data.ITKWriter()  # subclass of ImageWriter
writer.set_data_array(torch.squeeze(im_dict['image'][0].data),channel_dim=None)
writer.write("./test_sfcn_abcd.nii.gz")

if model_string=="SFCN":
    net = SFCN.SFCNModelMONAI().to(device)
else:
    net = MedNeXtRegressor.MedNeXtRegressor(1,64,1,do_res=False).to(device)
#print(torchinfo.summary(net))
#dummy call to avoid init problem
net.forward(im_dict['image'].to(device))

loss_fn = torch.nn.MSELoss()
#loss_fn = torch.nn.SmoothL1Loss(beta=.5)
opt = torch.optim.AdamW(net.parameters(), lr,weight_decay=wd,eps=eps)
#opt = torch.optim.Adam(net.parameters(), lr)
#torch_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, .1, last_epoch=- 1, verbose=True)
if lr_scheduler:
    #torch_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[30,70], gamma=0.5)
    #scheduler = ignite.handlers.param_scheduler.LRScheduler(torch_lr_scheduler)
    scheduler=ignite.handlers.param_scheduler.CosineAnnealingScheduler(opt, "lr", lr, lr*0.01, 10, start_value_mult=0.75, end_value_mult=1.0)

def output_transform_fn(x, y, y_pred, loss):
    # return only the loss is actually the default behavior for
    # trainer engine, but you can return anything you want 
    #print(loss.item())
    return loss.item()

# Ignite trainer expects batch=(img, label) and returns output=loss at every iteration,
# user can add output_transform to return other values, like: y_pred, y, etc.
def prepare_batch(batch, device=None, non_blocking=False):
    return ignite.engine._prepare_batch((batch["image"], batch["age"]), device, non_blocking)

trainer = ignite.engine.create_supervised_trainer(net, opt, loss_fn, device, False,output_transform=output_transform_fn,prepare_batch=prepare_batch,gradient_accumulation_steps=gradient_accumulation_steps,amp_mode=amp_mode,scaler=grad_scaler)

if lr_scheduler:
    trainer.add_event_handler(ignite.engine.Events.EPOCH_STARTED, scheduler)

train_metrics={"train MAE": ignite.metrics.MeanAbsoluteError(), "train loss": ignite.metrics.Loss(loss_fn)}
train_evaluator = ignite.engine.create_supervised_evaluator(net, metrics=train_metrics, device=device,prepare_batch=prepare_batch,amp_mode=amp_mode)

val_metrics = {"val MAE": ignite.metrics.MeanAbsoluteError(), "val loss":ignite.metrics.Loss(loss_fn)}
val_evaluator = ignite.engine.create_supervised_evaluator(net, val_metrics, device, True,prepare_batch=prepare_batch,amp_mode=amp_mode)

best_model_handler = ignite.handlers.ModelCheckpoint('/work/wilms_lab/ravi/spie_extension_models_'+model_string+'_'+data_string, model_string, n_saved=2, create_dir=True, require_empty=False,score_name="val_MAE",score_function=ignite.handlers.ModelCheckpoint.get_default_score_fn("val MAE", -1.0))
val_evaluator.add_event_handler(Events.COMPLETED, best_model_handler,{'model': net})

test_metrics = {"test MAE": ignite.metrics.MeanAbsoluteError(), "test loss":ignite.metrics.Loss(loss_fn)}
test_evaluator = ignite.engine.create_supervised_evaluator(net, test_metrics, device, True,prepare_batch=prepare_batch,amp_mode=amp_mode)

val_ds = monai.data.Dataset(data=val_files,transform=test_transforms)
val_loader = monai.data.DataLoader(val_ds, batch_size=batch_size_test, num_workers=num_workers, pin_memory=False)

@trainer.on(Events.EPOCH_COMPLETED(every=evaluate_every_x_epochs))
def compute_metrics(engine):
    train_evaluator.run(train_loader_no_rand)
    print("current learning rate:",opt.param_groups[0]["lr"])
    print("train metrics:",train_evaluator.state.metrics)
    if not train_only:
        val_evaluator.run(val_loader)
        test_evaluator.run(test_loader)
        print("val metrics:",val_evaluator.state.metrics)
        print("test metrics:",test_evaluator.state.metrics)




train_ds = monai.data.Dataset(data=train_files,transform=train_transforms)
train_loader = monai.data.DataLoader(train_ds, shuffle=True, batch_size=batch_size_train, num_workers=num_workers, pin_memory=pin_memory)

train_ds_no_rand = monai.data.Dataset(data=train_files,transform=test_transforms)
train_loader_no_rand = monai.data.DataLoader(train_ds_no_rand, shuffle=False, batch_size=batch_size_test, num_workers=num_workers, pin_memory=False)

test_ds = monai.data.Dataset(data=test_files,transform=test_transforms) #Change this
test_loader = monai.data.DataLoader(test_ds, shuffle=False, batch_size=batch_size_test, num_workers=num_workers, pin_memory=False)

ignite.metrics.RunningAverage(output_transform=lambda x: x).attach(trainer, 'avg. loss')
ignite.contrib.handlers.ProgressBar().attach(trainer,['avg. loss'])

if checkpoint_filename is not None:
    checkpoint = torch.load(checkpoint_filename, map_location=device) 
    ignite.handlers.Checkpoint.load_objects(to_load={'model': net}, checkpoint=checkpoint)  

if not evaluate_only:
    state = trainer.run(train_loader, train_epochs)
    print(state)
#else:
#    compute_metrics(trainer)

#torch.save(net.state_dict(), "/work/wilms_lab/ravi/SFCN_Models_Real/Adult/SFCN_Adult_output.pt")

def predict_on_batch(engine, batch):
    net.eval()
    if evaluate_only:
        log_path = '/work/wilms_lab/ravi/spie_extension_models/SFCN_Adult_Real.txt'
    else:
        log_path = '/work/wilms_lab/ravi/spie_extension_models/SFCN_Other.txt'

    with torch.no_grad(), open(log_path, "a") as log_file: 
        x, y = prepare_batch(batch, device=device)
        y_pred = net(x)

        for i in range(len(y)):
            actual = y[i].item()
            predicted = y_pred[i].item()
            ae = abs(actual - predicted)

            if "path" in batch:
                subject_path = str(batch["path"][i])
                subject_id = os.path.splitext(os.path.splitext(os.path.basename(subject_path))[0])[0]
                subject_id = subject_id.replace('_T1w', '').replace('_register', '')
                if "test" in batch and str(batch["test"][i]) == 'True':
                    line = f"{subject_id}: Actual: {actual:.3f}, Predicted: {predicted:.3f}, AE: {ae}\n"
                    log_file.write(line)
                #print(line.strip())

    return y_pred, y

def correct_age_pred(pred_age,chron_age,intercept,slope):
    return pred_age+(chron_age-(intercept+slope*chron_age))

def print_corr(pred_age,chron_age):
    corr_matrix=torch.corrcoef(torch.vstack((pred_age,chron_age,pred_age-chron_age)))
    print("pred vs chron:",corr_matrix[0,1].cpu().numpy(),"gap vs chron:",corr_matrix[1,2].cpu().numpy())

compute_engine = ignite.engine.engine.Engine(predict_on_batch)
eos=ignite.handlers.stores.EpochOutputStore()
eos.attach(compute_engine)
eos.reset()
train_loader = monai.data.DataLoader(monai.data.Dataset(data=train_files,transform=test_transforms), batch_size=batch_size_test, num_workers=num_workers, pin_memory=False)
compute_engine.run(train_loader)
res_train=torch.stack([torch.cat([e[0] for e in eos.data]),torch.cat([e[1] for e in eos.data])]).T
diff_val=torch.abs(res_train[:,0]-res_train[:,1])
print("Train data:",np.round(torch.mean(diff_val).cpu().numpy(),decimals=2),'+/-',np.round(torch.std(diff_val).cpu().numpy(),decimals=2))

eos.reset()
val_loader = monai.data.DataLoader(val_ds, batch_size=batch_size_test, num_workers=num_workers, pin_memory=False)
compute_engine.run(val_loader)
res_val=torch.stack([torch.cat([e[0] for e in eos.data]),torch.cat([e[1] for e in eos.data])]).T
#reg_result=torch.linalg.lstsq(torch.vstack((res_val[:,1],torch.ones(res_val[:,1].size()).to(device))).T,res_val[:,0])[0]
reg_result=(1.0,0.0)
#print_corr(res_val[:,0],res_val[:,1])
res_val_corr=correct_age_pred(res_val[:,0],res_val[:,1],reg_result[1],reg_result[0])
#print_corr(res_val_corr,res_val[:,1])
diff_val=res_val_corr-res_val[:,1]
print("MAE Val data:",np.round(torch.mean(torch.abs(diff_val)).cpu().numpy(),decimals=2),'+/-',np.round(torch.std(torch.abs(diff_val)).cpu().numpy(),decimals=2))
print("Gap Val data:",np.round(torch.mean(diff_val).cpu().numpy(),decimals=2),'+/-',np.round(torch.std(diff_val).cpu().numpy(),decimals=2))

eos.reset()
compute_engine.run(test_loader)
res_con=torch.stack([torch.cat([e[0] for e in eos.data]),torch.cat([e[1] for e in eos.data])]).T
#torch.corrcoef(torch.vstack((res_con[:,0],res_con[:,1],res_con[:,0]-res_con[:,1])))
res_con_corr=correct_age_pred(res_con[:,0],res_con[:,1],reg_result[1],reg_result[0])
#torch.corrcoef(torch.vstack((res_con_corr,res_con[:,1],res_con_corr-res_con[:,1])))
diff_con=res_con_corr-res_con[:,1]
print("MAE:", torch.abs(diff_con))
print("MAE Test data:",np.round(torch.mean(torch.abs(diff_con)).cpu().numpy(),decimals=2),'+/-',np.round(torch.std(torch.abs(diff_con)).cpu().numpy(),decimals=2))
print("GAP Test data:",np.round(torch.mean(diff_con).cpu().numpy(),decimals=2),'+/-',np.round(torch.std(diff_con).cpu().numpy(),decimals=2))


# # #XAI
# def generate_xai(net, test_item,output_dir='./'):
#     net.eval()
#     img=torch.unsqueeze(test_item['image'],0).to(device)
#     subject_path = str(test_item["path"])
#     subject_id = os.path.splitext(os.path.splitext(os.path.basename(subject_path))[0])[0]
#     subject_id = subject_id.replace('_T1w', '')
#     cap_sal=captum.attr.Saliency(net)
#     cap_nt = captum.attr.NoiseTunnel(cap_sal)
#     cap_map = cap_nt.attribute(img, nt_type='smoothgrad',nt_samples=25,nt_samples_batch_size=1,stdevs=0.15*(img.max()-img.min()).item())
#     new_dict={"image":test_item['image'] ,"grad":cap_map[0].detach().cpu()}
#     print(str(test_item["path"]))
#     savers=monai.transforms.Compose([
#             monai.transforms.ShiftIntensityd(keys=["image"],offset=.5),
#             monai.transforms.SaveImaged(keys=["image"],separate_folder="False",output_dir=os.path.join(output_dir, subject_id),output_postfix="img",resample=True,mode='bilinear',padding_mode='zeros',writer="ITKWriter"),monai.transforms.ScaleIntensityd(keys=["grad"]),monai.transforms.SaveImaged(keys=["grad"],separate_folder="False",output_dir=os.path.join(output_dir, subject_id),output_postfix="grad",resample=True,mode='nearest',padding_mode='zeros',writer="ITKWriter")])
#     _=savers(new_dict)


# # List of target ages for XAI
# target_ages = [50, 60, 70, 80]

# # Define output directory
# output_base_dir = "/home/ravi.bullock/BrainAgeVoxelLevelPredictions_SPIEExtension/Adult_Real/XAI"
# os.makedirs(output_base_dir, exist_ok=True)

# # Optional: to keep track of which files were processed
# xai_orig_files_list = []

# print(f"Starting filtered XAI generation for ages: {target_ages}")

# # Loop through test dataset and generate saliency maps
# for idx in range(len(test_ds)):
#     test_item = test_ds.__getitem__(idx)
    
#     # Get the age from the metadata (rounding to handle float precision)
#     current_age = int(round(float(test_item["age"])))
    
#     # Only proceed if the age is in our target list
#     if current_age in target_ages:
#         try:
#             print(f"Processing Subject at Age {current_age}: {test_item['path']}")
#             generate_xai(net, test_item, output_base_dir)
            
#             # Optional: store path
#             xai_orig_files_list.append(str(test_item['path']))
        
#         except Exception as e:
#             print(f"Failed at index {idx} (Age {current_age}): {e}")
#     else:
#         # Skip subjects that don't match the target ages
#         continue
