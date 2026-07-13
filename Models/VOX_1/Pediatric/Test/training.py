from monai.losses import DiceLoss
import torch
import numpy as np
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose, EnsureType
from monai.data.utils import pad_list_data_collate
from monai.data import decollate_batch
import wandb
from loss import global_mae_loss, voxel_mae
import os

def train(train_loader, val_loader, model, optimizer, scheduler, max_epochs, root_dir, start_epoch=1, best_val_loss=float('inf')):
    
    model.train()
    post_seglabel = Compose([EnsureType("tensor"), AsDiscrete(to_onehot=4)])
    
    # CHANGE 1: Added softmax=True. The paper's Eq 1 assumes probabilities, not raw logits.
    loss_object = DiceLoss(to_onehot_y=True, softmax=True)
    
    # Optimization: include_background=True matches the paper's "average of all classes"
    metric_object = DiceMetric(include_background=True, reduction="mean")

    for epoch in range(start_epoch, max_epochs + 1):
        train_loss = 0.0
        val_loss = 0.0
    
        print(f"\nEpoch {epoch}")
        
        if epoch < 8:
            dice_coef, glob_coef, voxel_coef = 80, 1, 1
        elif 8 <= epoch < 21:
            dice_coef, glob_coef, voxel_coef = 40, 1, 1
        else:
            dice_coef, glob_coef, voxel_coef = 15, 0.7, 1.3

        print("Train:", end="")
        model.train()
        for step, batch in enumerate(train_loader):
            img, brain_mask, tissue_mask, age = (batch["img"].cuda(), batch["brain_mask"].cuda(),
                                                batch["seg_label"].cuda(), batch["age_label"].cuda())
            brain_img = img * brain_mask
            optimizer.zero_grad()
            pred_tissue_mask, pred_glob_age, pred_voxel_age = model(brain_img)

            loss = (dice_coef * loss_object(pred_tissue_mask, tissue_mask)) + \
                   (glob_coef * global_mae_loss(pred_glob_age, age, brain_mask)) + \
                   (voxel_coef * voxel_mae(pred_voxel_age, age, brain_mask))
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            print("=", end="")
        
        train_loss /= (step + 1)

        print("\nVal:", end="")
        model.eval()
        with torch.no_grad():
            mae_val, voxel_mae_val = 0.0, 0.0
            for step, batch in enumerate(val_loader):
                img, brain_mask, tissue_mask, age = (batch["img"].cuda(), batch["brain_mask"].cuda(),
                                                    batch["seg_label"].cuda(), batch["age_label"].cuda())
                brain_img = img * brain_mask
                pred_tissue_mask, pred_glob_age, pred_voxel_age = model(brain_img)

                v_loss = (dice_coef * loss_object(pred_tissue_mask, tissue_mask)) + \
                         (glob_coef * global_mae_loss(pred_glob_age, age, brain_mask)) + \
                         (voxel_coef * voxel_mae(pred_voxel_age, age, brain_mask))
                val_loss += v_loss.item()
                
                # Metrics
                tissue_mask_decoll = [post_seglabel(i) for i in decollate_batch(tissue_mask)]
                tissue_mask_decoll = pad_list_data_collate(tissue_mask_decoll)

                # CHANGE 2: Apply Argmax/OneHot to prediction. 
                # This ensures we compare class 1 vs class 1, instead of raw scores.
                pred_discrete = [AsDiscrete(argmax=True, to_onehot=4)(i) for i in decollate_batch(pred_tissue_mask)]
                pred_discrete = pad_list_data_collate(pred_discrete)
                
                metric_object(y_pred=pred_discrete, y=tissue_mask_decoll)
                
                mae_val += global_mae_loss(pred_glob_age, age, brain_mask)
                voxel_mae_val += voxel_mae(pred_voxel_age, age, brain_mask)
                print("=", end="")

            val_loss /= (step + 1)
            dice_metric = metric_object.aggregate().item()
            metric_object.reset()
            mae_val /= (step + 1)
            voxel_mae_val /= (step + 1)

        lr_current = optimizer.param_groups[0]['lr']
        print(f"\nTraining epoch {epoch}, train loss: {train_loss:.4f}, val loss: {val_loss:.4f}, "
              f"val dice score: {dice_metric:.4f}, val glob mae: {mae_val.item():.4f}, "
              f"val voxel mae: {voxel_mae_val.item():.4f} | LR: {lr_current}")
        
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dice": dice_metric,
            "mae_global": mae_val.item(),
            "mae_voxel": voxel_mae_val.item(),
            "lr": optimizer.param_groups[0]['lr']
        })

        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_val_loss': best_val_loss
        }

        if val_loss < best_val_loss:
            print("--> Saving Best Model")
            best_val_loss = val_loss
            torch.save(state, os.path.join(root_dir, "model_best.pth.tar"))

        torch.save(state, os.path.join(root_dir, "checkpoint_last.pth.tar"))
        scheduler.step()
        
    return