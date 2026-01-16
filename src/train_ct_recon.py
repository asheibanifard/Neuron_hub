import os
import argparse
import shutil
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from utils import get_config, prepare_sub_folder, get_data_loader, save_image_3d
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skeleton_loss import SkeletonConstraint, skeleton_distance_loss

import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument('--swc_path', type=str, default=None, help="Path to SWC skeleton file for initialization")
parser.add_argument('--use_swc', action='store_true', help="Use SWC-based Gaussian initialization instead of FBP")

# Load experiment setting
opts = parser.parse_args()
config = get_config(opts.config)
max_iter = config['max_iter']

cudnn.benchmark = True

# Setup output folder
output_folder = os.path.splitext(os.path.basename(opts.config))[0]
output_subfolder = config['data']
model_name = os.path.join(output_folder, output_subfolder)


output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder

wandb.init(project="3dgr-ct-skeleton", 
           name = "swc_skeleton_constrained",
           config=config,
           mode="disabled")  # Disable wandb logging

# Setup data loader
print('Load image: {}'.format(config['img_path']))
data_loader = get_data_loader(config['data'], config['img_path'], config['img_size'], img_slice=None, train=True, batch_size=config['batch_size'])

config['img_size'] = (config['img_size'], config['img_size'], config['img_size']) if type(config['img_size']) == int else tuple(config['img_size'])
slice_idx = list(range(0, config['img_size'][0], int(config['img_size'][0]/config['display_image_num'])))
if config['num_proj'] > config['display_image_num']:
    proj_idx = list(range(0, config['num_proj'], int(config['num_proj']/config['display_image_num'])))
else:
    proj_idx = list(range(0, config['num_proj']))



class OptimizationParams():
    def __init__(self):
        self.position_lr_init = 0.002
        self.position_lr_final = 0.000002
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.intensity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01

def tv_regularization(image):
    return torch.mean(torch.abs(image[:, 1:, :, :, :] - image[:, :-1, :, :, :])) + torch.mean(torch.abs(image[:, :, 1:, :, :] - image[:, :, :-1, :, :])) + torch.mean(torch.abs(image[:, :, :, 1:, :] - image[:, :, :, :-1, :]))


# NO CT PROJECTOR - Direct volumetric training with skeleton constraint
print("\n" + "="*60)
print("DIRECT VOLUMETRIC TRAINING (No CT Projection)")
print("  - Direct 3D MSE loss against target volume")
print("  - Skeleton constraint keeps Gaussians on neuron structure")
print("="*60 + "\n")

for it, (grid, image) in enumerate(data_loader):
    grid = grid.cuda()  
    image = image.cuda()  
    
    # Prepare low-resolution grids/images for coarse-to-fine training
    image_low_resos = []
    grid_low_resos = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                image_low_reso = image[:, i::2, j::2, k::2, :]
                image_low_resos.append(image_low_reso)
                grid_low_resos.append(grid[:, i::2, j::2, k::2, :])

    # Data for training and testing
    test_data = (grid, image)  # [bs, z, x, y, 1]

    save_image_3d(test_data[1], slice_idx, os.path.join(image_directory, "test.png"))

    # Setup Gaussian Model
    from models.gaussian_model import GaussianModel

    op = OptimizationParams()
    gaussians = GaussianModel()
    
    # SWC skeleton initialization (required for this mode)
    swc_path = opts.swc_path or config.get('swc_path', None)
    
    if swc_path is None:
        raise ValueError("SWC path is required for skeleton-based training. Use --swc_path or set swc_path in config.")
    
    print("\n" + "="*60)
    print("SKELETON-CONSTRAINED Gaussian Initialization")
    print("  - Gaussians placed ONLY along neuron structure")
    print("  - NO FBP simulation")
    print("  - Direct 3D MSE + skeleton constraint")
    print("="*60 + "\n")
    
    volume_size = (config['img_size'][0]//2, config['img_size'][1]//2, config['img_size'][2]//2)
    gaussians.create_from_swc(
        swc_path=swc_path,
        volume_size=volume_size,
        num_samples=config['num_gaussian'],
        ini_intensity=config['ini_intensity'],
        spatial_lr_scale=config['spatial_lr_scale'],
        densify=config.get('swc_densify', True),
        points_per_unit=config.get('swc_points_per_unit', 5.0),
        radius_based_density=config.get('swc_radius_density', True)
    )
    
    # Create skeleton constraint for loss computation
    skeleton_weight = config.get('skeleton_weight', 0.5)  # Strong by default
    print(f"\nSkeleton constraint enabled (weight={skeleton_weight})")
    skeleton_constraint = SkeletonConstraint(
        swc_path=swc_path,
        volume_size=volume_size,
        device='cuda',
        densify_points=True,
        points_per_unit=1.0,
        max_skeleton_points=2000
    )

    gaussians.training_setup(op)

    # Train model - DIRECT 3D MSE LOSS (no projections!)
    import time
    start_time = time.time()
    
    for iteration in range(max_iter):
        iter_start = time.time()

        gaussians.update_learning_rate(iteration)

        # Coarse-to-fine: start with low resolution
        if iteration < config['low_reso_stage']:
            train_output = gaussians.grid_sample(grid_low_resos[iteration%8])
            target_image = image_low_resos[iteration%8]
        else:
            train_output = gaussians.grid_sample(grid)
            target_image = image
        
        # DIRECT 3D MSE LOSS (no projections!)
        loss_mse = torch.nn.functional.mse_loss(train_output, target_image)
        loss_tv = config['tv_weight'] * tv_regularization(train_output) if config['tv_weight'] > 0 else 0
        
        # Skeleton constraint - keeps Gaussians near skeleton structure
        skel_loss = skeleton_distance_loss(gaussians._xyz, skeleton_constraint, 
                                           weight=skeleton_weight)
        
        # Total loss
        loss = loss_mse + loss_tv + skel_loss

        loss.backward()
        
        # Accumulate gradients for densification BEFORE optimizer step
        if config.get('do_density_control', False):
            gaussians.add_densification_stats()

        gaussians.optimizer.step()
        
        # Clamp scales to prevent Gaussians from growing too large
        max_scale = config.get('max_scale', None)
        if max_scale is not None:
            with torch.no_grad():
                # Clamp the raw scaling parameter (log space)
                max_log_scale = torch.log(torch.tensor(max_scale, device='cuda'))
                gaussians._scaling.data = torch.clamp(gaussians._scaling.data, max=max_log_scale)

        if config.get('do_density_control', False):
            with torch.no_grad():
                # Densification with scale limits
                if gaussians.get_xyz.shape[-2] < config['max_gaussians'] and iteration < config['densify_until_iter']:
                        if iteration > config['densify_from_iter'] and iteration % config['densification_interval'] == 0:
                            gaussians.densify_and_prune(config['max_grad'], config['min_intensity'], 
                                                       sigma_extent=config['sigma_extent'], max_scale=max_scale)
                            # Reset stats after densification
                            gaussians.reset_densification_stats()
            
        gaussians.optimizer.zero_grad(set_to_none = True)    

        # Compute training psnr
        if (iteration + 1) % config['log_iter'] == 0:
            iter_time = time.time() - iter_start
            elapsed_time = time.time() - start_time
            remaining_iters = max_iter - (iteration + 1)
            avg_iter_time = elapsed_time / (iteration + 1)
            eta = remaining_iters * avg_iter_time
            
            # Direct 3D PSNR
            train_psnr = -10 * torch.log10(loss_mse).item()
            train_loss = loss.item()

            print("[Iter: {}/{}] Loss: {:.4g} | MSE: {:.4g} | Skel: {:.4g} | PSNR: {:.2f}dB | Time: {:.2f}s/iter | ETA: {:.1f}s".format(
                iteration + 1, max_iter, train_loss, loss_mse.item(), skel_loss.item(), train_psnr, avg_iter_time, eta))
            # Log training metrics to wandb
            wandb.log({
                "Iteration": iteration + 1,
                "Train Loss": train_loss,
                "Train PSNR": train_psnr,
                "Iter Time (s)": avg_iter_time,
            })
        
        # Save model checkpoint every 1000 iterations
        if (iteration + 1) % 1000 == 0:
            checkpoint_path = os.path.join(checkpoint_directory, f"model_iter_{iteration + 1}.pth")
            torch.save({
                'iteration': iteration + 1,
                'xyz': gaussians._xyz.data,
                'intensity': gaussians._intensity.data,
                'scaling': gaussians._scaling.data,
                'rotation': gaussians._rotation.data,
                'num_gaussians': gaussians._xyz.shape[0],
                'config': config,
            }, checkpoint_path)
            print(f"  [Checkpoint saved: {checkpoint_path}]")

        # Compute testing psnr
        if iteration == 0 or (iteration + 1) % config['val_iter'] == 0:

            with torch.no_grad():
                test_output = gaussians.grid_sample(test_data[0])  # [bs, z, x, y, 3]
                test_loss = 0.5 * torch.mean((test_output - test_data[1])**2)
                test_psnr = - 10 * torch.log10(2 * test_loss).item()
                test_loss = test_loss.item()
                test_ssim = compare_ssim(test_output.transpose(1,4).squeeze().cpu().numpy(), test_data[1].transpose(1,4).squeeze().cpu().numpy(), data_range=1.0, channel_axis=-1)

                test_output_low_reso = gaussians.grid_sample(grid_low_resos[iteration%8])  # [bs, z, x, y, 3]
                test_loss_low_reso = 0.5 * torch.mean((test_output_low_reso - image_low_resos[iteration%8])**2)
                test_psnr_low_reso = - 10 * torch.log10(2 * test_loss_low_reso).item()
                test_loss_low_reso = test_loss_low_reso.item()
                test_ssim_low_reso = compare_ssim(test_output_low_reso.transpose(1,4).squeeze().cpu().numpy(), image_low_reso.transpose(1,4).squeeze().cpu().numpy(), data_range=1.0, channel_axis=-1)

            save_image_3d(test_output, slice_idx, os.path.join(image_directory, "recon_{}_{:.4g}dB_ssim{:.4g}.png".format(iteration + 1, test_psnr, test_ssim)))

            wandb.log({
                "Iteration": iteration + 1,
                "Test Loss": test_loss, 
                "Test PSNR": test_psnr, 
                "Test SSIM": test_ssim, 
                "Test Loss-low_reso": test_loss_low_reso, 
                "Test PSNR-low_reso": test_psnr_low_reso, 
                "Test SSIM-low_reso": test_ssim_low_reso, 
            })

    # Save final model
    print("\n" + "="*60)
    print("Training Complete! Saving model...")
    print("="*60)
    
    model_path = os.path.join(checkpoint_directory, "final_model.pth")
    torch.save({
        'xyz': gaussians._xyz.data,
        'intensity': gaussians._intensity.data,
        'scaling': gaussians._scaling.data,
        'rotation': gaussians._rotation.data,
        'num_gaussians': gaussians._xyz.shape[0],
        'config': config,
    }, model_path)
    print(f"Model saved to: {model_path}")
    print(f"Final number of Gaussians: {gaussians._xyz.shape[0]}")
    print(f"Final Test PSNR: {test_psnr:.2f} dB")
    print(f"Final Test SSIM: {test_ssim:.4f}")
