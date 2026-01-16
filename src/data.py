import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import tifffile


def display_arr_stats(arr):
    shape, vmin, vmax, vmean, vstd = arr.shape, np.min(arr), np.max(arr), np.mean(arr), np.std(arr)
    print('shape:{} | min:{:.3f} | max:{:.3f} | mean:{:.3f} | std:{:.3f}'.format(shape, vmin, vmax, vmean, vstd))


def display_tensor_stats(tensor):
    shape, vmin, vmax, vmean, vstd = tensor.shape, tensor.min(), tensor.max(), torch.mean(tensor), torch.std(tensor)
    print('shape:{} | min:{:.3f} | max:{:.3f} | mean:{:.3f} | std:{:.3f}'.format(shape, vmin, vmax, vmean, vstd))


def create_grid(h, w):
    grid_y, grid_x = torch.meshgrid([torch.linspace(0, 1, steps=h),
                                     torch.linspace(0, 1, steps=w)])
    grid = torch.stack([grid_y, grid_x], dim=-1)
    return grid


def create_grid_3d(c, h, w):
    grid_z, grid_y, grid_x = torch.meshgrid([torch.linspace(0, 1, steps=c), \
                                            torch.linspace(0, 1, steps=h), \
                                            torch.linspace(0, 1, steps=w)])
    grid = torch.stack([grid_z, grid_y, grid_x], dim=-1)
    return grid


class TIFFDataset(Dataset):
    """
    Dataset for loading 3D TIFF microscopy images.
    
    Crops the image to the specified bounding box matching SWC skeleton extent.
    """
    
    def __init__(self, img_path, img_dim):
        """
        Args:
            img_path: Path to TIFF file
            img_dim: Target volume dimensions (D, H, W)
        """
        self.img_dim = (img_dim, img_dim, img_dim) if type(img_dim) == int else tuple(img_dim)
        
        # Read TIFF image
        image = tifffile.imread(img_path)
        print(f"Loaded TIFF: {img_path}, shape: {image.shape}")
        
        # Crop to match SWC skeleton bounding box
        # Original image is (Z, Y, X) = (163, 1024, 1024)
        # SWC bounds: X=[0-812], Y=[206-852], Z=[9-108]
        # Crop: Z=[9:109], Y=[206:853], X=[0:813]
        D, H, W = self.img_dim
        
        # Crop to bounding box
        image = image[9:9+D, 206:206+H, 0:W]
        print(f"Cropped to: {image.shape} (target: {self.img_dim})")
        
        # Convert to float and normalize to [0, 1]
        image = image.astype(np.float32)
        if image.max() > 0:
            image = image / image.max()
        
        # Add channel dimension: (D, H, W) -> (D, H, W, 1)
        self.img = torch.tensor(image, dtype=torch.float32).unsqueeze(-1)
        display_tensor_stats(self.img)
    
    def __getitem__(self, idx):
        grid = create_grid_3d(*self.img_dim)
        return grid, self.img
    
    def __len__(self):
        return 1


class SWCSkeletonDataset(Dataset):
    """
    Dataset for SWC-based neuron reconstruction.
    
    This dataset creates a 3D volume from SWC skeleton for training.
    Instead of loading a pre-existing CT volume, it generates a target volume
    by rasterizing the SWC skeleton with radius information.
    
    Coordinate system:
    - SWC files use (X, Y, Z) coordinates
    - Volume uses (D, H, W) = (Z, Y, X) indexing
    - Normalized positions are in [0.05, 0.95] range with margin
    """
    
    def __init__(self, swc_path, img_dim, intensity_value=1.0):
        """
        Args:
            swc_path: Path to SWC skeleton file
            img_dim: Target volume dimensions (D, H, W) or single int for cubic
            intensity_value: Intensity value for neuron voxels
        """
        from swc_utils import parse_swc_file, swc_to_arrays, get_skeleton_bounds, densify_skeleton
        
        self.img_dim = (img_dim, img_dim, img_dim) if type(img_dim) == int else tuple(img_dim)
        self.swc_path = swc_path
        
        # Parse SWC file
        nodes = parse_swc_file(swc_path)
        positions, radii, parent_ids = swc_to_arrays(nodes)
        # positions are in (X, Y, Z) order from SWC file
        
        print(f"Loaded SWC with {len(nodes)} nodes from {swc_path}")
        
        # Convert SWC (X, Y, Z) to volume (D, H, W) = (Z, Y, X) coordinate order
        # This must match the coordinate transform in swc_utils.create_gaussian_params_from_swc
        positions_dhw = positions[:, [2, 1, 0]]  # Reorder: X,Y,Z -> Z,Y,X
        
        # Normalize to [0, 1] based on skeleton bounds
        min_bounds, max_bounds = get_skeleton_bounds(positions_dhw)
        extent = max_bounds - min_bounds
        extent = np.where(extent < 1e-6, 1.0, extent)
        positions_norm = (positions_dhw - min_bounds) / extent
        
        # Apply margin (same as swc_utils.normalize_positions)
        margin = 0.05
        positions_norm = positions_norm * (1 - 2 * margin) + margin
        
        # Scale radii relative to volume size
        max_dim = max(self.img_dim)
        radii = radii / max_dim * 5  # Adjusted scale
        
        # Densify skeleton for volume rendering
        dense_positions, dense_radii = densify_skeleton(
            positions_norm, radii, parent_ids, points_per_unit=20.0
        )
        
        # Rasterize skeleton to volume
        self.img = self._rasterize_skeleton(dense_positions, dense_radii, intensity_value)
        display_tensor_stats(self.img)
    
    def _rasterize_skeleton(self, positions, radii, intensity_value):
        """
        Rasterize skeleton points with radius to a 3D volume.
        
        Creates a soft volume where intensity falls off based on distance
        from skeleton centerline, using radius information.
        """
        D, H, W = self.img_dim
        volume = torch.zeros((D, H, W, 1), dtype=torch.float32)
        
        # Convert normalized positions to voxel coordinates
        voxel_coords = positions * np.array([D-1, H-1, W-1])
        
        for i, (pos, radius) in enumerate(zip(voxel_coords, radii)):
            # Determine affected voxels based on radius (increased for better visibility)
            voxel_radius = max(3, int(radius * max(self.img_dim) * 2.0))
            
            z, y, x = int(pos[0]), int(pos[1]), int(pos[2])
            
            # Define bounding box
            z_min = max(0, z - voxel_radius)
            z_max = min(D, z + voxel_radius + 1)
            y_min = max(0, y - voxel_radius)
            y_max = min(H, y + voxel_radius + 1)
            x_min = max(0, x - voxel_radius)
            x_max = min(W, x + voxel_radius + 1)
            
            # Create local coordinate grids
            zz, yy, xx = np.mgrid[z_min:z_max, y_min:y_max, x_min:x_max]
            
            # Calculate distance from skeleton point
            dist_sq = (zz - pos[0])**2 + (yy - pos[1])**2 + (xx - pos[2])**2
            
            # Gaussian falloff based on radius (increased for thicker rendering)
            sigma = max(2.0, radius * max(self.img_dim) * 1.0)
            intensity = intensity_value * np.exp(-dist_sq / (2 * sigma**2))
            
            # Accumulate (max for overlapping regions)
            current = volume[z_min:z_max, y_min:y_max, x_min:x_max, 0].numpy()
            volume[z_min:z_max, y_min:y_max, x_min:x_max, 0] = torch.tensor(
                np.maximum(current, intensity), dtype=torch.float32
            )
        
        # Normalize volume
        if volume.max() > 0:
            volume = volume / volume.max()
        
        return volume
    
    def __getitem__(self, idx):
        grid = create_grid_3d(*self.img_dim)
        return grid, self.img
    
    def __len__(self):
        return 1


class ImageDataset_3D(Dataset):

    def __init__(self, img_path, img_dim):
        '''
        img_dim: new image size [z, h, w]
        '''
        self.img_dim = (img_dim, img_dim, img_dim) if type(img_dim) == int else tuple(img_dim)
        image = np.load(img_path)['data']  # [C, H, W]

        # Crop slices in z dim
        center_idx = int(image.shape[0] / 2)
        num_slice = int(self.img_dim[0] / 2)
        image = image[center_idx-num_slice:center_idx+num_slice, :, :]
        im_size = image.shape
        print(image.shape, center_idx, num_slice)

        # Complete 3D input image as a squared x-y image
        if not(im_size[1] == im_size[2]):
            zerp_padding = np.zeros([im_size[0], im_size[1], np.int((im_size[1]-im_size[2])/2)])
            image = np.concatenate([zerp_padding, image, zerp_padding], axis=-1)

        # Resize image in x-y plane
        image = torch.tensor(image, dtype=torch.float32)[None, ...]  # [B, C, H, W]
        image = F.interpolate(image, size=(self.img_dim[1], self.img_dim[2]), mode='bilinear', align_corners=False)

        # Scaling normalization
        #image = image / torch.max(image)  # [B, C, H, W], [0, 1]
        self.img = image.permute(1, 2, 3, 0)  # [C, H, W, 1]
        display_tensor_stats(self.img)

    def __getitem__(self, idx):
        grid = create_grid_3d(*self.img_dim)
        return grid, self.img

    def __len__(self):
        return 1



class ImageDataset_2D(Dataset):

    def __init__(self, img_path, img_dim, img_slice):
        '''
        img_dim: new image size [h, w]
        '''
        self.img_dim = (img_dim, img_dim) if type(img_dim) == int else tuple(img_dim)
        image = np.load(img_path)['data']
        image = image[img_slice, :, :]  # Choose one slice as 2D CT image
        imsize = image.shape

        # Complete as a squared image
        if not(imsize[0] == imsize[1]):
            zerp_padding = np.zeros([imsize[0], np.int((imsize[0] - imsize[1])/2)])
            image = np.concatenate([zerp_padding, image, zerp_padding], axis=1)

        # Interpolate image to predefined size
        image = cv2.resize(image, self.img_dim[::-1], interpolation=cv2.INTER_LINEAR) 

        # Scaling normalization
        image = image / np.max(image)
        self.img = torch.tensor(image, dtype=torch.float32)[:, :, None]
        display_tensor_stats(self.img)

        
    def __getitem__(self, idx):
        grid = create_grid(*self.img_dim)
        return grid, self.img

    def __len__(self):
        return 1


class ImageDataset(Dataset):

    def __init__(self, img_path, img_dim):
        self.img_dim = (img_dim, img_dim) if type(img_dim) == int else img_dim

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  
        h, w = image.shape
        left_w = int((w - h) / 2)

        image = image[:, left_w:left_w + h]
        image = cv2.resize(image, self.img_dim, interpolation=cv2.INTER_LINEAR)
        self.img = image

    def __getitem__(self, idx):
        image = self.img / 255
        grid = create_grid(*self.img_dim[::-1])

        return grid, torch.tensor(image, dtype=torch.float32)[:, :, None]

    def __len__(self):
        return 1


