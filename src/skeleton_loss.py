"""
Skeleton constraint loss for 3DGR-CT

Penalizes Gaussians that drift too far from the SWC skeleton.
"""

import torch
import numpy as np
from swc_utils import parse_swc_file, swc_to_arrays, get_skeleton_bounds, densify_skeleton


class SkeletonConstraint:
    """
    Computes distance from Gaussians to the nearest skeleton point.
    Used to penalize Gaussians that drift away from the neuron structure.
    """
    
    def __init__(self, swc_path: str, volume_size: tuple, device='cuda', 
                 densify_points=True, points_per_unit=2.0, max_skeleton_points=10000):
        """
        Args:
            swc_path: Path to SWC file
            volume_size: (D, H, W) volume dimensions
            device: torch device
            densify_points: Whether to densify skeleton for better coverage
            points_per_unit: Density of skeleton points for distance computation
            max_skeleton_points: Maximum skeleton points (subsample if needed to save memory)
        """
        self.device = device
        self.volume_size = volume_size
        D, H, W = volume_size
        
        # Parse SWC
        nodes = parse_swc_file(swc_path)
        positions, radii, parent_ids = swc_to_arrays(nodes)
        
        # Densify skeleton for better distance computation
        if densify_points:
            positions, radii = densify_skeleton(positions, radii, parent_ids, 
                                                points_per_unit=points_per_unit)
        
        # Subsample skeleton points if too many (to save GPU memory)
        if len(positions) > max_skeleton_points:
            indices = np.linspace(0, len(positions)-1, max_skeleton_points, dtype=int)
            positions = positions[indices]
            radii = radii[indices]
            print(f"  Subsampled skeleton to {max_skeleton_points} points (from {len(indices)*points_per_unit:.0f})")
        
        # Get bounds for normalization (same as swc_utils.py)
        min_bounds, max_bounds = get_skeleton_bounds(positions)
        extent = max_bounds - min_bounds
        extent = np.where(extent < 1e-6, 1.0, extent)
        
        # Convert SWC (X, Y, Z) to volume (D, H, W) = (Z, Y, X) coordinate order
        # This must match the coordinate transform in swc_utils.py
        positions_dhw = positions[:, [2, 1, 0]]  # Reorder: X,Y,Z -> Z,Y,X
        min_bounds_dhw = min_bounds[[2, 1, 0]]
        extent_dhw = extent[[2, 1, 0]]
        
        # Normalize to [0, 1] then apply margin (same as swc_utils.normalize_positions)
        margin = 0.05
        normalized = (positions_dhw - min_bounds_dhw) / extent_dhw
        skeleton_coords = normalized * (1 - 2 * margin) + margin  # Apply same margin as swc_utils
        
        self.skeleton_points = torch.tensor(skeleton_coords, dtype=torch.float32, device=device)
        self.skeleton_radii = torch.tensor(radii, dtype=torch.float32, device=device)
        
        # Normalize radii to volume scale
        max_dim = max(D, H, W)
        self.skeleton_radii = self.skeleton_radii / max_dim
        
        print(f"SkeletonConstraint: {len(self.skeleton_points)} skeleton points")
        print(f"  Position range: [{self.skeleton_points.min():.4f}, {self.skeleton_points.max():.4f}]")
    
    def compute_distance_loss(self, xyz: torch.Tensor, 
                               margin: float = 0.0,
                               use_radius: bool = True) -> torch.Tensor:
        """
        Compute minimum distance from each Gaussian to skeleton.
        
        Args:
            xyz: Gaussian positions [N, 3] normalized to [0, 1]
            margin: Allow Gaussians within this distance without penalty
            use_radius: Use skeleton radius to determine allowed distance
            
        Returns:
            loss: Mean distance loss (scalar)
        """
        N = xyz.shape[0]
        M = self.skeleton_points.shape[0]
        
        # Compute pairwise distances in chunks to save memory
        chunk_size = 500  # Small chunk size to avoid OOM
        min_distances = torch.zeros(N, device=self.device)
        
        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)
            xyz_chunk = xyz[i:end]  # [C, 3]
            
            # Distance to all skeleton points: [C, M]
            diff = xyz_chunk.unsqueeze(1) - self.skeleton_points.unsqueeze(0)
            dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)  # [C, M]
            
            if use_radius:
                # Subtract skeleton radius - points within radius have negative distance
                dist = dist - self.skeleton_radii.unsqueeze(0) * 2.0  # Allow 2x radius
            
            # Minimum distance to skeleton
            min_dist, _ = dist.min(dim=1)  # [C]
            min_distances[i:end] = min_dist
        
        # Apply margin - only penalize if beyond margin
        if margin > 0:
            min_distances = torch.relu(min_distances - margin)
        else:
            min_distances = torch.relu(min_distances)  # Only penalize positive distances
        
        # L2 loss on distances
        loss = (min_distances ** 2).mean()
        
        return loss
    
    def compute_soft_constraint(self, xyz: torch.Tensor, 
                                 sigma: float = 0.05) -> torch.Tensor:
        """
        Soft constraint using Gaussian proximity to skeleton.
        Higher values = closer to skeleton = less penalty.
        
        Args:
            xyz: Gaussian positions [N, 3]
            sigma: Softness of constraint
            
        Returns:
            loss: Mean soft constraint loss
        """
        N = xyz.shape[0]
        chunk_size = 10000
        
        proximity_scores = torch.zeros(N, device=self.device)
        
        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)
            xyz_chunk = xyz[i:end]
            
            # Distance to all skeleton points
            diff = xyz_chunk.unsqueeze(1) - self.skeleton_points.unsqueeze(0)
            dist_sq = (diff ** 2).sum(dim=-1)  # [C, M]
            
            # Gaussian proximity score
            scores = torch.exp(-dist_sq / (2 * sigma ** 2))  # [C, M]
            
            # Max proximity to any skeleton point
            max_scores, _ = scores.max(dim=1)  # [C]
            proximity_scores[i:end] = max_scores
        
        # Loss = 1 - proximity (want high proximity)
        loss = (1.0 - proximity_scores).mean()
        
        return loss


def skeleton_distance_loss(xyz: torch.Tensor, 
                           skeleton_constraint: SkeletonConstraint,
                           weight: float = 1.0,
                           use_soft: bool = False) -> torch.Tensor:
    """
    Convenience function for computing skeleton constraint loss.
    
    Args:
        xyz: Gaussian positions [N, 3]
        skeleton_constraint: SkeletonConstraint object
        weight: Loss weight
        use_soft: Use soft Gaussian constraint instead of hard distance
        
    Returns:
        Weighted skeleton loss
    """
    if use_soft:
        loss = skeleton_constraint.compute_soft_constraint(xyz)
    else:
        loss = skeleton_constraint.compute_distance_loss(xyz)
    
    return weight * loss
