"""
SWC Skeleton Utilities for 3DGR-CT

This module provides utilities for parsing SWC neuron skeleton files and
initializing Gaussian positions along neuron structures instead of from FBP images.

SWC Format:
    id, type, x, y, z, radius, parent_id
    
Benefits over FBP-based initialization:
- Gaussians placed only along actual neuron structures
- Avoids void regions (empty space)
- Uses skeleton radius for density allocation
- More efficient representation for sparse structures
"""

import numpy as np
import torch
from typing import Tuple, Optional, List, Dict


class SWCNode:
    """Represents a single node in the SWC skeleton."""
    def __init__(self, node_id: int, node_type: int, x: float, y: float, z: float, 
                 radius: float, parent_id: int):
        self.id = node_id
        self.type = node_type  # 1=soma, 2=axon, 3=dendrite, etc.
        self.x = x
        self.y = y
        self.z = z
        self.radius = radius
        self.parent_id = parent_id


def parse_swc_file(filepath: str) -> List[SWCNode]:
    """
    Parse an SWC file and return a list of SWCNode objects.
    
    Args:
        filepath: Path to the SWC file
        
    Returns:
        List of SWCNode objects representing the neuron skeleton
    """
    nodes = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line.startswith('#') or len(line) == 0:
                continue
            
            parts = line.split()
            if len(parts) >= 7:
                node = SWCNode(
                    node_id=int(parts[0]),
                    node_type=int(parts[1]),
                    x=float(parts[2]),
                    y=float(parts[3]),
                    z=float(parts[4]),
                    radius=float(parts[5]),
                    parent_id=int(parts[6])
                )
                nodes.append(node)
    
    return nodes


def swc_to_arrays(nodes: List[SWCNode]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert SWC nodes to numpy arrays.
    
    Args:
        nodes: List of SWCNode objects
        
    Returns:
        positions: (N, 3) array of xyz coordinates
        radii: (N,) array of radii
        parent_ids: (N,) array of parent node indices
    """
    n = len(nodes)
    positions = np.zeros((n, 3), dtype=np.float32)
    radii = np.zeros(n, dtype=np.float32)
    parent_ids = np.zeros(n, dtype=np.int32)
    
    for i, node in enumerate(nodes):
        positions[i] = [node.x, node.y, node.z]
        radii[i] = node.radius
        parent_ids[i] = node.parent_id
    
    return positions, radii, parent_ids


def get_skeleton_bounds(positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get the bounding box of the skeleton."""
    min_bounds = positions.min(axis=0)
    max_bounds = positions.max(axis=0)
    return min_bounds, max_bounds


def normalize_positions(positions: np.ndarray, 
                       target_size: Tuple[int, int, int] = None,
                       margin: float = 0.05) -> np.ndarray:
    """
    Normalize skeleton positions to [0, 1] range with optional margin.
    
    Args:
        positions: (N, 3) array of xyz coordinates
        target_size: Optional (D, H, W) volume size for alignment
        margin: Fraction of margin to leave on each side
        
    Returns:
        Normalized positions in [margin, 1-margin] range
    """
    min_bounds, max_bounds = get_skeleton_bounds(positions)
    extent = max_bounds - min_bounds
    
    # Avoid division by zero for flat dimensions
    extent = np.where(extent < 1e-6, 1.0, extent)
    
    # Normalize to [0, 1]
    normalized = (positions - min_bounds) / extent
    
    # Apply margin
    normalized = normalized * (1 - 2 * margin) + margin
    
    return normalized


def interpolate_skeleton_segment(p1: np.ndarray, p2: np.ndarray, r1: float, r2: float,
                                  num_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate points along a skeleton segment between two nodes.
    
    Args:
        p1, p2: Start and end positions
        r1, r2: Start and end radii
        num_points: Number of interpolated points
        
    Returns:
        positions: (num_points, 3) interpolated positions
        radii: (num_points,) interpolated radii
    """
    t = np.linspace(0, 1, num_points)[:, np.newaxis]
    positions = p1 + t * (p2 - p1)
    radii = r1 + t.squeeze() * (r2 - r1)
    return positions, radii


def densify_skeleton(positions: np.ndarray, radii: np.ndarray, parent_ids: np.ndarray,
                     points_per_unit: float = 10.0, 
                     min_points_per_segment: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Densify skeleton by interpolating between connected nodes.
    
    The density of points is based on:
    1. Distance between connected nodes
    2. Radius at each point (thicker regions get more points)
    
    Args:
        positions: (N, 3) node positions
        radii: (N,) node radii
        parent_ids: (N,) parent indices (-1 for root)
        points_per_unit: Base density of points per unit length
        min_points_per_segment: Minimum points to place per segment
        
    Returns:
        dense_positions: Interpolated positions
        dense_radii: Interpolated radii
    """
    all_positions = []
    all_radii = []
    
    # Build id to index mapping
    id_to_idx = {}
    for i, pid in enumerate(parent_ids):
        # Node IDs in SWC start from 1, so the index is i
        id_to_idx[i + 1] = i
    
    for i in range(len(positions)):
        parent_id = parent_ids[i]
        
        # Add the node itself
        all_positions.append(positions[i])
        all_radii.append(radii[i])
        
        # If this node has a parent, interpolate between them
        if parent_id > 0 and parent_id in id_to_idx:
            parent_idx = id_to_idx[parent_id]
            p1 = positions[parent_idx]
            p2 = positions[i]
            r1 = radii[parent_idx]
            r2 = radii[i]
            
            # Calculate segment length
            segment_length = np.linalg.norm(p2 - p1)
            
            # Determine number of interpolation points based on length and radius
            avg_radius = (r1 + r2) / 2
            # More points for longer segments and thicker regions
            num_interp = max(min_points_per_segment, 
                           int(segment_length * points_per_unit * (1 + avg_radius)))
            
            if num_interp > 2:
                # Interpolate (excluding endpoints which are added separately)
                interp_pos, interp_rad = interpolate_skeleton_segment(
                    p1, p2, r1, r2, num_interp
                )
                # Skip first and last (they are the actual nodes)
                all_positions.extend(interp_pos[1:-1])
                all_radii.extend(interp_rad[1:-1])
    
    return np.array(all_positions), np.array(all_radii)


def sample_gaussians_from_skeleton(positions: np.ndarray, radii: np.ndarray,
                                    num_samples: int,
                                    radius_based_density: bool = True,
                                    add_radial_jitter: bool = True,
                                    jitter_scale: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample Gaussian positions from the skeleton.
    
    Uses skeleton density and radius for allocation:
    - Regions with larger radius get more Gaussians
    - Points are placed along the neuron structure only
    
    Args:
        positions: (N, 3) skeleton positions (normalized to [0,1])
        radii: (N,) skeleton radii
        num_samples: Number of Gaussian positions to sample
        radius_based_density: If True, sample more points in regions with larger radius
        add_radial_jitter: If True, add small random offset perpendicular to skeleton
        jitter_scale: Scale of jitter relative to local radius
        
    Returns:
        sampled_positions: (num_samples, 3) Gaussian positions
        sampled_scales: (num_samples,) Initial Gaussian scales based on local radius
    """
    n = len(positions)
    
    if n == 0:
        raise ValueError("No skeleton positions provided")
    
    if radius_based_density:
        # Sample probability proportional to radius (thicker = more Gaussians)
        weights = radii / radii.sum()
    else:
        # Uniform sampling along skeleton
        weights = np.ones(n) / n
    
    # Sample indices based on weights
    indices = np.random.choice(n, size=num_samples, replace=True, p=weights)
    
    sampled_positions = positions[indices].copy()
    sampled_radii = radii[indices].copy()
    
    if add_radial_jitter:
        # Add small random offset to spread Gaussians around skeleton
        # This helps cover the volume of the neuron, not just the centerline
        jitter = np.random.randn(num_samples, 3) * sampled_radii[:, np.newaxis] * jitter_scale
        
        # Normalize jitter to be within bounds [0, 1]
        sampled_positions = sampled_positions + jitter * 0.01  # Scale down significantly
        sampled_positions = np.clip(sampled_positions, 0.001, 0.999)
    
    # Scale for Gaussians based on local radius
    # Smaller radius = smaller Gaussians for fine detail
    sampled_scales = sampled_radii / radii.max() * 0.02 + 0.005  # Base scale + radius contribution
    
    return sampled_positions.astype(np.float32), sampled_scales.astype(np.float32)


def create_gaussian_params_from_swc(swc_path: str,
                                     num_gaussians: int,
                                     volume_size: Tuple[int, int, int],
                                     ini_intensity: float = 0.1,
                                     densify: bool = True,
                                     points_per_unit: float = 5.0,
                                     radius_based_density: bool = True,
                                     device: str = "cuda") -> Dict[str, torch.Tensor]:
    """
    Create Gaussian model parameters directly from an SWC skeleton file.
    
    This is the main entry point for SWC-based Gaussian initialization.
    
    Benefits over FBP-based initialization:
    - Avoids void regions: Gaussians only placed along neuron
    - Uses skeleton radius for density allocation
    - More efficient for sparse neuron structures
    
    Args:
        swc_path: Path to SWC file
        num_gaussians: Number of Gaussians to create
        volume_size: (D, H, W) size of the target volume
        ini_intensity: Initial intensity value
        densify: Whether to densify skeleton by interpolation
        points_per_unit: Density for skeleton interpolation
        radius_based_density: Allocate more Gaussians to thicker regions
        device: PyTorch device
        
    Returns:
        Dictionary with keys: 'xyz', 'intensity', 'scaling', 'rotation'
        Note: xyz is in (D, H, W) = (Z, Y, X) order to match volume coordinates
    """
    # Parse SWC file
    nodes = parse_swc_file(swc_path)
    positions, radii, parent_ids = swc_to_arrays(nodes)
    # positions are in (X, Y, Z) order from SWC file
    
    print(f"Loaded SWC with {len(nodes)} nodes")
    print(f"SWC Position range (X,Y,Z): {positions.min(axis=0)} to {positions.max(axis=0)}")
    print(f"Radius range: {radii.min():.4f} to {radii.max():.4f}")
    
    # Convert SWC (X, Y, Z) to volume (D, H, W) = (Z, Y, X) coordinate order
    # This ensures Gaussians are placed correctly in the volume space
    positions_dhw = positions[:, [2, 1, 0]]  # Reorder: X,Y,Z -> Z,Y,X
    
    # Normalize positions to [0, 1] based on volume dimensions
    positions_dhw = normalize_positions(positions_dhw, target_size=volume_size)
    
    # Normalize radii relative to volume size
    max_dim = max(volume_size)
    radii = radii / max_dim * 10  # Scale radii appropriately
    
    # Optionally densify skeleton
    if densify:
        positions_dhw, radii = densify_skeleton(
            positions_dhw, radii, parent_ids,
            points_per_unit=points_per_unit
        )
        print(f"Densified skeleton to {len(positions_dhw)} points")
    
    # Sample Gaussian positions
    sampled_positions, sampled_scales = sample_gaussians_from_skeleton(
        positions_dhw, radii, num_gaussians,
        radius_based_density=radius_based_density,
        add_radial_jitter=True
    )
    
    print(f"Created {num_gaussians} Gaussians from skeleton")
    print(f"Gaussian positions in (D,H,W) order, range: [{sampled_positions.min():.4f}, {sampled_positions.max():.4f}]")
    
    # Convert to torch tensors - positions are now in (D, H, W) order
    xyz = torch.tensor(sampled_positions, dtype=torch.float32, device=device)
    
    # Initialize intensity based on local radius (thicker = more opaque)
    intensity_values = torch.full((num_gaussians, 1), ini_intensity, device=device)
    
    # Initialize scaling from sampled scales (3D isotropic)
    scaling = torch.tensor(sampled_scales, dtype=torch.float32, device=device)
    scaling = torch.log(scaling).unsqueeze(1).repeat(1, 3)
    
    # Initialize rotation to identity
    rotation = torch.zeros((num_gaussians, 4), device=device)
    rotation[:, 0] = 1  # w=1, x=y=z=0 for identity quaternion
    
    # Apply inverse sigmoid to intensity for proper initialization
    from gs_utils.general_utils import inverse_sigmoid
    intensity = inverse_sigmoid(intensity_values)
    
    return {
        'xyz': xyz,
        'intensity': intensity,
        'scaling': scaling,
        'rotation': rotation
    }


def visualize_skeleton_coverage(positions: np.ndarray, 
                                 gaussian_positions: np.ndarray,
                                 volume_size: Tuple[int, int, int],
                                 output_path: str = None):
    """
    Visualize how well Gaussians cover the skeleton (for debugging).
    
    Args:
        positions: Original skeleton positions
        gaussian_positions: Sampled Gaussian positions
        volume_size: Target volume size
        output_path: Optional path to save visualization
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 5))
        
        # Plot skeleton
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                   c='blue', s=1, alpha=0.5)
        ax1.set_title('Skeleton Points')
        
        # Plot Gaussian positions
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(gaussian_positions[:, 0], gaussian_positions[:, 1], 
                   gaussian_positions[:, 2], c='red', s=1, alpha=0.3)
        ax2.set_title('Gaussian Positions')
        
        if output_path:
            plt.savefig(output_path, dpi=150)
        plt.close()
        
    except ImportError:
        print("matplotlib not available for visualization")
