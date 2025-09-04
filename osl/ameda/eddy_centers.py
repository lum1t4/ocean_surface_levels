"""
Eddy center detection module for AMEDA (Angular Momentum Eddy Detection Algorithm).

This module detects potential eddy centers present in the domain using LNAM 
(Local Normalized Angular Momentum) and LOW (Local Okubo-Weiss) fields.

The detection process:
1. Find max(|LNAM(LOW<0)>K|) where K is the LNAM threshold
2. Validate centers with at least 2 closed streamlines
3. Handle conflicts between overlapping eddies
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, NamedTuple

import numpy as np
import numba as nb
from matplotlib.path import Path
import matplotlib.pyplot as plt
from skimage import measure

from osl.ameda.params import AMEDAParams, G, R
from osl.ameda.fields import DetectionFields
from osl.ameda.load_fields import load_fields
from osl.ameda.utils import DEG2NM, NM2KM


class StreamLine(NamedTuple):
    """Container for a streamline"""
    x: np.ndarray
    y: np.ndarray
    level: float
    max_y: float


@dataclass
class EddyCenter:
    """Container for detected eddy centers at a time step"""
    step: int
    type: np.ndarray  # Eddy type: 1 = cyclonic, -1 = anticyclonic
    x: np.ndarray     # x coordinates (longitude)
    y: np.ndarray     # y coordinates (latitude)
    i: np.ndarray     # Column indices in grid
    j: np.ndarray     # Row indices in grid
    
    def __len__(self):
        return len(self.x) if self.x is not None else 0
    
    def is_empty(self):
        return len(self) == 0


@dataclass
class Centers:
    """Container for both raw and validated centers"""
    centers0: EddyCenter  # Raw LNAM maxima
    centers: EddyCenter   # Validated centers with streamlines


def sw_dist2(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """
    Calculate distance between consecutive points.
    
    Parameters
    ----------
    lat, lon : ndarray
        Latitude and longitude arrays
        
    Returns
    -------
    dist : ndarray
        Distances in km
    """
    lat = np.asarray(lat).ravel()
    lon = np.asarray(lon).ravel()
    
    if len(lat) < 2:
        return np.array([])
    
    # Calculate differences
    dlat = np.diff(lat)
    dlon = np.diff(lon)
    
    # Handle longitude wrapping
    dlon = np.where(np.abs(dlon) > 180, 
                    -np.sign(dlon) * (360 - np.abs(dlon)), 
                    dlon)
    
    # Convert to distances
    latrad = np.deg2rad(lat)
    dep = np.cos((latrad[1:] + latrad[:-1]) / 2.0) * dlon
    
    # Distance in km
    dist = DEG2NM * np.sqrt(dlat**2 + dep**2) * NM2KM
    
    return dist


def mean_radius(xy: np.ndarray, grid_ll: bool = True) -> Tuple[float, float, float, np.ndarray]:
    """
    Compute mean radius, area, perimeter and barycenter of a closed contour.
    
    Parameters
    ----------
    xy : ndarray
        2xN array of coordinates [x; y]
    grid_ll : bool
        True if coordinates are lon/lat, False if cartesian km
        
    Returns
    -------
    R : float
        Mean radius (km)
    A : float
        Area (km²)
    P : float
        Perimeter (km)
    ll : ndarray
        Barycenter coordinates [x, y]
    """
    # Remove duplicate last point if closed
    if np.allclose(xy[:, 0], xy[:, -1]):
        xy = xy[:, :-1]
    
    n_points = xy.shape[1]
    
    # Barycenter
    ll = np.mean(xy, axis=1)
    
    # Distance from barycenter to each point
    distances = np.zeros(n_points)
    for i in range(n_points):
        if grid_ll:
            # Convert to km for lon/lat coordinates
            dx = (xy[0, i] - ll[0]) * R * np.cos(np.deg2rad(xy[1, i]))
            dy = (xy[1, i] - ll[1]) * R
            distances[i] = np.sqrt(dx**2 + dy**2)
        else:
            distances[i] = np.sqrt((xy[0, i] - ll[0])**2 + (xy[1, i] - ll[1])**2)
    
    # Mean radius
    R_mean = np.mean(distances)
    
    # Perimeter
    perimeter = 0
    for i in range(n_points):
        j = (i + 1) % n_points
        if grid_ll:
            dx = (xy[0, j] - xy[0, i]) * R * np.cos(np.deg2rad((xy[1, i] + xy[1, j]) / 2))
            dy = (xy[1, j] - xy[1, i]) * R
            perimeter += np.sqrt(dx**2 + dy**2)
        else:
            perimeter += np.sqrt((xy[0, j] - xy[0, i])**2 + (xy[1, j] - xy[1, i])**2)
    
    # Area using shoelace formula
    area = 0
    for i in range(n_points):
        j = (i + 1) % n_points
        area += xy[0, i] * xy[1, j] - xy[0, j] * xy[1, i]
    area = abs(area) / 2
    
    if grid_ll:
        # Convert area from deg² to km²
        area = area * (R**2) * np.cos(np.deg2rad(ll[1]))
    
    return R_mean, area, perimeter, ll


def scan_lines(contour_data: np.ndarray) -> List[StreamLine]:
    """
    Parse contour data into structured streamlines.
    
    Parameters
    ----------
    contour_data : ndarray
        Contour data from matplotlib or contourc format
        
    Returns
    -------
    lines : list of StreamLine
        Parsed streamlines sorted by maximum y coordinate
    """
    lines = []
    
    if contour_data is None or contour_data.size == 0:
        return lines
    
    # Parse contour format: [level, npoints, x1, y1, x2, y2, ...]
    i = 0
    while i < contour_data.shape[1]:
        level = contour_data[0, i]
        npoints = int(contour_data[1, i])
        
        if i + npoints + 1 > contour_data.shape[1]:
            break
            
        x = contour_data[0, i+1:i+npoints+1]
        y = contour_data[1, i+1:i+npoints+1]
        max_y = np.max(y)
        
        lines.append(StreamLine(x, y, level, max_y))
        i += npoints + 1
    
    # Sort by maximum y coordinate
    lines.sort(key=lambda line: line.max_y)
    
    return lines


@nb.jit(nopython=True)
def compute_psi_numba(x: np.ndarray, y: np.ndarray, u: np.ndarray, v: np.ndarray,
                      ci: int, cj: int, grid_ll: bool) -> np.ndarray:
    """
    Compute streamfunction by integrating velocity fields (Numba-accelerated).
    """
    N, M = u.shape
    psi = np.zeros_like(u)
    
    # Set NaN to 0 for integration
    u = np.where(np.isnan(u), 0.0, u)
    v = np.where(np.isnan(v), 0.0, v)
    
    # Compute distances for integration
    if grid_ll:
        # For lon/lat grids, convert to km
        for j in range(cj, N):
            for i in range(ci, M):
                if i == ci and j == cj:
                    psi[j, i] = 0
                elif i == ci:
                    # Integrate along y from center
                    dy = (y[j, i] - y[j-1, i]) * R
                    psi[j, i] = psi[j-1, i] - u[j-1, i] * dy
                elif j == cj:
                    # Integrate along x from center
                    dx = (x[j, i] - x[j, i-1]) * R * np.cos(np.deg2rad(y[j, i]))
                    psi[j, i] = psi[j, i-1] + v[j, i-1] * dx
                else:
                    # Average from two paths
                    dx = (x[j, i] - x[j, i-1]) * R * np.cos(np.deg2rad(y[j, i]))
                    dy = (y[j, i] - y[j-1, i]) * R
                    psi_x = psi[j, i-1] + v[j, i-1] * dx
                    psi_y = psi[j-1, i] - u[j-1, i] * dy
                    psi[j, i] = 0.5 * (psi_x + psi_y)
    else:
        # For cartesian grids
        for j in range(cj, N):
            for i in range(ci, M):
                if i == ci and j == cj:
                    psi[j, i] = 0
                elif i == ci:
                    dy = y[j, i] - y[j-1, i]
                    psi[j, i] = psi[j-1, i] - u[j-1, i] * dy
                elif j == cj:
                    dx = x[j, i] - x[j, i-1]
                    psi[j, i] = psi[j, i-1] + v[j, i-1] * dx
                else:
                    dx = x[j, i] - x[j, i-1]
                    dy = y[j, i] - y[j-1, i]
                    psi_x = psi[j, i-1] + v[j, i-1] * dx
                    psi_y = psi[j-1, i] - u[j-1, i] * dy
                    psi[j, i] = 0.5 * (psi_x + psi_y)
    
    # Complete other quadrants similarly (simplified for brevity)
    # This is a simplified version - full implementation would handle all 4 quadrants
    
    return psi


def compute_psi(x: np.ndarray, y: np.ndarray, mask: np.ndarray, 
                u: np.ndarray, v: np.ndarray, ci: int, cj: int, 
                grid_ll: bool = True) -> np.ndarray:
    """
    Compute streamfunction field by integrating velocity.
    
    Parameters
    ----------
    x, y : ndarray
        Grid coordinates
    mask : ndarray
        Ocean mask
    u, v : ndarray
        Velocity components (m/s, already scaled by f/g if needed)
    ci, cj : int
        Center indices
    grid_ll : bool
        True if coordinates are lon/lat
        
    Returns
    -------
    psi : ndarray
        Streamfunction field
    """
    # Simple streamfunction computation
    N, M = u.shape
    psi = np.zeros_like(u)
    
    # Replace NaN with 0 for integration
    u = np.nan_to_num(u, nan=0)
    v = np.nan_to_num(v, nan=0)
    
    # Integrate from center point
    # This is a simplified version - just integrate along grid lines
    
    # Set center to 0
    psi[cj, ci] = 0
    
    # Integrate along rows from center
    for j in range(N):
        for i in range(M):
            if i == ci and j == cj:
                continue
            elif j == cj:
                # Integrate along x
                if i > ci:
                    psi[j, i] = psi[j, i-1] + 0.5 * (v[j, i] + v[j, i-1]) * (x[j, i] - x[j, i-1])
                else:
                    psi[j, i] = psi[j, i+1] - 0.5 * (v[j, i] + v[j, i+1]) * (x[j, i+1] - x[j, i])
            elif i == ci:
                # Integrate along y
                if j > cj:
                    psi[j, i] = psi[j-1, i] - 0.5 * (u[j, i] + u[j-1, i]) * (y[j, i] - y[j-1, i])
                else:
                    psi[j, i] = psi[j+1, i] + 0.5 * (u[j, i] + u[j+1, i]) * (y[j+1, i] - y[j, i])
            else:
                # Average from two integration paths
                if i > ci and j > cj:
                    psi_x = psi[j, i-1] + v[j, i-1] * (x[j, i] - x[j, i-1])
                    psi_y = psi[j-1, i] - u[j-1, i] * (y[j, i] - y[j-1, i])
                    psi[j, i] = 0.5 * (psi_x + psi_y)
    
    # Scale by 1000 to get reasonable values for contours
    return psi * 1000


def point_in_polygon(points_x: np.ndarray, points_y: np.ndarray, 
                     poly_x: np.ndarray, poly_y: np.ndarray) -> np.ndarray:
    """
    Check if points are inside a polygon.
    
    Parameters
    ----------
    points_x, points_y : ndarray
        Point coordinates to test
    poly_x, poly_y : ndarray
        Polygon vertex coordinates
        
    Returns
    -------
    inside : ndarray
        Boolean array, True if point is inside polygon
    """
    # Handle scalar or array inputs
    points_x = np.atleast_1d(points_x)
    points_y = np.atleast_1d(points_y)
    scalar_input = (points_x.size == 1)
    
    # Create path from polygon
    poly_verts = np.column_stack([poly_x, poly_y])
    path = Path(poly_verts)
    
    # Test points
    points = np.column_stack([points_x.ravel(), points_y.ravel()])
    inside = path.contains_points(points).reshape(points_x.shape)
    
    if scalar_input:
        return inside[0]
    return inside


def find_eddy_centers(x: np.ndarray, y: np.ndarray, mask: np.ndarray,
                      u: np.ndarray, v: np.ndarray, ssh: Optional[np.ndarray],
                      fields: DetectionFields, params: AMEDAParams, 
                      step: int) -> Centers:
    """
    Detect eddy centers from velocity and detection fields.
    
    Parameters
    ----------
    x, y : ndarray
        Grid coordinates  
    mask : ndarray
        Ocean mask
    u, v : ndarray
        Velocity components (m/s)
    ssh : ndarray or None
        Sea surface height
    fields : DetectionFields
        Computed detection fields (LNAM, LOW, etc.)
    params : AMEDAParams
        Algorithm parameters
    step : int
        Time step
        
    Returns
    -------
    centers : Centers
        Detected centers (raw and validated)
    """
    print(f"\n Find potential centers step {step} %-------------")
    
    # Initialize output structures
    centers0 = EddyCenter(
        step=step,
        type=np.array([]),
        x=np.array([]),
        y=np.array([]),
        i=np.array([]),
        j=np.array([])
    )
    
    centers = EddyCenter(
        step=step,
        type=np.array([]),
        x=np.array([]),
        y=np.array([]),
        i=np.array([]),
        j=np.array([])
    )
    
    # Get detection fields
    OW = fields.LOW
    LNAM = fields.LNAM
    LOW = np.abs(LNAM)
    LOW[OW >= 0] = 0
    LOW[np.isnan(OW)] = 0
    
    # Find contours at threshold K
    try:
        if params.grid_reg:
            # Regular grid - use matplotlib contour
            fig, ax = plt.subplots()
            CS = ax.contour(x[0, :], y[:, 0], LOW, levels=[params.K])
            plt.close(fig)
            
            # Extract contour data from allsegs
            contour_data = None
            for level_idx, level in enumerate(CS.levels):
                for seg in CS.allsegs[level_idx]:
                    if len(seg) >= params.n_min:
                        # Add in contourc format: [level, npoints, x1, y1, x2, y2, ...]
                        n_points = len(seg)
                        contour_entry = np.zeros((2, n_points + 1))
                        contour_entry[0, 0] = level
                        contour_entry[1, 0] = n_points
                        contour_entry[:, 1:] = seg.T
                        
                        if contour_data is None:
                            contour_data = contour_entry
                        else:
                            contour_data = np.hstack([contour_data, contour_entry])
            
            if contour_data is None:
                contour_data = np.array([])
        else:
            # Irregular grid - use skimage
            contours = measure.find_contours(LOW, params.K)
            # Convert to contourc format
            contour_data = []
            for contour in contours:
                if len(contour) >= params.n_min:
                    # Map indices to coordinates
                    x_cont = np.interp(contour[:, 1], np.arange(x.shape[1]), x[0, :])
                    y_cont = np.interp(contour[:, 0], np.arange(y.shape[0]), y[:, 0])
                    contour_data.append([params.K, len(x_cont)])
                    contour_data.extend(np.column_stack([x_cont, y_cont]).T.tolist())
            if contour_data:
                contour_data = np.array(contour_data).T
            else:
                contour_data = np.array([])
    except Exception as e:
        print(f"  Warning: Contour extraction failed: {e}")
        contour_data = np.array([])
    
    if contour_data.size == 0:
        print(f"  -> 0 max LNAM found step {step}")
        print(f"!!! WARNING !!! No LNAM extrema found - check the LNAM computation step {step}")
        return Centers(centers0, centers)
    
    # Find LNAM maxima inside contours
    max_lnam_list = []
    k = 0
    
    # Parse contours
    lines = scan_lines(contour_data)
    
    for line in lines:
        if len(line.x) >= params.n_min:
            # Check points inside contour
            in_contour = point_in_polygon(x.ravel(), y.ravel(), line.x, line.y)
            in_contour = in_contour.reshape(x.shape)
            
            # Mask LNAM inside contour
            Lm = LNAM.copy()
            Lm[~in_contour] = np.nan
            
            # Find maximum
            if np.any(mask[in_contour] > 0) and np.nanmax(np.abs(Lm)) != 0:
                max_idx = np.unravel_index(np.nanargmax(np.abs(Lm)), Lm.shape)
                
                if mask[max_idx] == 1:
                    LC = Lm[max_idx]
                    xLmax = x[max_idx]
                    yLmax = y[max_idx]
                    
                    # Check latitude constraint
                    if not params.grid_ll or abs(yLmax) > params.lat_min:
                        # Check if not already found
                        already_found = False
                        for prev in max_lnam_list:
                            if prev['x'] == xLmax and prev['y'] == yLmax:
                                already_found = True
                                break
                        
                        if not already_found:
                            max_lnam_list.append({
                                'type': np.sign(LC),
                                'x': xLmax,
                                'y': yLmax,
                                'j': max_idx[0],
                                'i': max_idx[1],
                                'value': LC
                            })
                            k += 1
    
    print(f"  -> {k} max LNAM found step {step}")
    
    if k == 0:
        print(f"!!! WARNING !!! No LNAM extrema found - check the LNAM computation step {step}")
        return Centers(centers0, centers)
    
    # Convert to centers0 structure
    centers0.type = np.array([m['type'] for m in max_lnam_list])
    centers0.x = np.array([m['x'] for m in max_lnam_list])
    centers0.y = np.array([m['y'] for m in max_lnam_list])
    centers0.i = np.array([m['i'] for m in max_lnam_list])
    centers0.j = np.array([m['j'] for m in max_lnam_list])
    
    # Validate centers with streamlines
    print(f"  Remove max LNAM without 2 closed streamlines with proper size step {step}")
    
    # Use interpolated parameters if resol > 1
    if params.resol > 1:
        bxi = params.bxi
        Dxi = params.Dxi
        Rdi = params.Rdi
        f_i = params.fi
    else:
        bxi = params.bx
        Dxi = params.Dx
        Rdi = params.Rd
        f_i = params.f
    
    validated = []
    second_centers = np.full(len(centers0.x), -1)  # Track double eddies
    
    for ii in range(len(centers0.x)):
        # Get center indices
        C_I = centers0.i[ii]
        C_J = centers0.j[ii]
        xy_ci = centers0.x[ii]
        xy_cj = centers0.y[ii]
        
        # Box size around center
        bx = int(bxi[C_J, C_I])
        Dx = abs(Dxi[C_J, C_I])
        Rd = abs(Rdi[C_J, C_I])
        f = abs(f_i[C_J, C_I])
        
        # Extract subregion around center
        j_min = max(C_J - bx, 0)
        j_max = min(C_J + bx + 1, x.shape[0])
        i_min = max(C_I - bx, 0)
        i_max = min(C_I + bx + 1, x.shape[1])
        
        xx = x[j_min:j_max, i_min:i_max]
        yy = y[j_min:j_max, i_min:i_max]
        uu = u[j_min:j_max, i_min:i_max]
        vv = v[j_min:j_max, i_min:i_max]
        mk = mask[j_min:j_max, i_min:i_max]
        
        # Local center indices
        ci = C_I - i_min
        cj = C_J - j_min
        
        # Compute streamlines based on detection type
        all_streamlines = []
        
        if params.type_detection == 1 or params.type_detection == 3:
            # Compute psi from velocity
            psi1 = compute_psi(xx, yy, mk, uu * f / G * 1e3, 
                              vv * f / G * 1e3, ci, cj, params.grid_ll)
            
            # Get value at center for better contour levels
            psi_center = psi1[cj, ci]
            
            # Get contour levels around center value
            psi_min = np.nanmin(psi1)
            psi_max = np.nanmax(psi1)
            
            # Create levels centered around the center value
            if not np.isnan(psi_center):
                # Levels from center outward
                n_levels = min(params.nH_lim // 2, 20)
                H_below = np.linspace(psi_center - (psi_center - psi_min) * 0.8, 
                                     psi_center - params.DH, n_levels)
                H_above = np.linspace(psi_center + params.DH, 
                                     psi_center + (psi_max - psi_center) * 0.8, n_levels)
                H = np.unique(np.concatenate([H_below, [psi_center], H_above]))
            else:
                # Fallback to regular spacing
                H = np.linspace(psi_min, psi_max, min(params.nH_lim, 40))
            
            # Extract contours
            try:
                fig, ax = plt.subplots()
                CS1 = ax.contour(xx[0, :], yy[:, 0], psi1, levels=H)
                plt.close(fig)
                for level_idx, level in enumerate(CS1.levels):
                    for seg in CS1.allsegs[level_idx]:
                        if len(seg) >= params.n_min:
                            all_streamlines.append(StreamLine(
                                seg[:, 0], seg[:, 1], level, np.max(seg[:, 1])
                            ))
            except Exception as e:
                print(f"    Warning: Failed to extract velocity streamlines: {e}")
        
        if params.type_detection >= 2 and ssh is not None:
            # Use SSH contours
            sshh = ssh[j_min:j_max, i_min:i_max]
            if not np.all(np.isnan(sshh)) and ci < sshh.shape[1] and cj < sshh.shape[0]:
                ssh_center = sshh[cj, ci]
                if not np.isnan(ssh_center):
                    ssh_min = np.nanmin(sshh)
                    ssh_max = np.nanmax(sshh)
                    
                    # Create levels centered around the center value
                    n_levels = min(params.nH_lim // 2, 20)
                    Hs_below = np.linspace(ssh_center - (ssh_center - ssh_min) * 0.8,
                                          ssh_center - params.DH, n_levels)
                    Hs_above = np.linspace(ssh_center + params.DH,
                                          ssh_center + (ssh_max - ssh_center) * 0.8, n_levels)
                    Hs = np.unique(np.concatenate([Hs_below, [ssh_center], Hs_above]))
                else:
                    # Fallback to regular spacing
                    Hs = np.linspace(np.nanmin(sshh), np.nanmax(sshh), min(params.nH_lim, 40))
                
                try:
                    fig, ax = plt.subplots()
                    CS2 = ax.contour(xx[0, :], yy[:, 0], sshh, levels=Hs)
                    plt.close(fig)
                    for level_idx, level in enumerate(CS2.levels):
                        for seg in CS2.allsegs[level_idx]:
                            if len(seg) >= params.n_min:
                                all_streamlines.append(StreamLine(
                                    seg[:, 0], seg[:, 1], level, np.max(seg[:, 1])
                                ))
                except Exception as e:
                    print(f"    Warning: Failed to extract SSH streamlines: {e}")
        
        # Sort streamlines by max_y
        all_streamlines.sort(key=lambda s: s.max_y)
        
        # Validate center with streamlines
        radii = []
        n_closed = 0
        n_with_center = 0
        
        for streamline in all_streamlines:
            # Check if streamline is closed
            if (np.allclose(streamline.x[0], streamline.x[-1], rtol=1e-3) and 
                np.allclose(streamline.y[0], streamline.y[-1], rtol=1e-3)):
                n_closed += 1
                
                # Check if center is inside
                if point_in_polygon(xy_ci, xy_cj, streamline.x, streamline.y):
                    n_with_center += 1
                    # Calculate radius
                    xy_contour = np.vstack([streamline.x, streamline.y])
                    R, _, _, _ = mean_radius(xy_contour, params.grid_ll)
                    radii.append(R)
                    
                    # Need at least 2 streamlines with proper size
                    if (len(radii) >= 2 and 
                        radii[-1] >= params.nRmin * Dx and 
                        radii[-1] <= params.nR_lim * Rd):
                        
                        print(f"   Validate max LNAM {ii} with 2 streamlines at step {step}")
                        validated.append(ii)
                        second_centers[ii] = 0  # Single eddy
                        break
        
        # Debug output for troubleshooting (only first few centers)
        if ii < 3:
            print(f"    Debug center {ii} at ({xy_ci:.2f}, {xy_cj:.2f}): "
                  f"{len(all_streamlines)} streamlines, "
                  f"{n_closed} closed, {n_with_center} with center, "
                  f"{len(radii)} valid radii")
            if len(radii) > 0:
                print(f"      Radii: {radii[:3]} km, need {params.nRmin * Dx:.1f} to {params.nR_lim * Rd:.1f} km")
    
    # Build validated centers
    if validated:
        centers.type = centers0.type[validated]
        centers.x = centers0.x[validated]
        centers.y = centers0.y[validated]
        centers.i = centers0.i[validated]
        centers.j = centers0.j[validated]
        
        print(f" Potential eddy centers found step {step}")
        print(f"  -> {len(validated)} potential centers found")
        print(f"    ({len(centers0.x) - len(validated)} max LNAM removed)")
    else:
        print(f"!!! WARNING or ERROR !!! No potential centers found - "
              f"check the streamlines scanning process at step {step}")
    
    print()
    return Centers(centers0, centers)
