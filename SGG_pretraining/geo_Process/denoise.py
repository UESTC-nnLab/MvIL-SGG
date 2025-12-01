import numpy as np
import time
import collections

# Optional imports with checks
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Warning: Open3D not found. Statistical Outlier Removal step will be skipped.")

try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: Scikit-learn not found. DBSCAN clustering step will be skipped.")

def filter_instances_combined(
    points: np.ndarray,
    instance_ids: np.ndarray,
    # --- SOR Parameters ---
    enable_sor: bool = True, # Flag to enable/disable SOR step
    sor_nb_neighbors: int = 50,
    sor_std_ratio: float = 1.0,
    # --- DBSCAN Largest Cluster Keeping Parameters ---
    enable_dbscan_cleanup: bool = True, # Flag to enable/disable DBSCAN step
    dbscan_eps: float = 0.3,           # VERY sensitive parameter - needs tuning! Defines max distance between points in a cluster.
    dbscan_min_samples: int = 30,      # Min points required to form a dense core region in DBSCAN.
    # --- General Parameters ---
    min_points_for_sor: int = 15,      # Min points needed *before* attempting SOR
    min_points_for_dbscan: int = 15,   # Min points needed *after* SOR to attempt DBSCAN (should be >= dbscan_min_samples)
    verbose: bool = True
    ) :
    """
    Filters noise from a point cloud on a per-instance basis using a two-stage approach:
    1. (Optional) Statistical Outlier Removal (SOR) using Open3D.
    2. (Optional) DBSCAN clustering to keep only the largest spatially connected component.

    Args:
        points (np.ndarray): Input points (Nx3).
        instance_ids (np.ndarray): Corresponding instance IDs (N,).
        enable_sor (bool): Whether to perform the initial SOR filtering.
        sor_nb_neighbors (int): SOR parameter: Number of neighbors.
        sor_std_ratio (float): SOR parameter: Standard deviation ratio threshold.
        enable_dbscan_cleanup (bool): Whether to perform DBSCAN clustering to keep the largest component.
        dbscan_eps (float): DBSCAN parameter: Max distance between samples for neighborhood. Crucial for defining cluster connectivity.
        dbscan_min_samples (int): DBSCAN parameter: Min number of samples in a neighborhood for a core point.
        min_points_for_sor (int): Min points in an instance to attempt SOR.
        min_points_for_dbscan (int): Min points remaining *after* SOR to attempt DBSCAN cleanup.
        verbose (bool): Print progress and summary.

    Returns:
        tuple[np.ndarray, np.ndarray]: Filtered points and corresponding IDs.
    """
    # --- Input Validation and Library Checks ---
    if points.shape[0] != instance_ids.shape[0]:
        raise ValueError("Points and instance_ids must have the same number of elements.")
    if points.shape[0] == 0: return np.empty((0, 3)), np.empty((0,))

    if enable_sor and not OPEN3D_AVAILABLE:
        print("Warning: Open3D not available, disabling SOR step.")
        enable_sor = False
    if enable_dbscan_cleanup and not SKLEARN_AVAILABLE:
        print("Warning: Scikit-learn not available, disabling DBSCAN cleanup step.")
        enable_dbscan_cleanup = False

    start_time = time.time()
    unique_ids = np.unique(instance_ids)
    if verbose:
        print(f"Found {len(unique_ids)} unique instances.")
        print(f"SOR enabled: {enable_sor}, DBSCAN Cleanup enabled: {enable_dbscan_cleanup}")
        if enable_sor: print(f"  SOR params: neighbors={sor_nb_neighbors}, std_ratio={sor_std_ratio}, min_pts={min_points_for_sor}")
        if enable_dbscan_cleanup: print(f"  DBSCAN params: eps={dbscan_eps}, min_samples={dbscan_min_samples}, min_pts={min_points_for_dbscan}")

    filtered_points_list = []
    filtered_ids_list = []
    total_pts_initial = points.shape[0]
    total_pts_after_sor = 0
    total_pts_after_dbscan = 0


    # --- Process each instance ---
    for i, current_id in enumerate(unique_ids):
        instance_mask = (instance_ids == current_id)
        instance_points = points[instance_mask]
        num_initial_instance_points = len(instance_points)

        if verbose and (i + 1) % 50 == 0:
             print(f"  Processing instance {i+1}/{len(unique_ids)} (ID: {current_id})...")

        # Points after SOR (initially same as input)
        sor_filtered_points = instance_points
        num_after_sor = num_initial_instance_points

        # --- Stage 1: Statistical Outlier Removal (Optional) ---
        if enable_sor and num_initial_instance_points >= min_points_for_sor:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(instance_points)
            try:
                cl, sor_inlier_indices = pcd.remove_statistical_outlier(
                    nb_neighbors=sor_nb_neighbors, std_ratio=sor_std_ratio
                )
                sor_filtered_points = instance_points[sor_inlier_indices]
                num_after_sor = len(sor_filtered_points)
                if verbose and (num_initial_instance_points - num_after_sor > 0):
                    print(f"  Instance {current_id}: SOR removed {num_initial_instance_points - num_after_sor} points ({num_initial_instance_points} -> {num_after_sor}).")
            except Exception as e:
                 print(f"  Warning: SOR failed for instance {current_id}: {e}. Keeping original points for this stage.")
                 sor_filtered_points = instance_points # Keep original if SOR fails
                 num_after_sor = len(sor_filtered_points)
        elif enable_sor: # SOR enabled but too few points
             if verbose: print(f"  Instance {current_id}: Only {num_initial_instance_points} points (<{min_points_for_sor}), skipping SOR.")

        total_pts_after_sor += num_after_sor

        # Points after DBSCAN (initially same as after SOR)
        dbscan_filtered_points = sor_filtered_points
        num_after_dbscan = num_after_sor

        # --- Stage 2: DBSCAN Largest Cluster Cleanup (Optional) ---
        if enable_dbscan_cleanup and num_after_sor >= min_points_for_dbscan:
            if num_after_sor == 0: # Handle empty input to DBSCAN
                 if verbose: print(f"  Instance {current_id}: No points remaining after SOR, skipping DBSCAN.")
                 dbscan_filtered_points = sor_filtered_points
                 num_after_dbscan = 0
            else:
                try:
                    # Apply DBSCAN
                    db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, n_jobs=-1).fit(sor_filtered_points)
                    labels = db.labels_ # Cluster labels: -1 for noise, 0+ for clusters

                    # Find the largest valid cluster (label >= 0)
                    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)

                    if len(unique_labels) > 0: # If any valid clusters were found
                        largest_cluster_label = unique_labels[np.argmax(counts)]
                        largest_cluster_mask = (labels == largest_cluster_label)
                        dbscan_filtered_points = sor_filtered_points[largest_cluster_mask]
                        num_after_dbscan = len(dbscan_filtered_points)
                        num_removed_dbscan = num_after_sor - num_after_dbscan
                        if verbose and num_removed_dbscan > 0:
                             print(f"  Instance {current_id}: DBSCAN kept largest cluster ({num_after_dbscan} pts), removed {num_removed_dbscan} smaller/noise points.")
                    else: # DBSCAN labeled everything as noise (-1) or no points passed SOR
                        if verbose: print(f"  Instance {current_id}: DBSCAN found no valid clusters. Discarding all {num_after_sor} points.")
                        dbscan_filtered_points = np.empty((0, 3), dtype=points.dtype) # Discard all
                        num_after_dbscan = 0

                except Exception as e:
                    print(f"  Warning: DBSCAN failed for instance {current_id}: {e}. Keeping SOR results for this instance.")
                    dbscan_filtered_points = sor_filtered_points # Revert to SOR result on error
                    num_after_dbscan = len(dbscan_filtered_points)

        elif enable_dbscan_cleanup: # DBSCAN enabled but too few points after SOR
             if verbose: print(f"  Instance {current_id}: Only {num_after_sor} points remaining (<{min_points_for_dbscan}), skipping DBSCAN cleanup.")
             dbscan_filtered_points = sor_filtered_points # Keep SOR results
             num_after_dbscan = len(dbscan_filtered_points)

        total_pts_after_dbscan += num_after_dbscan

        # --- Collect Results for this Instance ---
        final_instance_points = dbscan_filtered_points
        num_final_points = len(final_instance_points)

        if num_final_points > 0:
            filtered_points_list.append(final_instance_points)
            filtered_ids_list.append(np.full(num_final_points, current_id, dtype=instance_ids.dtype))


    # --- Combine final results ---
    if not filtered_points_list:
        print("Warning: No points remaining after all filtering stages.")
        final_points = np.empty((0, 3), dtype=points.dtype)
        final_ids = np.empty((0,), dtype=instance_ids.dtype)
    else:
        final_points = np.vstack(filtered_points_list)
        final_ids = np.concatenate(filtered_ids_list)

    end_time = time.time()
    if verbose:
        print("-" * 30)
        print("Filtering Summary:")
        print(f"  Processed {len(unique_ids)} unique instances.")
        print(f"  Initial total points: {total_pts_initial}")
        if enable_sor: print(f"  Points remaining after SOR stage (approx): {total_pts_after_sor}")
        if enable_dbscan_cleanup: print(f"  Points remaining after DBSCAN stage (approx): {total_pts_after_dbscan}")
        print(f"  Final total points: {final_points.shape[0]}")
        print(f"  Total points removed: {total_pts_initial - final_points.shape[0]}")
        print(f"  Processing time: {end_time - start_time:.2f} seconds")
        print("-" * 30)

    return final_points, final_ids


# --- Example Usage ---
if __name__ == "__main__":
    print("Generating sample data with disconnected components...")
    np.random.seed(42)
    num_instances = 3
    points_per_component = 500
    noise_points = 50
    num_components_per_instance = [1, 2, 3] # Instance 0: 1 component, Inst 1: 2, Inst 2: 3
    component_separation = 2.0 # Distance between components of the same instance

    all_points = []
    all_ids = []
    instance_base_centers = (np.random.rand(num_instances, 3) - 0.5) * 10

    for inst_id in range(num_instances):
        num_components = num_components_per_instance[inst_id]
        base_center = instance_base_centers[inst_id]
        for comp_idx in range(num_components):
            # Generate component points (e.g., sphere patch)
            phi = np.random.uniform(0, np.pi / 2, points_per_component) # Hemisphere
            theta = np.random.uniform(0, 2 * np.pi, points_per_component)
            radius = 0.5
            x = radius * np.sin(phi) * np.cos(theta)
            y = radius * np.sin(phi) * np.sin(theta)
            z = radius * np.cos(phi)
            comp_pts = np.vstack([x, y, z]).T
            # Offset component relative to instance base center
            comp_offset = np.array([comp_idx * component_separation, 0, 0])
            comp_pts += base_center + comp_offset
            all_points.append(comp_pts)
            all_ids.append(np.full(points_per_component, inst_id + 1))

            # Make the first component the largest for instances > 0
            if inst_id > 0 and comp_idx == 0:
                 # Add extra points to make it clearly the largest
                 extra_pts = np.random.rand(200, 3)*0.1 + base_center + comp_offset
                 all_points.append(extra_pts)
                 all_ids.append(np.full(200, inst_id + 1))


        # Add some SOR-type noise scattered widely
        sor_noise = (np.random.rand(noise_points, 3) - 0.5) * 10 + base_center
        all_points.append(sor_noise)
        all_ids.append(np.full(noise_points, inst_id + 1))

    original_points = np.vstack(all_points)
    original_ids = np.concatenate(all_ids)

    # Shuffle
    shuffle_idx = np.random.permutation(len(original_points))
    original_points = original_points[shuffle_idx]
    original_ids = original_ids[shuffle_idx]

    print(f"Sample data generated: {len(original_points)} points across {len(np.unique(original_ids))} instances.")
    print("Instance 1 should have 1 component, Instance 2 two, Instance 3 three (before filtering).")
    print("------------------------------------")

    # --- Filtering Parameters (CRITICAL TO TUNE) ---
    # SOR
    sor_neighbors = 20
    sor_std = 2.0 # Less aggressive SOR initially

    # DBSCAN
    # eps depends HEAVILY on point density and scale. Measure distances between points
    # within a component vs between components to estimate a good value.
    # For this example data, points within a component are < 1.0 apart usually.
    # Components are 'component_separation=2.0' apart. So eps between 0.1 and 1.0 might work.
    dbscan_separation_eps = 0.3 # Should be smaller than component_separation but larger than intra-component distances
    dbscan_core_min_pts = 15

    # --- Run Filtering ---
    filtered_points, filtered_ids = filter_instances_combined(
        original_points,
        original_ids,
        enable_sor=True,
        sor_nb_neighbors=sor_neighbors,
        sor_std_ratio=sor_std,
        enable_dbscan_cleanup=True,
        dbscan_eps=dbscan_separation_eps,
        dbscan_min_samples=dbscan_core_min_pts,
        min_points_for_sor=20,
        min_points_for_dbscan=20, # Must be >= dbscan_min_samples
        verbose=True
    )

    # --- Use Results ---
    if filtered_points is not None:
        print("\n--- Filtered Results ---")
        print("Filtered Points Shape:", filtered_points.shape)
        print("Filtered IDs Shape:", filtered_ids.shape)
        print("Unique IDs remaining:", np.unique(filtered_ids))

        # Optional: Visualize (Color by instance ID)
        if OPEN3D_AVAILABLE:
            print("\nVisualizing result (Close window to continue)...")
            pcd_filtered = o3d.geometry.PointCloud()
            pcd_filtered.points = o3d.utility.Vector3dVector(filtered_points)

            # Color points by instance ID
            unique_final_ids = np.unique(filtered_ids)
            colors_filtered = np.zeros_like(filtered_points)
            # Simple coloring based on ID modulo 10
            for uid in unique_final_ids:
                 colors_filtered[filtered_ids == uid] = plt.get_cmap("tab10")(int(uid % 10))[:3] # Use matplotlib colormap
            pcd_filtered.colors = o3d.utility.Vector3dVector(colors_filtered)

            # Also visualize original for comparison (optional)
            # pcd_original = o3d.geometry.PointCloud()
            # pcd_original.points = o3d.utility.Vector3dVector(original_points)
            # colors_original = np.zeros_like(original_points)
            # unique_orig_ids = np.unique(original_ids)
            # for uid in unique_orig_ids:
            #     colors_original[original_ids == uid] = plt.get_cmap("tab10")(int(uid % 10))[:3]
            # pcd_original.colors = o3d.utility.Vector3dVector(colors_original)
            # pcd_original.translate((-15,0,0)) # Shift original

            # o3d.visualization.draw_geometries([pcd_original, pcd_filtered], window_name="Original (Shifted Left) vs Filtered (Right)")
            o3d.visualization.draw_geometries([pcd_filtered], window_name="Filtered Point Cloud by Instance")

        else:
            print("\nInstall Open3D and Matplotlib (`pip install open3d matplotlib`) to visualize results.")