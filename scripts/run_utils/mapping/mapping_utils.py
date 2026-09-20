import numpy as np
import torch
import open3d as o3d
import quaternion
import cv2
from argparse import Namespace

from habitat import logger
"""
Adapted from UniGoal depth_utils.py and rotation_utils.py and src/utils/model.py
Utilities for processing depth images and camera transformations.
"""


ANGLE_EPS = 0.001


def normalize(v):
    return v / np.linalg.norm(v)


def get_r_matrix(ax_, angle):
    ax = normalize(ax_)
    if np.abs(angle) > ANGLE_EPS:
        S_hat = np.array(
            [[0.0, -ax[2], ax[1]], [ax[2], 0.0, -ax[0]], [-ax[1], ax[0], 0.0]],
            dtype=np.float32)
        R = np.eye(3) + np.sin(angle) * S_hat + \
            (1 - np.cos(angle)) * (np.linalg.matrix_power(S_hat, 2))
    else:
        R = np.eye(3)
    return R


def habitat_translation(position):
    return np.array([position[0],position[2],position[1]])


def habitat_rotation(rotation):
    rotation_matrix = quaternion.as_rotation_matrix(rotation)
    transform_matrix = np.array([[1,0,0],
                                 [0,0,1],
                                 [0,1,0]])
    rotation_matrix = np.matmul(transform_matrix,rotation_matrix)
    return rotation_matrix


def ori_preprocess_depth(depth: np.ndarray, lower_bound: float = 0.51, upper_bound: float = 4.9):
    """
    Preprocess depth map by filling holes caused by thin/occluding structures.

    For objects like tables or chairs, depth discontinuities often occur at their legs
    or edges, creating "holes" in the depth map where background shows through.
    This function detects such depth jumps along each vertical scanline and fills
    the gap with the foreground (smaller) depth value.
    """
    assert depth.ndim == 2

    for i in range(depth.shape[1]):
        depth_line = depth[:, i]
        if not np.any(depth_line > 0):
            continue

        half_len = int(len(depth_line) * 0.5)
        indexes = []
        values = []
        for j in range(len(depth_line) - 2, half_len - 1, -1):
            prev_val = depth_line[j - 1]
            if prev_val < depth_line[j] - 0.08 and prev_val > 0.58:
                closest_idx = np.argmin(np.abs(depth_line[j:] - prev_val)) + j
                indexes.append([j, closest_idx])
                values.append(prev_val)

        if len(indexes) == 0:
            continue
        for [start, end], value in zip(indexes, values):
            if start == end:
                continue
            for idx in range(start, end + 1):
                if depth[idx, i] > value:
                    depth[idx, i] = value

    depth[(depth < lower_bound) | (depth > upper_bound)] = 0
    return depth


def get_intrinsic_matrix(width, height, fov):
    '''
    Returns a camera matrix from image size and fov in degrees.
    input: image width + height, horizontal fov of the camera
    output:
        xc: x-coordinate of the image center
        zc: y-coordinate of the image center
        f: focal length in pixels
    '''
    xc = (width - 1.) / 2.
    zc = (height - 1.) / 2.
    f = (width / 2.) / np.tan(np.deg2rad(fov / 2.))
    camera_matrix = np.array([[f, 0, xc],
                             [0, f, zc],
                             [0, 0, 1]], dtype=np.float32)
    return camera_matrix


def get_point_cloud_from_z_t(Y_t, camera_matrix, device, scale=1):
    """Projects the depth image Y into a 3D point cloud.
    Inputs:
        Y is BxHxW, unit=meters
        camera_matrix
    Outputs:
        X is positive going right
        Y is positive into the image
        Z is positive up in the image
        XYZ is ...xHxWx3 (in meters)
    """
    grid_x, grid_z = torch.meshgrid(torch.arange(Y_t.shape[-1]),
                                    torch.arange(Y_t.shape[-2] - 1, -1, -1))
    grid_x = grid_x.transpose(1, 0).to(device)
    grid_z = grid_z.transpose(1, 0).to(device)
    grid_x = grid_x.unsqueeze(0).expand(Y_t.shape)
    grid_z = grid_z.unsqueeze(0).expand(Y_t.shape)

    X_t = (grid_x[:, ::scale, ::scale] - camera_matrix.xc) * \
        Y_t[:, ::scale, ::scale] / camera_matrix.f
    Z_t = (grid_z[:, ::scale, ::scale] - camera_matrix.zc) * \
        Y_t[:, ::scale, ::scale] / camera_matrix.f

    XYZ = torch.stack(
        (X_t, Y_t[:, ::scale, ::scale], Z_t), dim=len(Y_t.shape))

    return XYZ


def get_pointcloud_from_depth(rgb:np.ndarray,depth:np.ndarray,intrinsic:np.ndarray):
    if len(depth.shape) == 3:
        depth = depth[:,:,0]

    filter_z,filter_x = np.where(depth>0)
    depth_values = depth[filter_z,filter_x]
    pixel_z = (depth.shape[0] - 1 - filter_z - intrinsic[1][2]) * depth_values / intrinsic[1][1]
    pixel_x = (filter_x - intrinsic[0][2])*depth_values / intrinsic[0][0]
    pixel_y = depth_values
    color_values = rgb[filter_z,filter_x]
    point_values = np.stack([pixel_x,pixel_z,-pixel_y],axis=-1)
    return point_values,color_values

def transform_camera_view_t(
        XYZ, sensor_height, camera_elevation_degree, device):
    """
    Transforms the point cloud into geocentric frame to account for
    camera elevation and angle
    Input:
        XYZ                     : ...x3
        sensor_height           : height of the sensor
        camera_elevation_degree : camera elevation to rectify.
    Output:
        XYZ : ...x3
    """
    R = get_r_matrix(
        [1., 0., 0.], angle=np.deg2rad(camera_elevation_degree))
    XYZ = torch.matmul(XYZ.reshape(-1, 3),
                       torch.from_numpy(R).float().transpose(1, 0).to(device)
                       ).reshape(XYZ.shape)
    XYZ[..., 2] = XYZ[..., 2] + sensor_height
    return XYZ

def transform_pose_t(XYZ, current_pose, device):
    """
    Transforms the point cloud into geocentric frame to account for
    camera position
    Input:
        XYZ                     : ...x3
        current_pose            : camera position (x, y, theta (radians))
    Output:
        XYZ : ...x3
    """
    R = get_r_matrix([0., 0., 1.], angle=current_pose[2] - np.pi / 2.)
    XYZ = np.matmul(XYZ.reshape(-1, 3), R.T).reshape(XYZ.shape)
    XYZ[..., 0] += current_pose[0]
    XYZ[..., 1] += current_pose[1]
    return XYZ


def translate_to_world(pointcloud, position, rotation):
    extrinsic = np.eye(4)
    extrinsic[0:3,0:3] = rotation
    extrinsic[0:3,3] = position
    world_points = np.matmul(extrinsic,
        np.concatenate((pointcloud, np.ones((pointcloud.shape[0],1))),axis=-1).T).T
    return world_points[:,0:3]


def accumulate_history_rgbd_to_panoramic(
    history_buffer,
    current_pose,
    camera_matrix,
    sensor_height,
    camera_elevation_degree,
    device,
    output_size=(480, 640),
    window_size=5,
    sparse_factor_old=4,
):
    """
    Accumulates point clouds from historical observations and projects them
    to create a temporally-aggregated panoramic-like RGB image.

    Args:
        history_buffer: List/deque of dicts with keys:
            - 'rgb': (H, W, 3) numpy array, RGB image
            - 'depth': (H, W) or (H, W, 1) numpy array, depth in meters
            - 'pose': (x, y, theta) tuple, agent pose in world frame at that timestep
            Each entry represents one timestep, ordered from oldest to newest.

        current_pose: (x, y, theta) tuple, current agent pose in world frame

        camera_matrix: Namespace with (xc, zc, f) from get_camera_matrix

        sensor_height: float, camera height above ground in meters

        camera_elevation_degree: float, camera pitch angle in degrees

        device: torch.device

        output_size: (H, W) tuple, size of output RGB image

        window_size: int, maximum number of historical frames to use

        sparse_factor_old: int, subsampling factor for older frames
            Recent frames (last 2) are dense, older frames are subsampled
            to create "black holes" and reduce computation.

    Returns:
        aggregated_rgb: (H, W, 3) numpy array, temporally-aggregated RGB image
            with black holes where no historical data is available
    """

    if len(history_buffer) == 0:
        # Return black image if no history
        return np.zeros((*output_size, 3), dtype=np.uint8)

    # Use only the most recent window_size frames
    recent_history = list(history_buffer)[-window_size:]

    W_pano = 1920
    H_pano = 960

    # Initialize output buffers
    # Use a depth buffer for z-buffering (occlusion handling)
    rgb_buffer = np.zeros((H_pano, W_pano, 3), dtype=np.uint8)
    depth_buffer = np.full((H_pano, W_pano), np.inf, dtype=np.float32)

    # Accumulate all point clouds in world frame first
    all_points_world = []  # List of (N, 3) tensors
    all_colors = []  # List of (N, 3) numpy arrays

    # Process each historical frame to accumulate point clouds
    for frame_idx, frame_data in enumerate(recent_history):
        rgb_hist = frame_data['rgb']  # (H, W, 3)
        depth_hist = frame_data['depth']  # (H, W) or (H, W, 1)
        pose_hist = frame_data['pose']  # (x, y, theta)

        # Uniform moderate sparsity for all frames
        # Spatial regions seen multiple times naturally get denser through accumulation
        # Regions seen once (like rear views) stay at this base density
        subsample = 2  # 2x subsampling for all frames - balances density and performance

        # Convert to torch tensors
        if isinstance(depth_hist, np.ndarray):
            depth_t = torch.from_numpy(depth_hist).to(device=device, dtype=torch.float32)
        else:
            depth_t = depth_hist.to(device=device, dtype=torch.float32)

        # Ensure depth is (1, H, W)
        if depth_t.ndim == 2:
            depth_t = depth_t.unsqueeze(0)
        elif depth_t.ndim == 3 and depth_t.shape[-1] == 1:
            depth_t = depth_t.squeeze(-1).unsqueeze(0)
        elif depth_t.ndim == 3 and depth_t.shape[0] == 1:
            pass  # Already (1, H, W)
        else:
            raise ValueError(f"Unexpected depth shape: {depth_t.shape}")

        H_hist, W_hist = depth_t.shape[1], depth_t.shape[2]

        # 1. Generate point cloud in historical camera frame
        XYZ_cam = get_point_cloud_from_z_t(depth_t, camera_matrix, device, scale=subsample)
        # Shape: (1, H_sub, W_sub, 3)

        # # Get corresponding RGB colors for this frame
        # if subsample > 1:
        #     rgb_sub = rgb_hist[::subsample, ::subsample, :]
        # else:
        #     rgb_sub = rgb_hist
        # colors_for_ply = rgb_sub.reshape(-1, 3).astype(np.float64) / 255.0

        # # Debug: save point clouds at each step using Open3D
        # import open3d as o3d
        # import os
        # debug_dir = "debug_ply"
        # os.makedirs(debug_dir, exist_ok=True)

        # # Step 1: Camera frame
        # pts_cam = XYZ_cam.reshape(-1, 3).cpu().numpy()
        # pcd_cam = o3d.geometry.PointCloud()
        # pcd_cam.points = o3d.utility.Vector3dVector(pts_cam)
        # pcd_cam.colors = o3d.utility.Vector3dVector(colors_for_ply)
        # o3d.io.write_point_cloud(f"{debug_dir}/frame{frame_idx:02d}_step1_camera_frame.ply", pcd_cam)
        # logger.info(f"Saved frame{frame_idx:02d}_step1_camera_frame.ply with {len(pts_cam)} points")

        # 2. Transform to historical agent frame (geocentric)
        XYZ_agent_hist = transform_camera_view_t(
            XYZ_cam, sensor_height, camera_elevation_degree, device
        )

        # # Step 2: Agent frame
        # pts_agent = XYZ_agent_hist.reshape(-1, 3).cpu().numpy()
        # pcd_agent = o3d.geometry.PointCloud()
        # pcd_agent.points = o3d.utility.Vector3dVector(pts_agent)
        # pcd_agent.colors = o3d.utility.Vector3dVector(colors_for_ply)
        # o3d.io.write_point_cloud(f"{debug_dir}/frame{frame_idx:02d}_step2_agent_frame.ply", pcd_agent)
        # logger.info(f"Saved frame{frame_idx:02d}_step2_agent_frame.ply with {len(pts_agent)} points")

        # 3. Transform to world frame using historical pose
        XYZ_world = transform_pose_t(XYZ_agent_hist, pose_hist, device)
        # Shape: (1, H_sub, W_sub, 3)
        # print(f"pose_hist: {pose_hist}")

        # # Step 3: World frame
        # pts_world = XYZ_world.reshape(-1, 3).cpu().numpy()
        # pcd_world = o3d.geometry.PointCloud()
        # pcd_world.points = o3d.utility.Vector3dVector(pts_world)
        # pcd_world.colors = o3d.utility.Vector3dVector(colors_for_ply)
        # o3d.io.write_point_cloud(f"{debug_dir}/frame{frame_idx:02d}_step3_world_frame.ply", pcd_world)
        # logger.info(f"Saved frame{frame_idx:02d}_step3_world_frame.ply with {len(pts_world)} points, pose={pose_hist}")

        # Flatten to (N, 3) and accumulate
        XYZ_world_flat = XYZ_world.reshape(-1, 3)
        all_points_world.append(XYZ_world_flat)

        # Get corresponding RGB colors
        if subsample > 1:
            rgb_sub = rgb_hist[::subsample, ::subsample, :]
        else:
            rgb_sub = rgb_hist
        colors_flat = rgb_sub.reshape(-1, 3)
        all_colors.append(colors_flat)

    # Concatenate all accumulated points
    if len(all_points_world) == 0:
        logger.info("WARNING: No accumulated points in history buffer")
        return np.zeros((H_pano, W_pano, 3), dtype=np.uint8)

    XYZ_world_all = torch.cat(all_points_world, dim=0)  # (N_total, 3)
    colors_all = np.concatenate(all_colors, axis=0)  # (N_total, 3)
    # logger.info(f"Accumulated {XYZ_world_all.shape[0]} points from {len(recent_history)} frames")

    # Now transform entire accumulated cloud to current agent frame
    x_curr, y_curr, theta_curr = current_pose

    # 1. Translate by -current_position
    XYZ_translated = XYZ_world_all.clone()
    XYZ_translated[:, 0] -= x_curr
    XYZ_translated[:, 1] -= y_curr

    # 2. Rotate by -(theta_curr - pi/2)
    angle = -(theta_curr - np.pi / 2.)
    R = get_r_matrix([0., 0., 1.], angle)
    R_t = torch.from_numpy(R).float().transpose(1, 0).to(device)

    XYZ_current_agent = torch.matmul(XYZ_translated, R_t)

    # 3. Transform from agent frame to camera frame (inverse of transform_camera_view_t)
    XYZ_current_agent[:, 2] -= sensor_height

    angle_cam = -np.deg2rad(camera_elevation_degree)
    R_cam = get_r_matrix([1., 0., 0.], angle_cam)
    R_cam_t = torch.from_numpy(R_cam).float().transpose(1, 0).to(device)

    XYZ_cam_current = torch.matmul(XYZ_current_agent, R_cam_t)
    # Shape: (N_total, 3)

    # 4. Project accumulated point cloud using equirectangular projection (standard 2:1 panorama)
    # Equirectangular: azimuth -> u (360°), elevation -> v (180°)
    # X, Y, Z in camera frame where Y is forward depth
    X = XYZ_cam_current[:, 0]  # (N_total,) right
    Y = XYZ_cam_current[:, 1]  # forward (depth)
    Z = XYZ_cam_current[:, 2]  # up

    # Compute 3D distance for filtering
    dist_3d = torch.sqrt(X**2 + Y**2 + Z**2)

    # Filter out points too close or too far
    valid_mask = (dist_3d > 0.1) & (dist_3d < 15.0)

    X_valid = X[valid_mask]
    Y_valid = Y[valid_mask]
    Z_valid = Z[valid_mask]
    dist_valid = dist_3d[valid_mask]
    colors_valid = colors_all[valid_mask.cpu().numpy()]

    # logger.info(f"After depth filtering: {X_valid.shape[0]} valid points (from {XYZ_world_all.shape[0]})")

    if X_valid.shape[0] == 0:
        logger.info("WARNING: No valid points after depth filtering")
        return np.zeros((H_pano, W_pano, 3), dtype=np.uint8)

    # Equirectangular projection:
    # azimuth (horizontal angle): atan2(X, Y), range [-π, π] -> maps to [0, W_pano]
    # elevation (vertical angle): asin(Z / dist), range [-π/2, π/2] -> maps to [0, H_pano]

    # Compute azimuth (horizontal angle from forward direction)
    azimuth = torch.atan2(X_valid, Y_valid)  # [-π, π]

    # Compute elevation (vertical angle from horizontal plane)
    # Use asin for proper spherical projection
    elevation = torch.asin(torch.clamp(Z_valid / dist_valid, -1.0, 1.0))  # [-π/2, π/2]

    # Map to pixel coordinates:
    # u: azimuth [-π, π] -> [0, W_pano], with 0° (forward) at center
    u = (azimuth + np.pi) / (2 * np.pi) * (W_pano - 1)

    # v: elevation [π/2, -π/2] -> [0, H_pano], with +90° (up) at top, -90° (down) at bottom
    v = (np.pi / 2 - elevation) / np.pi * (H_pano - 1)

    # Round to integer pixel coordinates
    u_int = torch.round(u).long()
    v_int = torch.round(v).long()

    # Filter valid pixels within panoramic bounds
    bound_mask = (u_int >= 0) & (u_int < W_pano) & (v_int >= 0) & (v_int < H_pano)

    u_final = u_int[bound_mask].cpu().numpy()
    v_final = v_int[bound_mask].cpu().numpy()
    depth_final = dist_valid[bound_mask].cpu().numpy()  # Use 3D distance for z-buffering
    colors_final = colors_valid[bound_mask.cpu().numpy()]

    # logger.info(f"After boundary filtering: {len(u_final)} pixels to render")

    # Z-buffer rendering: single pixel per point (no splatting for performance)
    # Cylindrical projection naturally provides better coverage than perspective
    for i in range(len(u_final)):
        v_px, u_px = v_final[i], u_final[i]
        d = depth_final[i]

        if d < depth_buffer[v_px, u_px]:
            depth_buffer[v_px, u_px] = d
            rgb_buffer[v_px, u_px, :] = colors_final[i]

    # Interpolate to fill holes only within 3 pixels of colored regions
    import cv2

    gray = cv2.cvtColor(rgb_buffer, cv2.COLOR_RGB2GRAY)
    has_color = (gray > 0).astype(np.uint8)

    # Dilate colored region by 3 pixels to get the interpolation boundary
    kernel = np.ones((11, 11), np.uint8)  # 7x7 kernel gives ~3 pixel expansion
    dilated = cv2.dilate(has_color, kernel, iterations=1)

    # Holes to fill = dilated region AND no color (black pixels near colored regions)
    hole_mask = ((dilated > 0) & (gray == 0)).astype(np.uint8) * 255

    # Inpaint only those holes
    rgb_inpainted = cv2.inpaint(rgb_buffer, hole_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    return rgb_inpainted


def calculate_yaw(current_agent_world_rotation, world_rotation):
    """
    计算当前旋转相对于初始参考旋转的 yaw 角度（偏航角）。

    Args:
        current_agent_world_rotation: 当前的 3x3 旋转矩阵（经过 habitat_rotation 变换后）
        world_rotation: 初始参考的 3x3 旋转矩阵（经过 habitat_rotation 变换后）

    Returns:
        yaw: 相对偏航角，单位为角度（degrees），范围 [-180, 180]
    """
    # 计算相对旋转矩阵：R_rel = R_current @ R_reference^T
    R_rel = np.matmul(current_agent_world_rotation, world_rotation.T)

    # 从相对旋转矩阵中提取 yaw 角度（绕 Z 轴的旋转）
    # 对于绕 Z 轴的旋转矩阵：
    # R_z = [[cos(θ), -sin(θ), 0],
    #        [sin(θ),  cos(θ), 0],
    #        [0,       0,      1]]
    # 所以 yaw = atan2(R[1,0], R[0,0])
    yaw_rad = np.arctan2(R_rel[1, 0], R_rel[0, 0])
    yaw_deg = np.rad2deg(yaw_rad)

    return yaw_deg


def gpu_pointcloud_from_array(points,colors,device):
    pointcloud = o3d.t.geometry.PointCloud(device)
    pointcloud.point.positions = o3d.core.Tensor(points,dtype=o3d.core.Dtype.Float32,device=device)
    pointcloud.point.colors = o3d.core.Tensor(colors.astype(np.float32)/255.0,dtype=o3d.core.Dtype.Float32,device=device)
    return pointcloud


def gpu_merge_pointcloud(pcdA,pcdB, merge_color=True):
    device = o3d.core.Device('cuda:0'.upper())
    if pcdA is None and pcdB is None:
        # return empty pointcloud
        empty_pc = o3d.t.geometry.PointCloud(device)
        empty_pc.point.positions = o3d.core.Tensor([],dtype=o3d.core.Dtype.Float32,device=device)
        empty_pc.point.colors = o3d.core.Tensor([],dtype=o3d.core.Dtype.Float32,device=device)
        return empty_pc
    if pcdA is None or pcdA.is_empty():
        return pcdB
    if pcdB is None or pcdB.is_empty():
        return pcdA
    if merge_color:
        # change color of pcdB to pcdA
        colorA = pcdA.point.colors.cpu().numpy()[0]
        colorB = np.array([colorA] * pcdB.point.positions.shape[0])
        pcdB.point.colors = o3d.core.Tensor(colorB,dtype=pcdA.point.colors.dtype,device=pcdA.point.colors.device)
    return pcdA + pcdB




def get_closest_disances_and_points(current_pcd_position,
                                    whole_pcd_position,
                                    current_position,
                                    max_distance=1.7):
    distance_current = np.linalg.norm(current_pcd_position - current_position, axis=1)
    distance_whole = np.linalg.norm(whole_pcd_position - current_position, axis=1)

    angles_current = np.arctan2(current_pcd_position[:, 1] - current_position[1],
                                current_pcd_position[:, 0] - current_position[0])

    angles_whole = np.arctan2(whole_pcd_position[:, 1] - current_position[1],
                                whole_pcd_position[:, 0] - current_position[0])

    start_angle = -np.pi
    end_angle = np.pi

    angle_resolution = (2 * np.pi) / 360 * 3
    angle_bins = np.arange(start_angle, end_angle, angle_resolution)

    nearest_distances = np.full_like(angle_bins, max_distance, dtype=float)
    nearest_points = np.empty((0, 3))

    for i, angle in enumerate(angle_bins):
        pcd_position_within_range = current_pcd_position[(angles_current >= angle) & (
            angles_current < (angle + angle_resolution))]
        distance_within_range = distance_current[(angles_current >= angle) & (
            angles_current < (angle + angle_resolution))]

        whole_pcd_position_within_range = whole_pcd_position[(angles_whole >= angle) & (
            angles_whole < (angle + angle_resolution))]
        distance_whole_within_range = distance_whole[(angles_whole >= angle) & (
            angles_whole < (angle + angle_resolution))]

        if len(pcd_position_within_range) != 0:
            current_distance_min = np.min(distance_within_range)
        else:
            current_distance_min = 100.0

        if len(whole_pcd_position_within_range) != 0:
            whole_distance_min = np.min(distance_whole_within_range)
        else:
            whole_distance_min = 100.0

        minimal_distance = min(current_distance_min, whole_distance_min)
        if nearest_distances[i] > minimal_distance:
            nearest_distances[i] = minimal_distance
            nearest_point = np.concatenate((
                pcd_position_within_range[
                    distance_within_range < nearest_distances[i] + 0.06],
                whole_pcd_position_within_range[
                    distance_whole_within_range < nearest_distances[i] + 0.06]), axis=0)

            if len(nearest_point) > 0:
                nearest_points = np.concatenate((nearest_points, nearest_point), axis=0)

    return nearest_distances, nearest_points


def keep_the_max_connected_component(pcd):
    # use the dbscan to cluster the points
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(pcd.cluster_dbscan(eps=0.12, min_points=12, print_progress=False))
    if len(labels) == 0:
        return pcd
    max_label = labels.max().cpu().numpy()
    # extract the max connect component
    max_cluster = []
    for i in range(max_label + 1):
        mask_idx_tensor = o3d.core.Tensor((labels == i).nonzero()[0],
                                            o3d.core.Dtype.Int64,
                                            device=pcd.device)
        cluster = pcd.select_by_index(mask_idx_tensor)
        max_cluster.append(cluster)
    max_cluster = max_cluster[np.argmax(
        [len(cluster.point.positions.cpu().numpy()) for cluster in max_cluster])]
    return max_cluster


def post_process_map(full_map):
    occ = full_map[:, :, 0]
    exp = full_map[:, :, 1]

    # 填洞
    occ = cv2.dilate(occ.astype(np.uint8), np.ones((3,3)))
    occ = cv2.erode(occ.astype(np.uint8), np.ones((3,3)))
    exp = cv2.dilate(exp.astype(np.uint8), np.ones((3,3)))
    exp = cv2.erode(exp.astype(np.uint8), np.ones((3,3)))

    exp[occ == 1] = 0.0

    # keep only the max connected component of exp
    exp_binary = (exp > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        exp_binary, connectivity=4
    )

    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_label = np.argmax(areas) + 1

        exp_filtered = (labels == max_label).astype(exp.dtype)
        exp = exp_filtered

    exp[occ == 1] = 1.0
    full_map[:, :, 1] = exp
    exp_copy_1 = exp.copy()

    # keep only the occ components, where it connectes to the exp components
    exp_copy = exp.copy()
    exp_binary = (exp_copy > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        exp_binary, connectivity=4
    )
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_label = np.argmax(areas) + 1
        exp_filtered = (labels == max_label).astype(exp.dtype)
        exp = exp_filtered

    occ[exp == 0] = 0.0
    full_map[:, :, 0] = occ

    exp_copy_1[exp == 0] = 0.0
    full_map[:, :, 1] = exp_copy_1

    return full_map

def pixel_to_agent_wp(
    pixel,
    depth_image,
    camera_matrix,
    sensor_height,
    camera_elevation_degree=0,
    device='cpu',
    depth_unit='m',
    direction='front',
):
    """
    Transforms pixel location(s) to waypoint(s) in the agent egocentric frame.

    Args:
        pixel: (2,) or (N, 2) tensor / array, (u, v) in image coordinates.
        depth_image: (H, W), (H, W, 1) or (1, H, W), depth in 'm' or 'cm'.
        camera_matrix: same object you already use in Mapping (xc, zc, f).
        sensor_height: camera height above ground in meters.
        camera_elevation_degree: camera pitch angle in degrees.
        device: torch device.
        depth_unit: 'm' or 'cm'.
        direction: Direction of the panoramic image - one of 'front', 'left', 'back', 'right'
           The waypoint will be rotated to account for the camera orientation:
           - 'front': 0° (no rotation)
           - 'left': +90° CCW
           - 'back': 180°
           - 'right': -90° (or +270° CCW)


    Returns:
        waypoints_m: (N, 3) tensor in agent egocentric frame, in meters.
    """

    # ---- depth -> torch (1, H, W) in cm ----
    if isinstance(depth_image, np.ndarray):
        depth_t = torch.from_numpy(depth_image).to(device=device, dtype=torch.float32)
    else:
        depth_t = depth_image.to(device=device, dtype=torch.float32)

    # depth_t = preprocess_depth(depth_t)

    # collapse extra dims to (H, W)
    if depth_t.ndim == 3:
        depth_t = depth_t.squeeze()
    if depth_t.ndim != 2:
        raise ValueError(f"Depth image must be (H, W) or equivalent; got {tuple(depth_t.shape)}")

    depth_t = depth_t.unsqueeze(0)  # (1, H, W)

    # No conversion needed - all units now in meters
    if depth_unit not in ["m", "cm"]:
        raise ValueError("depth_unit must be 'm' or 'cm'.")
    if depth_unit == "cm":
        raise ValueError("cm units are no longer supported - use meters")

    B, H, W = depth_t.shape
    assert B == 1, "pixel_to_agent_wp currently supports only batch size 1."

    # ---- full point cloud in agent frame (meters) ----
    point_cloud_cam = get_point_cloud_from_z_t(depth_t, camera_matrix, device)
    point_cloud_agent = transform_camera_view_t(
        point_cloud_cam,
        sensor_height,  # meters
        camera_elevation_degree,
        device,
    )  # (1, H, W, 3) in meters

    # ---- index pixels ----
    pixel_t = torch.as_tensor(pixel, dtype=torch.float32, device=device)
    if pixel_t.ndim == 1:
        pixel_t = pixel_t.unsqueeze(0)  # (1, 2)

    u = pixel_t[:, 0]
    v = pixel_t[:, 1]

    u_idx = torch.round(u).long().clamp(0, W - 1)
    v_idx = torch.round(v).long().clamp(0, H - 1)

    waypoints_m = point_cloud_agent[0, v_idx, u_idx, :]  # (N, 3) in meters

    return waypoints_m.cpu().numpy()  # (N, 3) tensor


def get_camera_matrix(width, height, fov):
    '''
    Returns a camera matrix from image size and fov in degrees.
    input: image width + height, horizontal fov of the camera
    output:
        xc: x-coordinate of the image center
        zc: y-coordinate of the image center
        f: focal length in pixels
    '''
    xc = (width - 1.) / 2.
    zc = (height - 1.) / 2.
    f = (width / 2.) / np.tan(np.deg2rad(fov / 2.))
    camera_matrix = {'xc': xc, 'zc': zc, 'f': f}
    camera_matrix = Namespace(**camera_matrix)
    return camera_matrix

def world_to_agent(
    world_wp,
    world_pos,
    world_rot,
):
    """
    Convert a waypoint in world coordinates to the agent-centric frame.
    This is the inverse of agent_to_world().

    Args:
        world_wp: (3,) array-like
            [x_world, y_world, z_world] in world frame (meters).
        world_pos: sim.get_agent_state().position, iterable of length 3
            [x, y, z] in world frame (meters).
        world_rot: sim.get_agent_state().rotation, quaternion.quaternion
            Rotation of the agent from local to world frame.

    Returns:
        wp_agent: np.ndarray of shape (3,) = [right, forward, up] in agent frame (meters).
    """
    import quaternion

    world_wp = np.asarray(world_wp, dtype=np.float32)
    world_pos = np.asarray(world_pos, dtype=np.float32)

    if world_wp.shape[0] != 3:
        raise ValueError("world_wp must have 3 elements [x, y, z].")

    # Get vector from agent to waypoint in world frame
    v_world = world_wp - world_pos

    # Convert quaternion to rotation matrix and get its inverse (transpose)
    R = quaternion.as_rotation_matrix(world_rot)  # shape (3,3)
    R_inv = R.T  # Inverse of rotation matrix is its transpose

    # Rotate world vector into local frame
    v_local = R_inv @ v_world  # (3,)

    # Convert from Habitat's local frame (X right, Y up, -Z forward)
    # to agent frame (forward, right, up)
    right = float(v_local[0])   # X
    up = float(v_local[1])      # Y
    fwd = float(-v_local[2])    # -Z

    wp_agent = np.array([right, fwd, up], dtype=np.float32)
    return wp_agent



def agent_to_pixel(
    wp_agent,
    camera_matrix,
    sensor_height,
    camera_elevation_degree=0,
    image_height=480,
    image_width=640,
):
    """
    Project a waypoint in agent frame to pixel coordinates in the camera image.
    This is the inverse of pixel_to_agent_wp().

    Args:
        wp_agent: (3,) array-like [right, forward, up] in agent frame (meters).
        camera_matrix: Camera matrix object with attributes xc, zc, f.
        sensor_height: Camera height above ground in meters.
        camera_elevation_degree: Camera pitch angle in degrees.
        image_height: Height of the image in pixels.
        image_width: Width of the image in pixels.

    Returns:
        pixel: (2,) numpy array [u, v] in image coordinates, or None if point is behind camera.
    """
    wp_agent = np.asarray(wp_agent, dtype=np.float32)
    if wp_agent.shape[0] != 3:
        raise ValueError("wp_agent must have 3 elements [right, forward, up].")

    right, fwd, up = float(wp_agent[0]), float(wp_agent[1]), float(wp_agent[2])

    # Convert from agent frame to camera frame
    # Agent frame: (right, forward, up) in meters
    # Camera frame: X=right, Y=forward (depth), Z=up, units in cm
    X_cam = right * 100.0      # right -> X (cm)
    Y_cam = fwd * 100.0        # forward -> Y (depth, cm)
    Z_cam = up * 100.0         # up -> Z (cm)

    # Apply inverse camera elevation transformation
    # This reverses the rotation applied in transform_camera_view_t
    angle_rad = np.deg2rad(camera_elevation_degree)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # Adjust for sensor height
    Z_cam_adjusted = Z_cam - sensor_height * 100.0

    # Inverse rotation around X-axis
    Y_cam_rot = Y_cam * cos_a + Z_cam_adjusted * sin_a
    Z_cam_rot = -Y_cam * sin_a + Z_cam_adjusted * cos_a

    # Check if point is behind the camera
    if Y_cam_rot <= 0:
        return None  # Point is behind camera, cannot project

    # Project to image coordinates using camera intrinsics
    # From get_point_cloud_from_z_t, we have:
    # X = (u - xc) * Y / f  =>  u = X * f / Y + xc
    # Z = (v - zc) * Y / f  =>  v = Z * f / Y + zc
    u = X_cam * camera_matrix.f / Y_cam_rot + camera_matrix.xc
    v = Z_cam_rot * camera_matrix.f / Y_cam_rot + camera_matrix.zc

    # Note: v coordinate is inverted in get_point_cloud_from_z_t
    # grid_z uses torch.arange(Y_t.shape[-2] - 1, -1, -1)
    # So we need to invert v
    v = (image_height - 1) - v

    # Check if pixel is within image bounds
    if u < 0 or u >= image_width or v < 0 or v >= image_height:
        return None  # Point projects outside image

    pixel = np.array([u, v], dtype=np.float32)
    return pixel

def world_to_pixel(
    world_wp,
    world_pos,
    world_rot,
    camera_matrix,
    sensor_height,
    camera_elevation_degree=0,
    image_height=480,
    image_width=640,
):
    """
    Project a waypoint in world coordinates to pixel coordinates in the current camera view.
    This combines world_to_agent() and agent_to_pixel().

    Args:
        world_wp: (3,) array-like [x_world, y_world, z_world] in world frame (meters).
        world_pos: Agent world position from sim.get_agent_state().position.
        world_rot: Agent world rotation from sim.get_agent_state().rotation (quaternion).
        camera_matrix: Camera matrix object with attributes xc, zc, f.
        sensor_height: Camera height above ground in meters.
        camera_elevation_degree: Camera pitch angle in degrees.
        image_height: Height of the image in pixels.
        image_width: Width of the image in pixels.

    Returns:
        pixel: (2,) numpy array [u, v] in image coordinates, or None if point cannot be projected.
    """
    # Step 1: Convert world coordinates to agent frame
    wp_agent = world_to_agent(world_wp, world_pos, world_rot)

    # Step 2: Project agent frame to pixel coordinates
    pixel = agent_to_pixel(
        wp_agent,
        camera_matrix,
        sensor_height,
        camera_elevation_degree,
        image_height,
        image_width,
    )

    return pixel, wp_agent


def agent_to_world(wp_agent, world_pos, world_rot):
    """
    Convert a waypoint in the agent-centric frame to world coordinates.

    Args:
        wp_agent: (3,) array-like [right, forward, up] in agent frame (meters).
        world_pos: Agent world position from sim.get_agent_state().position.
        world_rot: Agent world rotation from sim.get_agent_state().rotation (quaternion).

    Returns:
        world_wp: np.ndarray of shape (3,) = [x_world, y_world, z_world] in meters.
    """
    import quaternion

    wp_agent = np.asarray(wp_agent, dtype=np.float32)
    world_pos = np.asarray(world_pos, dtype=np.float32)

    if wp_agent.shape[0] != 3:
        raise ValueError("wp_agent must have 3 elements [right, forward, up].")

    right, fwd, up = float(wp_agent[0]), float(wp_agent[1]), float(wp_agent[2])

    # Convert from agent frame (right, forward, up) to Habitat's local frame (X right, Y up, -Z forward)
    v_local = np.array([right, up, -fwd], dtype=np.float32)

    # Get rotation matrix from quaternion
    R = quaternion.as_rotation_matrix(world_rot)  # shape (3,3)

    # Rotate local vector to world frame
    v_world = R @ v_local  # (3,)

    # Add agent position to get world waypoint
    world_wp = world_pos + v_world

    return world_wp





def pixel_to_agent_wp(
    pixel,      # input (X, Y), (colum, row)
    depth_image,
    camera_matrix,
    sensor_height,
    camera_elevation_degree=0,
    device='cpu',
    depth_unit='m',
    direction='front',
):
    """
    Transforms pixel location(s) to waypoint(s) in the agent egocentric frame.

    Args:
        pixel: (2,) or (N, 2) tensor / array, (u, v) in image coordinates.
        depth_image: (H, W), (H, W, 1) or (1, H, W), depth in 'm' or 'cm'.
        camera_matrix: same object you already use in Mapping (xc, zc, f).
        sensor_height: camera height above ground in meters.
        camera_elevation_degree: camera pitch angle in degrees.
        device: torch device.
        depth_unit: 'm' or 'cm'.
        direction: Direction of the panoramic image - one of 'front', 'left', 'back', 'right'
           The waypoint will be rotated to account for the camera orientation:
           - 'front': 0° (no rotation)
           - 'left': +90° CCW
           - 'back': 180°
           - 'right': -90° (or +270° CCW)


    Returns:
        waypoints_m: (N, 3) tensor in agent egocentric frame, in meters.
    """

    # ---- depth -> torch (1, H, W) in cm ----
    if isinstance(depth_image, np.ndarray):
        depth_t = torch.from_numpy(depth_image).to(device=device, dtype=torch.float32)
    else:
        depth_t = depth_image.to(device=device, dtype=torch.float32)

    # depth_t = preprocess_depth(depth_t)

    # collapse extra dims to (H, W)
    if depth_t.ndim == 3:
        depth_t = depth_t.squeeze()
    if depth_t.ndim != 2:
        raise ValueError(f"Depth image must be (H, W) or equivalent; got {tuple(depth_t.shape)}")

    depth_t = depth_t.unsqueeze(0)  # (1, H, W)

    # No conversion needed - all units now in meters
    if depth_unit not in ["m", "cm"]:
        raise ValueError("depth_unit must be 'm' or 'cm'.")
    if depth_unit == "cm":
        raise ValueError("cm units are no longer supported - use meters")

    B, H, W = depth_t.shape
    assert B == 1, "pixel_to_agent_wp currently supports only batch size 1."

    # ---- full point cloud in agent frame (meters) ----
    point_cloud_cam = get_point_cloud_from_z_t(depth_t, camera_matrix, device)
    point_cloud_agent = transform_camera_view_t(
        point_cloud_cam,
        sensor_height,  # meters
        camera_elevation_degree,
        device,
    )  # (1, H, W, 3) in meters

    # ---- index pixels ----
    pixel_t = torch.as_tensor(pixel, dtype=torch.float32, device=device)
    if pixel_t.ndim == 1:
        pixel_t = pixel_t.unsqueeze(0)  # (1, 2)

    u = pixel_t[:, 0]
    v = pixel_t[:, 1]

    u_idx = torch.round(u).long().clamp(0, W - 1)
    v_idx = torch.round(v).long().clamp(0, H - 1)

    waypoints_m = point_cloud_agent[0, v_idx, u_idx, :]  # (N, 3) in meters

    return waypoints_m.cpu().numpy()  # (N, 3) tensor

def safe_select_by_index(pcd: o3d.t.geometry.PointCloud, indices):
    if indices.shape[0] == 0:
        empty = o3d.t.geometry.PointCloud(pcd.device)
        empty.point["positions"] = o3d.core.Tensor.zeros(
            (0, 3),
            dtype=pcd.point.positions.dtype,
            device=pcd.device
        )
        return empty

    return pcd.select_by_index(indices)


def grid_coords_to_world(grid_x, grid_y, resolution, map_center, floor_height):
    points = np.stack((grid_y, grid_x, np.ones_like(grid_x) * floor_height), axis=1).astype(np.float32)
    points[:, 0] = points[:, 0] * resolution - map_center
    points[:, 1] = points[:, 1] * resolution - map_center
    return points


def world_coords_to_grid(points, resolution, map_center, global_width, global_height):
    grid_x = (points[:, 0] + map_center) / resolution
    grid_y = (points[:, 1] + map_center) / resolution
    cols = grid_x.astype(int)
    rows = grid_y.astype(int)
    valid = (rows >= 0) & (rows < global_height) & (cols >= 0) & (cols < global_width)
    rows, cols = rows[valid], cols[valid]
    return rows, cols