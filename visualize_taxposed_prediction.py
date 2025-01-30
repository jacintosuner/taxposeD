import numpy as np
import open3d as o3d
import os
import argparse

# Example usage: python visualize_taxposed_prediction.py --taxposed_prediction /home/jacinto/robot-grasp/data/contact_graspnet_pipeline_results_for_red_mug/taxposed_prediction.npy --point_cloud /home/jacinto/robot-grasp/data/contact_graspnet_pipeline_results_for_red_mug/point_cloud.npy --segmented_point_cloud /home/jacinto/robot-grasp/data/contact_graspnet_pipeline_results_for_red_mug/segmented_point_cloud.npy
# Example usage: python visualize_taxposed_prediction.py --taxposed_prediction /home/jacinto/robot-grasp/data/contact_graspnet_pipeline_results_20250113_192755/taxposed_prediction.npy --point_cloud /home/jacinto/robot-grasp/data/contact_graspnet_pipeline_results_20250113_192755/point_cloud.npy --segmented_point_cloud /home/jacinto/robot-grasp/data/contact_graspnet_pipeline_results_20250113_192755/segmented_point_cloud.npy

def prepare_data(taxposed_prediction, point_cloud, segmented_point_cloud):
    taxposed_prediction = np.load(taxposed_prediction)
    point_cloud = np.load(point_cloud)
    segmented_point_cloud = np.load(segmented_point_cloud)
    # Take the first transformation matrix
    transformation_matrix = taxposed_prediction[0]
    
    return transformation_matrix, point_cloud, segmented_point_cloud

def visualize(transformation_matrix, point_cloud, segmented_point_cloud):

    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    # Extract object points (where segmented_point_cloud >= 0)
    object_points = point_cloud[segmented_point_cloud >= 0]

    # Create a new point cloud for the transformed object points
    transformed_pcd = o3d.geometry.PointCloud()
    transformed_pcd.points = o3d.utility.Vector3dVector(object_points)
    # Transform the object points
    transformed_pcd.transform(transformation_matrix)

    # Create colors for the point cloud
    colors = np.zeros_like(point_cloud)
    colors[segmented_point_cloud == -1] = [0, 0, 0]  # Environment in black
    colors[segmented_point_cloud >= 0] = [0, 0, 1]   # Objects in blue
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Create colors for the transformed point cloud
    transformed_colors = np.ones_like(object_points) * np.array([1, 0, 0])
    transformed_pcd.colors = o3d.utility.Vector3dVector(transformed_colors)

    # Visualize the point clouds
    o3d.visualization.draw_geometries([pcd, transformed_pcd])

if __name__ == '__main__':
    # print('Starting robot')
    # fa = FrankaArm()
    # EEF_POSE = np.array(fa._state_client._get_current_robot_state().robot_state.O_T_EE).reshape(4, 4).transpose()

    parser = argparse.ArgumentParser()
    parser.add_argument('--taxposed_prediction', type=str, help='taxposed prediction')
    parser.add_argument('--point_cloud', type=str, help='point cloud')
    parser.add_argument('--segmented_point_cloud', type=str, help='segmented point cloud')

    cfgs = parser.parse_args()

    # visualize_rgbdk(cfgs.file_paths, EEF_POSE)
    taxposed_prediction, point_cloud, segmented_point_cloud = prepare_data(cfgs.taxposed_prediction,
                                                                           cfgs.point_cloud,
                                                                           cfgs.segmented_point_cloud)
    visualize(taxposed_prediction,
              point_cloud,
              segmented_point_cloud)