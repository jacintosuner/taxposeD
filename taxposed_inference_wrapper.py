import argparse
import numpy as np
import hydra
import json
from torch.utils.data import DataLoader
from python.equivariant_pose_graph.dataset.point_cloud_dataset_inference import TestPointCloudDataset
from python.equivariant_pose_graph.utils.load_model_utils import load_model
from scipy.spatial.transform import Rotation as R
from pycocotools import mask as maskUtils


# Example usage: python taxposed_inference_wrapper.py --input_path /home/jacinto/robot-grasp/data/contact_graspnet_pipeline_results_for_red_mug/contact_graspnet_input.npy --output_dir /home/jacinto/robot-grasp/data/contact_graspnet_pipeline_results_for_red_mug --gsam2_pred_path /home/jacinto/robot-grasp/data/contact_graspnet_pipeline_results_for_red_mug/grounded_sam_seg_mug.json --cfg plan.yaml


class TaxposedWrapper:
    def __init__(self,
                 output_dir,
                 input_path,
                 gsam2_pred_path,
                 cfg,
                 vis_threshold = 1.8,
                 debug=False):
        self.output_dir = output_dir
        self.input_path = input_path
        self.gsam2_pred_path = gsam2_pred_path
        self.cfg = cfg
        self.vis_threshold = vis_threshold
        self.debug = debug
        self.num_suggestions = self.cfg.model.num_suggestions

        # Load model hyperparameters:
        with hydra.initialize(
            version_base=None, config_path=self.cfg.model.config_folder
        ):
            self.model_cfg = hydra.compose(self.cfg.model.config_file)

        # Load the model
        self.model = load_model(
            self.cfg.model.weights,
            has_pzX=True,
            conditioning=self.model_cfg.conditioning,
            cfg=self.model_cfg,
        )

    def _load_segmentation_from_gsam2(self, segmentation_path):
        with open(segmentation_path, 'r') as f:
            segmentation_data = json.load(f)
        return maskUtils.decode(segmentation_data['annotations'][0]['segmentation'])

    def _invert_transformation(self, T):
        """
        Invert a 4x4 similarity transformation matrix.
        
        Args:
            T (np.ndarray): A 4x4 transformation matrix.
            
        Returns:
            np.ndarray: The inverted 4x4 transformation matrix.
        """
        assert T.shape == (4, 4), "Input must be a 4x4 matrix."
        
        # Extract rotation and scaling (upper-left 3x3) and translation (top-right 3x1)
        RS = T[:3, :3]
        t = T[:3, 3]
        
        # Invert rotation and scaling
        RS_inv = np.linalg.inv(RS)  # Assumes RS is non-singular
        
        # Invert translation
        t_inv = -RS_inv @ t
        
        # Construct the inverted transformation matrix
        T_inv = np.eye(4)
        T_inv[:3, :3] = RS_inv
        T_inv[:3, 3] = t_inv
        
        return T_inv

    def _filter_rotation(self, T, point_cloud, x_thresh=0.5, y_thresh=0.5, z_thresh=0.5):
        """
        Check if the rotation matrix exceeds specified thresholds for x, y, z rotations.

        Parameters:
        matrix (np.ndarray): 3x3 rotation matrix.
        x_thresh (float): Threshold for rotation about the x-axis (roll) in degrees.
        y_thresh (float): Threshold for rotation about the y-axis (pitch) in degrees.
        z_thresh (float): Threshold for rotation about the z-axis (yaw) in degrees.

        Returns:
        bool: True if all rotations are within thresholds, False otherwise.
        """
        # Validate the input is a proper rotation matrix
        # if not (matrix.shape == (3, 3) and np.allclose(np.dot(matrix.T, matrix), np.eye(3), atol=1e-6)):
        #     raise ValueError("Input must be a valid 3x3 rotation matrix.")
        
        # import pdb; pdb.set_trace()
        
        mean_point = np.mean(point_cloud, axis=0)
        T_camtomean = np.eye(4)
        T_camtomean[:3, 3] = -mean_point
        T_meantocam = self._invert_transformation(T_camtomean)

        # Calculate the transformation that would be applied to the mean centered point cloud
        T_mean = T @ T_meantocam

        # Get the rotation matrix from it:
        r_matrix = T_mean[:3, :3]

        # Extract Euler angles using scipy's Rotation class
        r = R.from_matrix(r_matrix)
        roll, pitch, yaw = r.as_euler('xyz', degrees=True)  # Roll, pitch, yaw in degrees

        # Clip the angles to the thresholds
        roll_clipped = np.clip(roll, -x_thresh, x_thresh)
        pitch_clipped = np.clip(pitch, -y_thresh, y_thresh)
        yaw_clipped = np.clip(yaw, -z_thresh, z_thresh)

        # Reconstruct the rotation matrix from the clipped angles
        clipped_rotation = R.from_euler('xyz', [roll_clipped, pitch_clipped, yaw_clipped], degrees=True)
        r_matrix_clipped = clipped_rotation.as_matrix()

        # Replace the rotation part of the transformation matrix with the clipped rotation
        T_mean_clipped = T_mean.copy()
        T_mean_clipped[:3, :3] = r_matrix_clipped

        # Convert the clipped transformation back to the camera frame
        T_clipped = T_mean_clipped @ T_camtomean
        
        return T_clipped

    def get_point_cloud_and_seg(self, depth, K, seg):

        # Get the dimensions of the depth image
        height, width = depth.shape

        # Create a meshgrid of pixel coordinates
        u, v = np.meshgrid(np.arange(width), np.arange(height))

        # Flatten the arrays
        u = u.flatten()
        v = v.flatten()
        depth = depth.flatten()
        seg = seg.flatten()

        # Filter out points with zero depth
        valid = (depth > 0) & (depth <= self.vis_threshold)
        u = u[valid]
        v = v[valid]
        depth = depth[valid]
        seg = seg[valid]

        # Compute the 3D coordinates
        x = (u - K[0, 2]) * depth / K[0, 0]
        y = (v - K[1, 2]) * depth / K[1, 1]
        z = depth

        # Stack the coordinates to form the point cloud
        point_cloud = np.stack((x, y, z), axis=-1)

        # Create the segmented point cloud
        seg = seg.astype(int)  # Convert to integer type
        segmented_point_cloud = seg - 1 # Subtract 1 to make the object ids start from 0

        # Save the point cloud and segmented point cloud
        if self.debug:
            np.save(f"{self.output_dir}/point_cloud.npy", point_cloud)
            np.save(f"{self.output_dir}/segmented_point_cloud.npy", segmented_point_cloud)

        return point_cloud, segmented_point_cloud
    
    def get_taxposed_input(self, point_cloud, segmented_point_cloud, action_obj=0):
        # Get the action and anchor indexes
        action_idx = np.where(segmented_point_cloud == action_obj)[
            0
        ]  # This is the target object's point cloud
        anchor_idx = np.where(segmented_point_cloud != action_obj)[
            0
        ]  # The remaining environment is the anchor

        # Downsample to the min between the number of points and 4*1024
        action_idx = np.random.choice(action_idx, min(len(action_idx), 4 * 1024))
        anchor_idx = np.random.choice(anchor_idx, min(len(anchor_idx), 4 * 1024))

        action_pcd, action_mask = point_cloud[action_idx], segmented_point_cloud[action_idx]
        anchor_pcd, anchor_mask = point_cloud[anchor_idx], segmented_point_cloud[anchor_idx]

        taxposed_input = {
            "clouds": np.concatenate([action_pcd, anchor_pcd]),
            "classes": np.concatenate(
                [np.zeros(len(action_pcd)), np.ones(len(anchor_pcd))]
            ),
            "masks": np.concatenate([action_mask, anchor_mask]),
        }
        if self.debug:
            idx = 10
            np.savez(self.output_dir + f"/{idx}_{self.cfg.model.cloud_type}_obj_points.npz", clouds=taxposed_input["clouds"], classes=taxposed_input["classes"], masks=taxposed_input["masks"])

        return taxposed_input
    
    def get_data_loader(self, data):
        return DataLoader(
            TestPointCloudDataset(
                data,
                action_class=0,
                anchor_class=1,
                num_suggestions=self.num_suggestions,
                ),
            batch_size=1,
            shuffle=False,
        )

    def predict(self, point_cloud: np.ndarray, segmented_point_cloud: np.ndarray, action_obj: int=0) -> list:
        """
        Input:
        pcd => point cloud, np.ndarray of size (n, 3)
        pcd_seg => segmentation ids , np.ndarray of size (n,)
        action_obj => object id to be moved, int

        Output:
        suggested_transforms => list of np.ndarrays of shape (4,4)
        """


        taxposed_data = self.get_taxposed_input(point_cloud, segmented_point_cloud, action_obj)
        dataloader = self.get_data_loader(taxposed_data)
        data = next(iter(dataloader))

        action_pcd = data["points_action_trans"]
        anchor_pcd = data["points_anchor_trans"]

        preds = self.model.get_transform(
            action_pcd,  # Downsampled point cloud of shape (1, 1024, 3) obtained from DataLoader
            anchor_pcd,  # Downsampled point cloud of shape (1, 1024, 3) obtained from DataLoader
            n_samples=self.num_suggestions,
            sampling_method=self.model_cfg.sampling_method,
        )

        # preds => list of 3 suggestions. Each one has keys ['pred_T_action', 'pred_points_action', 'flow_components']

        mean_point = data["mean_point"].squeeze(0).numpy()
        T0 = data["T0"].cpu().squeeze(0).detach().numpy()
        T1 = data["T1"].cpu().squeeze(0).detach().numpy()

        # undo the normalization performed prior evaluation
        Normalize = np.eye(4)
        Normalize[:3, 3] = -mean_point
        Normalize_inv = np.eye(4)
        Normalize_inv[:3, 3] = mean_point

        # Transform to the original frame
        T1_inv = T1.T
        T1_inv[:3, :3] = T1_inv[:3, :3].T
        T1_inv[:3, 3] = -T1_inv[:3, :3] @ T1_inv[:3, 3]

        suggested_transforms = []

        for pred in preds:
            # from equivariant_pose_graph.utils.visualizations import plot_taxposed_embeddings
            # pred["flow_components"]["pred_points_action"] = pred["pred_points_action"]
            # plot_taxposed_embeddings(data["points_action_trans"],
            #     data["points_anchor_trans"],
            #     pred["flow_components"], None)
            # breakpoint()

            T = pred["pred_T_action"].cpu().get_matrix().squeeze(0).detach().numpy()
            T = T1_inv @ T.T @ T0.T

            # Normalize the transformation
            T = Normalize_inv @ T @ Normalize

            full_action_pcd = point_cloud[segmented_point_cloud == action_obj]

            # Filter the transformation with a rotation threshold:
            T = self._filter_rotation(T, full_action_pcd,
                              x_thresh=self.cfg.model.x_thresh,
                              y_thresh=self.cfg.model.y_thresh,
                              z_thresh=self.cfg.model.z_thresh)
            
            # TODO: For predictions below the plane, we need to clip the z value to be above the plane.
            # T_world = self.T_cam_to_world @ T
            # T_world[2, 3] = np.clip(T_world[2, 3], self.cfg.model.min_height, None)
            # T = self.T_world_to_cam @ T_world

            suggested_transforms.append(T)
            if len(suggested_transforms) == self.num_suggestions:
                break

        return suggested_transforms
        

    def run(self):
        # Load and Extract the input data
        data = np.load(self.input_path, allow_pickle=True).item()
        rgb, depth, K = data["rgb"], data["depth"], data["K"]
        if self.gsam2_pred_path:
            seg = self._load_segmentation_from_gsam2(self.gsam2_pred_path)
        elif "seg" in data:
            seg = data["seg"]
        else:
            raise ValueError("Segmentation ids not found in the files provided.")

        # Get the point cloud and segmented point cloud
        point_cloud, segmented_point_cloud = self.get_point_cloud_and_seg(depth, K, seg)

        # Predict the transformation
        suggested_transforms = self.predict(point_cloud, segmented_point_cloud)
        
        # Save the suggested transforms
        np.save(f"{self.output_dir}/taxposed_prediction.npy", suggested_transforms)

        print(suggested_transforms)
        print("Finished processing the input data.")
        return suggested_transforms


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process Contact-GraspNet input data.')
    parser.add_argument('--output_dir', required=True, help='Directory to save the processed output')
    parser.add_argument('--input_path', required=True, help='Input file for npy file containing rgb, depth, K (camera intrinsics), and (optional) segmentation with object ids')
    parser.add_argument('--gsam2_pred_path', required=False, help='Specify if input does not contains segmentation ids')
    parser.add_argument('--cfg', required=True, help='Path to the config file for the model')
    
    args = parser.parse_args()

    with hydra.initialize(version_base=None, config_path="configs/inference"):
        cfg = hydra.compose(args.cfg)

    taxposed_input_preprocessing = TaxposedWrapper(output_dir=args.output_dir,
                                                   input_path=args.input_path,
                                                   gsam2_pred_path=args.gsam2_pred_path,
                                                   cfg=cfg,
                                                   debug=True)
    
    taxposed_input_preprocessing.run()