import torch
import numpy as np
from equivariant_pose_graph.utils.se3 import random_se3
from pytorch3d.ops import sample_farthest_points
from torch.utils.data import Dataset

class TestPointCloudDataset(Dataset):

    # TODO: Compress this class into the suggester class. It need not be seperate just for a single batch

    def __init__(
        self,
        point_data,
        cloud_type="final",
        action_class=0,
        anchor_class=1,
        pzY_input_dims=3,
        num_points=1024,
        rotation_variance=np.pi,
        translation_variance=0.5,
        symmetric_class=None,
        angle_degree=180,
        downsample_type="fps",
        action_rot_sample_method="quat_uniform",
        anchor_rot_sample_method="random_flat_upright",
        num_suggestions=4,
        dataset_size=1,
        seed=0,
    ):
        self.num_points = num_points
        self.point_data = point_data
        self.pzY_input_dims = pzY_input_dims
        # Path('/home/bokorn/src/ndf_robot/notebooks')
        self.cloud_type = cloud_type
        self.cloud_type_init = "init"
        self.rot_var = rotation_variance
        self.trans_var = translation_variance
        self.action_class = action_class
        self.anchor_class = anchor_class
        self.symmetric_class = symmetric_class  # None if no symmetric class exists
        self.angle_degree = angle_degree
        self.action_rot_sample_method = action_rot_sample_method
        self.anchor_rot_sample_method = anchor_rot_sample_method
        self.downsample_type = downsample_type

        self.num_suggestions = num_suggestions
        self.dataset_size = dataset_size
        self.seed = seed

    def __getitem__(self, idx):
        return self.get_data(idx + self.seed)

    def __len__(self):
        # we can set this to any number since we only return random samples of
        # the data by using a random seed
        return max(self.num_suggestions, self.dataset_size)

    def downsample_pcd(self, points, type="fps"):
        points = points.unsqueeze(0)
        if type == "fps":
            return sample_farthest_points(
                points, K=self.num_points, random_start_point=True
            )
        elif type == "random":
            random_idx = torch.randperm(points.shape[1])[: self.num_points]
            return points[:, random_idx], random_idx
        elif type.startswith("random_"):
            prob = float(type.split("_")[1])
            if np.random.random() < prob:
                return sample_farthest_points(
                    points, K=self.num_points, random_start_point=True
                )
            else:
                random_idx = torch.randperm(points.shape[1])[: self.num_points]
                return points[:, random_idx], random_idx

    def load_data(self, point_data, action_class, anchor_class):
        points_raw_np = point_data["clouds"]
        classes_raw_np = point_data["classes"]
        masks_raw_np = point_data["masks"]

        points_action_np = points_raw_np[classes_raw_np == action_class].copy()
        points_action_mean_np = points_action_np.mean(axis=0)
        points_action_np = points_action_np - points_action_mean_np

        points_anchor_np = points_raw_np[classes_raw_np == anchor_class].copy()
        points_anchor_np = points_anchor_np - points_action_mean_np
        points_anchor_mean_np = points_anchor_np.mean(axis=0)

        mask_action_np = masks_raw_np[classes_raw_np == action_class].copy()
        mask_anchor_np = masks_raw_np[classes_raw_np == anchor_class].copy()

        points_action = torch.from_numpy(points_action_np).float()
        points_anchor = torch.from_numpy(points_anchor_np).float()
        mask_action = torch.from_numpy(mask_action_np).float().unsqueeze(1)
        mask_anchor = torch.from_numpy(mask_anchor_np).float().unsqueeze(1)

        points_action = torch.cat([points_action, mask_action], dim=1)
        points_anchor = torch.cat([points_anchor, mask_anchor], dim=1)

        symmetric_cls = torch.Tensor([])

        return points_action, points_anchor, symmetric_cls, points_action_mean_np

    def get_data(self, seed=0):
        points_action, points_anchor, symmetric_cls, mean_point = self.load_data(
            self.point_data, self.action_class, self.anchor_class
        )
        torch.manual_seed(seed)
        points_action, _ = self.downsample_pcd(points_action, type=self.downsample_type)
        points_anchor, _ = self.downsample_pcd(points_anchor, type=self.downsample_type)

        points_action, mask_action = points_action[:, :, :3], points_action[:, :, 3:]
        points_anchor, mask_anchor = points_anchor[:, :, :3], points_anchor[:, :, 3:]

        T0 = random_se3(
            1,
            rot_var=self.rot_var,
            trans_var=self.trans_var,
            device=points_action.device,
            rot_sample_method=self.action_rot_sample_method,
        )
        T1 = random_se3(
            1,
            rot_var=self.rot_var,
            trans_var=self.trans_var,
            device=points_anchor.device,
            rot_sample_method=self.anchor_rot_sample_method,
        )

        points_action_trans = T0.transform_points(points_action)
        points_anchor_trans = T1.transform_points(points_anchor)

        points_action = torch.cat([points_action, mask_action], axis=-1)[
            :, :, : self.pzY_input_dims
        ]
        points_anchor = torch.cat([points_anchor, mask_anchor], axis=-1)[
            :, :, : self.pzY_input_dims
        ]
        points_action_trans = torch.cat([points_action_trans, mask_action], axis=-1)[
            :, :, : self.pzY_input_dims
        ]
        points_anchor_trans = torch.cat([points_anchor_trans, mask_anchor], axis=-1)[
            :, :, : self.pzY_input_dims
        ]

        data = {
            "points_action": points_action.squeeze(0),
            "points_anchor": points_anchor.squeeze(0),
            "points_action_trans": points_action_trans.cuda().squeeze(0),
            "points_anchor_trans": points_anchor_trans.cuda().squeeze(0),
            "T0": T0.get_matrix().squeeze(0),
            "T1": T1.get_matrix().squeeze(0),
            "symmetric_cls": symmetric_cls,
            "mean_point": mean_point,
        }

        return data


