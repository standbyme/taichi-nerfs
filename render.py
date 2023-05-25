import argparse
import os
import random
import time

import cv2
import torch
import numpy as np
import taichi as ti
from einops import rearrange
from scipy.spatial.transform import Rotation as R

from datasets import dataset_dict
from datasets.ray_utils import get_ray_directions, get_rays

from modules.networks import NGP
from modules.rendering import render


@ti.kernel
def write_buffer(
        W: ti.i32, H: ti.i32, x: ti.types.ndarray(), final_pixel: ti.template()
):
    for i, j in ti.ndrange(W, H):
        for p in ti.static(range(3)):
            final_pixel[i, j][p] = x[H - j, i, p]


class OrbitCamera:
    def __init__(self, K, img_wh, poses, r):
        self.K = K
        self.W, self.H = img_wh
        self.radius = r
        self.center = np.zeros(3)

        pose_np = poses.cpu().numpy()
        self.rot = pose_np[0][:3, :3]

        self.rotate_speed = 0.8
        self.res_defalut = pose_np[0]

    @property
    def pose(self):
        res = np.eye(4)
        res[2, 3] -= self.radius
        rot = np.eye(4)
        rot[:3, :3] = self.rot
        res = rot @ res
        res[:3, 3] -= self.center
        return res

    def reset(self, pose=None):
        self.rot = np.eye(3)
        self.center = np.zeros(3)
        self.radius = 2.0
        if pose is not None:
            self.rot = pose.cpu().numpy()[:3, :3]

    def orbit(self, dx, dy):
        rotvec_x = self.rot[:, 1] * np.radians(100 * self.rotate_speed * dx)
        rotvec_y = self.rot[:, 0] * np.radians(-100 * self.rotate_speed * dy)
        self.rot = (
                R.from_rotvec(rotvec_y).as_matrix()
                @ R.from_rotvec(rotvec_x).as_matrix()
                @ self.rot
        )

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        self.center += 1e-4 * self.rot @ np.array([dx, dy, dz])


class NeRFGUI:
    def __init__(self, hparams, model_config, K, img_wh, poses, radius=4.5):
        self.hparams = hparams
        self.model = NGP(**model_config).cuda()

        print(f"loading ckpt from: {hparams.ckpt_path}")
        state_dict = torch.load(hparams.ckpt_path)
        self.model.load_state_dict(state_dict)

        self.poses = poses

        self.cam = OrbitCamera(K, img_wh, poses, r=radius)
        self.W, self.H = img_wh
        self.render_buffer = ti.Vector.field(n=3, dtype=float, shape=(self.W, self.H))

        if self.hparams.dataset_name in ["colmap", "nerfpp"]:
            self.exp_step_factor = 1 / 256
        else:
            self.exp_step_factor = 0

        # placeholders
        self.dt = 0
        self.mean_samples = 0
        self.img_mode = 0

    def render_cam(self):
        t = time.time()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            directions = get_ray_directions(
                self.cam.H, self.cam.W, self.cam.K, device="cuda"
            )
            rays_o, rays_d = get_rays(directions, torch.cuda.FloatTensor(self.cam.pose))
            results = render(
                self.model,
                rays_o,
                rays_d,
                test_time=True,
                exp_step_factor=self.exp_step_factor,
            )

        rgb = rearrange(results["rgb"], "(h w) c -> h w c", h=self.H)
        depth = rearrange(results["depth"], "(h w) -> h w", h=self.H)
        self.dt = time.time() - t
        self.mean_samples = results["total_samples"] / len(rays_o)

        if self.img_mode == 0:
            return rgb
        assert self.img_mode == 1
        return depth2img(depth.cpu().numpy()).astype(np.float32) / 255.0

    def check_cam_rotate(self, window, last_orbit_x, last_orbit_y):
        if window.is_pressed(ti.ui.RMB):
            curr_mouse_x, curr_mouse_y = window.get_cursor_pos()
            if last_orbit_x is None or last_orbit_y is None:
                last_orbit_x, last_orbit_y = curr_mouse_x, curr_mouse_y
            else:
                dx = curr_mouse_x - last_orbit_x
                dy = curr_mouse_y - last_orbit_y
                self.cam.orbit(dx, -dy)
                last_orbit_x, last_orbit_y = curr_mouse_x, curr_mouse_y
        else:
            last_orbit_x = None
            last_orbit_y = None

        return last_orbit_x, last_orbit_y

    def render(self):
        window = ti.ui.Window(
            "Hongyu Cai: Neural Radiance Field",
            (self.W, self.H),
        )
        canvas = window.get_canvas()

        # GUI controls variables
        last_orbit_x = None
        last_orbit_y = None

        while window.running:
            last_orbit_x, last_orbit_y = self.check_cam_rotate(
                window, last_orbit_x, last_orbit_y
            )

            ngp_buffer = self.render_cam()
            write_buffer(self.W, self.H, ngp_buffer, self.render_buffer)
            canvas.set_image(self.render_buffer)
            window.show()


def get_arg(prefix_args=None):
    parser = argparse.ArgumentParser()

    # dataset parameters
    parser.add_argument('--root_dir',
                        type=str,
                        required=True,
                        help='root directory of dataset')
    parser.add_argument('--dataset_name',
                        type=str,
                        default='nsvf',
                        choices=['nerf', 'nsvf', 'colmap', 'ngp'],
                        help='which dataset to train/test')
    parser.add_argument('--split',
                        type=str,
                        default='train',
                        choices=['train', 'trainval', 'trainvaltest'],
                        help='use which split to train')
    parser.add_argument('--downsample',
                        type=float,
                        default=1.0,
                        help='downsample factor (<=1.0) for the images')

    # model parameters
    parser.add_argument(
        '--scale',
        type=float,
        default=0.5,
        help='scene scale (whole scene must lie in [-scale, scale]^3')

    parser.add_argument('--half_opt',
                        action='store_true',
                        default=False,
                        help='whether to use half optimization')

    parser.add_argument('--encoder_type',
                        type=str,
                        default='hash',
                        choices=['hash', 'triplane'],
                        help='which encoder to use')

    # loss parameters
    parser.add_argument('--distortion_loss_w',
                        type=float,
                        default=0,
                        help='''weight of distortion loss (see losses.py),
                        0 to disable (default), to enable,
                        a good value is 1e-3 for real scene and 1e-2 for synthetic scene
                        ''')

    # training options
    parser.add_argument('--batch_size',
                        type=int,
                        default=8192,
                        help='number of rays in a batch')
    parser.add_argument('--ray_sampling_strategy',
                        type=str,
                        default='all_images',
                        choices=['all_images', 'same_image'],
                        help='''
                        all_images: uniformly from all pixels of ALL images
                        same_image: uniformly from all pixels of a SAME image
                        ''')
    parser.add_argument('--max_steps',
                        type=int,
                        default=20000,
                        help='number of steps to train')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument(
        '--random_bg',
        action='store_true',
        default=False,
        help='''whether to train with random bg color (real scene only)
                        to avoid objects with black color to be predicted as transparent
                        ''')
    # misc
    parser.add_argument('--exp_name',
                        type=str,
                        default='exp',
                        help='experiment name')
    parser.add_argument(
        '--ckpt_path',
        type=str,
        default=None,
        help='pretrained checkpoint to load (including optimizers, etc)')

    parser.add_argument(
        '--gui',
        action='store_true',
        default=False,
        help='whether to show interactive GUI after training is done'
    )
    # use deployment or not
    parser.add_argument('--deployment', action='store_true', default=False)
    parser.add_argument('--deployment_model_path', type=str, default="./")

    return parser.parse_args(prefix_args)


def main():
    seed = 23
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    hparams = get_arg()

    val_dir = "results/"
    assert os.path.exists(val_dir)
    hparams.ckpt_path = os.path.join(val_dir, "model.pth")

    dataset = dataset_dict[hparams.dataset_name](
        root_dir=hparams.root_dir,
        downsample=hparams.downsample,
        read_meta=True,
    )
    model_config = {
        "scale": hparams.scale,
        "pos_encoder_type": hparams.encoder_type,
        "max_res": 1024 if hparams.scale == 0.5 else 4096,
        "half_opt": hparams.half_opt,
    }

    ti.init(arch=ti.cuda)

    NeRFGUI(hparams, model_config, dataset.K, dataset.img_wh, dataset.poses).render()


def depth2img(depth):
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth_img = cv2.applyColorMap((depth * 255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)

    return depth_img


if __name__ == "__main__":
    main()
