"""Generate KITTI-format label files from images using the two-stage 3D detector.

This script runs the trained model on one or more images and writes
`label_2`-style text files with 3D boxes in KITTI format.

Usage example (regular KITTI-like camera):
    python generate_kitti_labels_from_images.py \
        --checkpoint outputs/two_stage/checkpoints/checkpoint_best.pth \
        --input path/to/image_or_dir \
        --output-dir outputs/generated_labels

Usage example (fisheye camera with JSON config, undistorted internally):
    python generate_kitti_labels_from_images.py \
        --checkpoint outputs/two_stage/checkpoints/checkpoint_best.pth \
        --input path/to/image_or_dir \
        --output-dir outputs/generated_labels \
        --fisheye-config path/to/fisheye_config.json
"""

import os
import sys
import argparse
from pathlib import Path
import json

import torch
import numpy as np
import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from models.two_stage_detector import build_model
from fisheye_utils import FisheyeCamera


class LabelGenerationEngine:
    """
    Inference engine that produces 3D boxes suitable for KITTI label export.

    By default it behaves similarly to `ImageInferenceEngine` from
    `inference_images_two_stage.py`, but exposes everything needed to
    write KITTI-style labels.
    """

    def __init__(
        self,
        checkpoint_path,
        fisheye_config=None,
        device="cuda",
        use_intrinsics_normalization=True,
        dim_scale=None,
    ):
        """
        Args:
            checkpoint_path: Path to model checkpoint (.pth).
            fisheye_config: Dict or path to fisheye camera config (None for KITTI camera).
            device: 'cuda' or 'cpu'.
            use_intrinsics_normalization: Normalize intrinsics to KITTI fx/width ratio.
            dim_scale: Optional scaling factor for predicted dimensions.
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.use_fisheye = fisheye_config is not None
        self.use_intrinsics_normalization = use_intrinsics_normalization
        self.dim_scale = dim_scale

        # ------------------------------------------------------------------
        # Camera / intrinsics setup
        # ------------------------------------------------------------------
        if self.use_fisheye:
            print("\nInitializing fisheye camera...")
            if isinstance(fisheye_config, str):
                with open(fisheye_config, "r") as f:
                    fisheye_config = json.load(f)

            self.fisheye_camera = FisheyeCamera(fisheye_config)
            self.intrinsics_original = self.fisheye_camera.get_undistorted_intrinsics()
            self.image_shape = (fisheye_config["height"], fisheye_config["width"])
        else:
            print("\nUsing regular KITTI-like camera (no fisheye correction)")
            # Default KITTI intrinsics and image shape
            self.intrinsics_original = np.array(
                [
                    [721.5377, 0, 609.5593],
                    [0, 721.5377, 172.854],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )
            self.image_shape = (375, 1242)

        if use_intrinsics_normalization:
            print("\nNormalizing intrinsics to KITTI scale...")
            self.intrinsics_normalized = self.normalize_intrinsics_to_kitti(
                self.intrinsics_original,
                self.image_shape,
            )
            print(f"  Original fx:   {self.intrinsics_original[0, 0]:.1f}")
            print(f"  Normalized fx: {self.intrinsics_normalized[0, 0]:.1f}")
            print(
                f"  Scale factor:  "
                f"{self.intrinsics_normalized[0, 0] / self.intrinsics_original[0, 0]:.3f}"
            )

            self.intrinsics = self.intrinsics_normalized
        else:
            self.intrinsics = self.intrinsics_original

        # ------------------------------------------------------------------
        # Model
        # ------------------------------------------------------------------
        print(f"\nLoading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        if "classes" in checkpoint:
            self.classes = checkpoint["classes"]
            print(f"Loaded trained classes: {self.classes}")
        else:
            self.classes = ["Car"]
            print(f"Warning: Using default classes: {self.classes}")

        self.model = build_model(active_classes=self.classes)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"✓ Model loaded from epoch {checkpoint['epoch']}")
        if "best_val_loss" in checkpoint:
            print(f"  Best val loss: {checkpoint['best_val_loss']:.4f}")

        # ImageNet normalization params
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        if dim_scale is not None:
            print(f"\nUsing dimension scaling: {dim_scale}")

    def normalize_intrinsics_to_kitti(self, K_yours, image_shape):
        """Normalize intrinsics so fx / image_width matches KITTI."""
        fx_kitti = 721.5377
        width_kitti = 1242

        fx_yours = K_yours[0, 0]
        height_yours, width_yours = image_shape

        ratio_kitti = fx_kitti / width_kitti
        ratio_yours = fx_yours / width_yours
        scale_factor = ratio_kitti / ratio_yours

        K_normalized = K_yours.copy()
        K_normalized[0, 0] *= scale_factor
        K_normalized[1, 1] *= scale_factor
        K_normalized[0, 2] *= scale_factor
        K_normalized[1, 2] *= scale_factor

        return K_normalized

    def preprocess_image(self, image):
        """Preprocess image for model input; returns tensor and scale factors."""
        # Undistort if fisheye
        if self.use_fisheye:
            image = self.fisheye_camera.undistort(image)

        image_original = image.copy()
        h_orig, w_orig = image.shape[:2]

        # Resize to model input size (same as training)
        target_h, target_w = 384, 1280
        image_resized = cv2.resize(image, (target_w, target_h))

        # Normalize
        image_norm = image_resized.astype(np.float32) / 255.0
        image_norm = (image_norm - self.mean) / self.std

        # To tensor
        image_tensor = torch.from_numpy(image_norm.transpose(2, 0, 1)).unsqueeze(0).float()

        # Scale factors from model input back to original resolution
        scale_x = w_orig / target_w
        scale_y = h_orig / target_h

        return image_tensor, image_original, (scale_x, scale_y)

    @torch.no_grad()
    def predict_image(self, image):
        """
        Run inference on a single image.

        Returns:
            boxes_3d: (N, 7) ndarray [x_center, y_center, z, h, w, l, ry]
            boxes_2d: (N, 4) ndarray [x1, y1, x2, y2] in original image coords
            scores:   (N,) ndarray
            class_names: list[str] of length N
            image_processed: RGB image after any undistortion
        """
        image_tensor, image_processed, scales = self.preprocess_image(image)
        image_tensor = image_tensor.to(self.device)

        # Scale intrinsics to model input size
        K_scaled = self.intrinsics.copy()
        scale_x, scale_y = scales
        K_scaled[0, 0] /= scale_x
        K_scaled[1, 1] /= scale_y
        K_scaled[0, 2] /= scale_x
        K_scaled[1, 2] /= scale_y

        intrinsics_tensor = torch.from_numpy(K_scaled).float().to(self.device)

        # Inference
        predictions = self.model(
            image_tensor,
            intrinsics_tensor.unsqueeze(0),
            gt_boxes_2d=None,
        )

        boxes_3d, boxes_2d, scores, class_names = self.decode_predictions(predictions, scales)
        return boxes_3d, boxes_2d, scores, class_names, image_processed

    def decode_predictions(self, predictions, scales):
        """
        Decode raw model outputs into 3D boxes and 2D boxes (original image coords).
        """
        boxes_2d = predictions["boxes_2d"][0]
        depth_pred = predictions["depth"][0]
        dims_pred = predictions["dimensions"][0]
        rot_bins, rot_res = predictions["rotation"][0]
        scores = predictions["scores_2d"][0]
        class_names = predictions["classes"][0]

        if len(boxes_2d) == 0:
            return (
                np.zeros((0, 7), dtype=np.float32),
                np.zeros((0, 4), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                [],
            )

        # Depth with learned offset
        depth = depth_pred[:, 0].cpu().numpy()
        depth_offset = depth_pred[:, 2].cpu().numpy()
        depth = depth + depth_offset

        # Rotation
        rot_bin_idx = torch.argmax(rot_bins, dim=1).cpu().numpy()
        bin_size = 2 * np.pi / 12
        rotation = (rot_bin_idx + 0.5) * bin_size
        rot_res_selected = rot_res[torch.arange(len(rot_bin_idx)), torch.from_numpy(rot_bin_idx)]
        rotation = rotation + rot_res_selected.cpu().numpy()
        rotation = (rotation + np.pi) % (2 * np.pi) - np.pi

        # Dimensions
        dims = dims_pred.cpu().numpy()
        if self.dim_scale is not None:
            dims = dims * self.dim_scale

        # Use (possibly normalized) intrinsics for unprojection
        scale_x, scale_y = scales
        K = self.intrinsics.copy()

        # Scale 2D boxes back to original resolution
        boxes_2d_np = boxes_2d.cpu().numpy()
        boxes_2d_np[:, [0, 2]] *= scale_x
        boxes_2d_np[:, [1, 3]] *= scale_y

        # Bottom-center unprojection in camera frame
        x_2d = (boxes_2d_np[:, 0] + boxes_2d_np[:, 2]) / 2
        y_2d = boxes_2d_np[:, 3]  # bottom edge

        x_cam = (x_2d - K[0, 2]) * depth / K[0, 0]
        y_cam_bottom = (y_2d - K[1, 2]) * depth / K[1, 1]
        z_cam = depth

        # Geometric center: shift up by h/2 (dims[:, 0] is height)
        y_cam_center = y_cam_bottom - dims[:, 0] / 2.0

        boxes_3d = np.stack(
            [x_cam, y_cam_center, z_cam, dims[:, 0], dims[:, 1], dims[:, 2], rotation],
            axis=1,
        ).astype(np.float32)

        scores_np = scores.cpu().numpy().astype(np.float32)

        return boxes_3d, boxes_2d_np.astype(np.float32), scores_np, class_names


def kitti_alpha_from_box(x, z, ry):
    """
    Approximate KITTI observation angle alpha from 3D location and rotation_y.

    KITTI defines:
        alpha = ry - atan2(x, z)
    """
    return float(ry - np.arctan2(x, z))


def write_kitti_labels(
    image_path,
    boxes_3d,
    boxes_2d,
    scores,
    class_names,
    output_dir,
):
    """
    Write a KITTI-format label file for one image.

    Each line:
        type truncated occluded alpha
        bbox_left bbox_top bbox_right bbox_bottom
        h w l
        x y z
        rotation_y

    Note: This matches KITTI's training label_2 format (no score column).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(image_path).stem
    label_path = output_dir / f"{stem}.txt"

    with open(label_path, "w") as f:
        for box_3d, box_2d, score, cls in zip(
            boxes_3d, boxes_2d, scores, class_names
        ):
            x_center, y_center, z, h, w, l, ry = box_3d.astype(float)
            x1, y1, x2, y2 = box_2d.astype(float)

            # Convert geometric center back to KITTI's bottom-center location
            y_bottom = y_center + h / 2.0

            # Truncation/occlusion are unknown for predictions; set to 0
            truncated = 0.0
            occluded = 0

            # Approximate observation angle alpha
            alpha = kitti_alpha_from_box(x_center, z, ry)

            # Format line
            line = (
                f"{cls} "
                f"{truncated:.2f} {occluded:d} {alpha:.4f} "
                f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} "
                f"{h:.4f} {w:.4f} {l:.4f} "
                f"{x_center:.4f} {y_bottom:.4f} {z:.4f} "
                f"{ry:.4f}"
            )
            f.write(line + "\n")

    print(f"Wrote labels: {label_path}")


def collect_image_paths(input_path):
    """Return list of image paths given a file or directory."""
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        exts = ["*.png", "*.jpg", "*.jpeg"]
        paths = []
        for pat in exts:
            paths.extend(sorted(input_path.glob(pat)))
        return paths
    raise ValueError(f"Input not found: {input_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate KITTI-format labels from images using the two-stage 3D detector",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth)",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input image or directory of images",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save KITTI-style label files",
    )
    parser.add_argument(
        "--fisheye-config",
        type=str,
        default=None,
        help="Path to fisheye JSON config (optional; if omitted, uses KITTI intrinsics)",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for keeping detections",
    )
    parser.add_argument(
        "--no-intrinsics-norm",
        action="store_true",
        help="Disable intrinsics normalization to KITTI scale",
    )
    parser.add_argument(
        "--dim-scale",
        type=float,
        default=None,
        help="Optional dimension scaling factor (e.g., 0.3) to tune box sizes",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device: cuda or cpu",
    )

    args = parser.parse_args()

    # Load fisheye config if provided
    fisheye_config = None
    if args.fisheye_config:
        with open(args.fisheye_config, "r") as f:
            fisheye_config = json.load(f)
        print(f"Loaded fisheye config from {args.fisheye_config}")

    # Build engine
    engine = LabelGenerationEngine(
        checkpoint_path=args.checkpoint,
        fisheye_config=fisheye_config,
        device=args.device,
        use_intrinsics_normalization=not args.no_intrinsics_norm,
        dim_scale=args.dim_scale,
    )

    # Collect images
    image_paths = collect_image_paths(args.input)
    print(f"\nFound {len(image_paths)} images")

    # Process images
    for img_path in tqdm(image_paths, desc="Generating labels"):
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Warning: Could not read image: {img_path}")
            continue

        boxes_3d, boxes_2d, scores, class_names, image_processed = engine.predict_image(
            image
        )

        # Confidence filtering
        keep_mask = scores >= args.conf_threshold
        boxes_3d = boxes_3d[keep_mask]
        boxes_2d = boxes_2d[keep_mask]
        scores_filtered = scores[keep_mask]
        class_names_filtered = [
            class_names[i] for i in range(len(class_names)) if keep_mask[i]
        ]

        if len(boxes_3d) == 0:
            # Still write an empty file to indicate no objects
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(img_path).stem
            label_path = out_dir / f"{stem}.txt"
            open(label_path, "w").close()
            continue

        write_kitti_labels(
            img_path,
            boxes_3d,
            boxes_2d,
            scores_filtered,
            class_names_filtered,
            args.output_dir,
        )


if __name__ == "__main__":
    main()


