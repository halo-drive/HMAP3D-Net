"""Fisheye Camera Utilities for Entron F008A GMSL Camera"""

import numpy as np
import cv2


class FisheyeCamera:
    """Handle fisheye camera undistortion and intrinsics"""
    
    def __init__(self, config):
        """
        Args:
            config (dict): Camera configuration with keys:
                - fx, fy: focal lengths
                - cx, cy: principal point
                - width, height: image dimensions
                - k1, k2, k3, k4: distortion coefficients (optional)
        """
        self.fx = config['fx']
        self.fy = config['fy']
        self.cx = config['cx']
        self.cy = config['cy']
        self.width = config['width']
        self.height = config['height']
        
        # Fisheye intrinsics matrix
        self.K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Fisheye distortion coefficients
        # If not provided, use typical values for fisheye
        if 'k1' in config and 'k2' in config:
            self.D = np.array([
                config.get('k1', 0.0),
                config.get('k2', 0.0),
                config.get('k3', 0.0),
                config.get('k4', 0.0)
            ], dtype=np.float32)
        else:
            # Default fisheye distortion (you may need to calibrate for exact values)
            print("Warning: Using estimated fisheye distortion coefficients")
            print("For best results, calibrate your camera using a checkerboard pattern")
            self.D = np.array([-0.15, 0.02, -0.005, 0.001], dtype=np.float32)
        
        # Prepare undistortion maps
        self.undistort_maps = None
        self.new_K = None
        self._prepare_undistortion()
    
    def _prepare_undistortion(self):
        """Prepare undistortion maps for efficient processing"""
        # Estimate new camera matrix for undistorted image
        self.new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self.K, 
            self.D,
            (self.width, self.height),
            np.eye(3),
            balance=0.0,  # 0 = maximize view, 1 = minimize invalid pixels
            new_size=(self.width, self.height)
        )
        
        # Compute undistortion maps
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            self.K, 
            self.D, 
            np.eye(3), 
            self.new_K,
            (self.width, self.height), 
            cv2.CV_32FC1
        )
        
        print(f"Fisheye undistortion prepared:")
        print(f"  Original K:\n{self.K}")
        print(f"  Undistorted K:\n{self.new_K}")
        print(f"  Distortion coeffs: {self.D}")
    
    def undistort(self, image):
        """
        Undistort fisheye image to pinhole model
        
        Args:
            image: Input fisheye image (H, W, 3)
            
        Returns:
            Undistorted image (H, W, 3)
        """
        if self.map1 is None or self.map2 is None:
            raise RuntimeError("Undistortion maps not initialized")
        
        undistorted = cv2.remap(
            image, 
            self.map1, 
            self.map2, 
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )
        
        return undistorted
    
    def get_undistorted_intrinsics(self):
        """Get camera intrinsics matrix after undistortion"""
        return self.new_K.copy()
    
    def save_intrinsics(self, path):
        """Save undistorted intrinsics to .npy file"""
        np.save(path, self.new_K)
        print(f"Saved undistorted intrinsics to: {path}")


def calibrate_fisheye_camera(image_paths, checkerboard_size=(9, 6), square_size=0.025):
    """
    Calibrate fisheye camera using checkerboard images
    
    Args:
        image_paths: List of paths to checkerboard images
        checkerboard_size: (cols, rows) of inner corners
        square_size: Size of checkerboard square in meters
        
    Returns:
        K: Camera intrinsics matrix
        D: Distortion coefficients
    """
    print("Starting fisheye camera calibration...")
    print(f"Checkerboard size: {checkerboard_size}")
    print(f"Square size: {square_size}m")
    
    # Prepare object points
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size
    
    objpoints = []  # 3D points in real world
    imgpoints = []  # 2D points in image plane
    
    image_size = None
    
    for i, img_path in enumerate(image_paths):
        print(f"Processing image {i+1}/{len(image_paths)}: {img_path}")
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"  Warning: Could not read {img_path}")
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if image_size is None:
            image_size = gray.shape[::-1]
        
        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(
            gray, 
            checkerboard_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        if ret:
            objpoints.append(objp)
            
            # Refine corner positions
            corners_refined = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            imgpoints.append(corners_refined)
            
            print(f"  ✓ Found corners")
        else:
            print(f"  ✗ Could not find corners")
    
    if len(objpoints) < 3:
        raise ValueError(f"Need at least 3 valid images, got {len(objpoints)}")
    
    print(f"\nCalibrating with {len(objpoints)} images...")
    
    # Calibrate fisheye camera
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = []
    tvecs = []
    
    calibration_flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
        cv2.fisheye.CALIB_CHECK_COND +
        cv2.fisheye.CALIB_FIX_SKEW
    )
    
    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        image_size,
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )
    
    print("\nCalibration complete!")
    print(f"RMS reprojection error: {rms:.4f}")
    print(f"\nCamera matrix K:\n{K}")
    print(f"\nDistortion coefficients D:\n{D.ravel()}")
    
    return K, D.ravel(), rms


def create_entron_f008a_config(fx=2700.0, fy=2700.0, cx=1927.764404, cy=1096.686646,
                                width=3848, height=2168, k1=None, k2=None, k3=None, k4=None):
    """
    Create configuration for Entron F008A fisheye camera
    
    Args:
        fx, fy: Focal lengths (default: 2700.0)
        cx, cy: Principal point (default: centered on 3848x2168)
        width, height: Image dimensions
        k1, k2, k3, k4: Distortion coefficients (optional, will use defaults if None)
        
    Returns:
        Camera configuration dict
    """
    config = {
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy,
        'width': width,
        'height': height,
    }
    
    if k1 is not None:
        config.update({
            'k1': k1,
            'k2': k2 or 0.0,
            'k3': k3 or 0.0,
            'k4': k4 or 0.0
        })
    
    return config


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fisheye camera calibration and testing')
    parser.add_argument('--mode', choices=['calibrate', 'test-undistort'], required=True)
    parser.add_argument('--images', nargs='+', help='Input images')
    parser.add_argument('--output', help='Output path')
    parser.add_argument('--checkerboard', type=int, nargs=2, default=[9, 6],
                       help='Checkerboard size (cols rows)')
    parser.add_argument('--square-size', type=float, default=0.025,
                       help='Checkerboard square size in meters')
    
    args = parser.parse_args()
    
    if args.mode == 'calibrate':
        if not args.images:
            print("Error: --images required for calibration")
            exit(1)
        
        K, D, rms = calibrate_fisheye_camera(
            args.images,
            tuple(args.checkerboard),
            args.square_size
        )
        
        if args.output:
            np.savez(args.output, K=K, D=D, rms=rms)
            print(f"\nSaved calibration to: {args.output}")
    
    elif args.mode == 'test-undistort':
        if not args.images or len(args.images) != 1:
            print("Error: Provide exactly one image with --images")
            exit(1)
        
        # Create Entron F008A config
        config = create_entron_f008a_config()
        camera = FisheyeCamera(config)
        
        # Load and undistort
        img = cv2.imread(args.images[0])
        undistorted = camera.undistort(img)
        
        # Save result
        output = args.output or 'undistorted.jpg'
        cv2.imwrite(output, undistorted)
        
        print(f"\nUndistorted image saved to: {output}")
        print(f"New intrinsics:\n{camera.get_undistorted_intrinsics()}")