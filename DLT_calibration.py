import cv2
import numpy as np
import os
import glob

class DLT_Calibrator:
    def __init__(self, img_dir, corner_dir, shape_inner_corner, size_grid, visualization=True):
        self.img_dir = img_dir
        self.corner_dir = corner_dir
        self.shape_inner_corner = shape_inner_corner
        self.size_grid = size_grid
        self.visualization = visualization
        self.mat_intri = None
        self.coff_dis = np.zeros(5)
        self.R = None
        self.t = None
        self.P = None

        w, h = shape_inner_corner
        self.cp_world = self.create_cube_world_points(h, size_grid)

        self.img_paths = []
        self.corner_paths = []

        for extension in ["jpg", "jpeg"]:
            img_files = glob.glob(os.path.join(img_dir, "*.{}".format(extension)))
            self.img_paths.extend(img_files)

            for img_path in img_files:
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                corner_path = os.path.join(corner_dir, f"{img_name}.txt")
                if os.path.exists(corner_path):
                    self.corner_paths.append(corner_path)

    def create_cube_world_points(self, rows, size):
        points = []
        for row in range(1, rows):
            for col in range(7):
                points.append([(7 - col) * size, (7 - row) * size, 0])

            points.append([0, (7 - row) * size, 0])

            for col in range(1, 8):
                points.append([0, (7 - row) * size, col * size])

        return np.array(points, dtype=np.float32)

    def load_corner_points(self, corner_path):
        points = []
        with open(corner_path, 'r') as f:
            for line in f:
                if line.strip():
                    x, y = line.strip().split()
                    points.append([float(x), float(y)])

        return np.array(points, dtype=np.float32).reshape(-1, 1, 2)

    def calibrate_camera_dlt(self):
        all_world_points = []
        all_pixel_points = []
        images = []

        for img_path, corner_path in zip(self.img_paths, self.corner_paths):
            img = cv2.imread(img_path)
            images.append(img)
            img_name = os.path.basename(img_path)

            cp_img = self.load_corner_points(corner_path)

            all_world_points.append(self.cp_world)
            all_pixel_points.append(cp_img)

            if self.visualization:
                vis_img = img.copy()
                for i, point in enumerate(cp_img):
                    x, y = point[0]
                    cv2.circle(vis_img, (int(x), int(y)), 5, (0, 0, 255), -1)
                    cv2.putText(vis_img, str(i), (int(x) + 10, int(y)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                cv2.imshow('Manually Marked Corners', vis_img)
                cv2.waitKey(500)
                save_path = os.path.join(self.img_dir, f"{img_name[:-4]}_corner.png")
                cv2.imwrite(save_path, vis_img)

        if self.visualization:
            cv2.destroyAllWindows()

        world_points_homo = []
        pixel_points_homo = []

        for world_points, pixel_points in zip(all_world_points, all_pixel_points):
            for wp, pp in zip(world_points, pixel_points):
                world_points_homo.append([wp[0], wp[1], wp[2], 1.0])
                pixel_points_homo.append([pp[0][0], pp[0][1], 1.0])

        world_points_homo = np.array(world_points_homo)
        pixel_points_homo = np.array(pixel_points_homo)

        A = []
        for i in range(len(world_points_homo)):
            X, Y, Z, _ = world_points_homo[i]
            u, v, _ = pixel_points_homo[i]

            A.append([X, Y, Z, 1, 0, 0, 0, 0, -u * X, -u * Y, -u * Z, -u])
            A.append([0, 0, 0, 0, X, Y, Z, 1, -v * X, -v * Y, -v * Z, -v])

        A = np.array(A)

        _, _, Vt = np.linalg.svd(A)
        P_vec = Vt[-1]
        P = P_vec.reshape(3, 4)

        P = P / P[2, 3]
        self.P = P
        P = P.astype(np.float64)
        K, R, t, _, _, _, _ = cv2.decomposeProjectionMatrix(P)
        K = K / K[2, 2]
        t = t[:3] / t[3]

        self.mat_intri = K
        self.R = R
        self.t = t

        # Calculate reprojection error
        total_error = 0
        num_points = 0

        for i, (world_points, pixel_points, img, img_path) in enumerate(zip(
                all_world_points, all_pixel_points, images, self.img_paths)):

            world_points_homo = np.hstack((world_points, np.ones((len(world_points), 1))))

            projected = self.P @ world_points_homo.T
            projected = projected / projected[2, :]
            projected_2D = projected[:2, :].T

            img_error = 0
            for j, pp in enumerate(pixel_points):
                dist = np.linalg.norm(pp[0] - projected_2D[j])
                img_error += dist

            avg_img_error = img_error / len(world_points)
            total_error += img_error
            num_points += len(world_points)

            # Visualization
            vis_img = self.visualize_reprojection(img,pixel_points,projected_2D,
                                                  avg_img_error,img_path)

            img_name = os.path.basename(img_path)
            reproj_path = os.path.join(self.img_dir, f"{img_name[:-4]}_reproj.png")
            cv2.imwrite(reproj_path, vis_img)

            if self.visualization:
                cv2.imshow(f"Reprojection {i + 1}/{len(images)}", vis_img)
                cv2.waitKey(500)

        if self.visualization:
            cv2.destroyAllWindows()

        avg_error = total_error / num_points

        print("Projection matrix P:\n", P)
        print("Intrinsics matrix K:\n", K)
        print("Rotation matrix R:\n", R)
        print("Translation vector t:\n", t)
        print("Average reprojection error: {:.4f} pixel".format(avg_error))

        return K, R, t

    def visualize_reprojection(self, img, original_points, projected_points, avg_error, img_path):
        vis_img = img.copy()
        height, width = img.shape[:2]
        for pt in original_points:
            x, y = pt[0]
            cv2.circle(vis_img, (int(x), int(y)), 5, (0, 0, 255), -1)

        for pt in projected_points:
            x, y = pt
            cv2.drawMarker(vis_img, (int(x), int(y)), (0, 255, 0),
                           markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)

        for orig_pt, proj_pt in zip(original_points, projected_points):
            x1, y1 = orig_pt[0]
            x2, y2 = proj_pt
            cv2.line(vis_img, (int(x1), int(y1)), (int(x2), int(y2)),
                     (255, 0, 0), 1)

        img_name = os.path.basename(img_path)
        error_text = f"Avg Error: {avg_error:.2f}px"
        cv2.putText(vis_img, error_text, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.putText(vis_img, "Original (Red)", (width - 200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(vis_img, "Reprojected (Green)", (width - 200, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(vis_img, "Error Vector (Blue)", (width - 200, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return vis_img


if __name__ == "__main__":
    img_dir = "Data"
    corner_dir = "Data"
    shape_inner_corner = (16, 8)
    size_grid = 0.016

    calibrator = DLT_Calibrator(
        img_dir=img_dir,
        corner_dir=corner_dir,
        shape_inner_corner=shape_inner_corner,
        size_grid=size_grid,
        visualization=True
    )

    K, R, t = calibrator.calibrate_camera_dlt()