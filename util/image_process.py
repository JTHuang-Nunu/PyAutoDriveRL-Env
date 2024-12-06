'''
Date: 2024/11/28
Reference: https://hackmd.io/@yoch/ByVdEVZP_

Update:
    Fix some road can't draw the lane line and use the mean of the slope and intercept to draw the lane line.
'''

import numpy as np
import cv2
import torch
from sklearn.linear_model import RANSACRegressor


class ImageProcessing:
    _last_right_line = None
    _last_left_line = None

    @staticmethod
    def preprocess_image(image):
        """轉為灰階影像並進行高斯模糊處理。"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return blurred

    @staticmethod
    def detect_edges(image):
        """使用 Canny 邊緣檢測。"""
        edges = cv2.Canny(image, 40, 120)
        return edges
    
    @staticmethod
    def region_of_interest_1(image):
        height, width = image.shape
        polygons = np.array([
            [(0, height), (width // 2 - 100, height // 2), (width // 2 + 100, height // 2), (width, height)]
        ])
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, polygons, 255)
        return cv2.bitwise_and(image, mask)

    @staticmethod
    def region_of_interest(edges):
        """定義感興趣的區域 (ROI)。"""
        height, width = edges.shape
        mask = np.zeros_like(edges)
        ROI_threshold = 0.4

        polygon = np.array([[ # ROI Square
            (0, height),
            (width, height),
            (width, int(height * ROI_threshold)),
            (0, int(height * ROI_threshold))
        ]], np.int32)

        cv2.fillPoly(mask, polygon, 255)
        cropped_edges = cv2.bitwise_and(edges, mask)
        # cv2.imshow('cropped_edges', cropped_edges)
        # cv2.waitKey(0)
        return cropped_edges
    
    @staticmethod
    def load_yolo_model(model_path='yolov5s.pt'):
        """
        Load the YOLO model.
        :param model_path: Path to the pre-trained YOLO model.
        :return: Loaded YOLO model.
        """
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        #model = torch.hub.load('yolo11n.pt', 'custom', path=model_path)
        return model

    @staticmethod
    def detect_objects_yolo(image, model, conf_threshold=0.5, classes_to_detect=None):
        """
        Detect objects in an image using YOLO.
        :param image: Input image (BGR format).
        :param model: Pre-loaded YOLO model.
        :param conf_threshold: Confidence threshold for detections.
        :param classes_to_detect: List of class indices to detect (e.g., cars, roads).
        :return: Image with detected objects and detection results.
        """
        # Ensure image is valid
        if image is None or len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be a valid 3-channel image (RGB/BGR).")

        # Resize to model input size (e.g., 640x640)
        input_size = 640
        resized_image = cv2.resize(image, (input_size, input_size))

        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

        # Perform detection
        results = model(rgb_image)

        # Debug model output
        detections = results.pred[0].cpu().numpy()
        print("Detections:", detections)

        # Filter detections
        detections = [d for d in detections if d[4] > conf_threshold and 
                    (classes_to_detect is None or int(d[5]) in classes_to_detect)]

        # Annotate and return detections
        annotated_image = image.copy()
        objects = []
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            x1 = int(x1 * (image.shape[1] / input_size))
            x2 = int(x2 * (image.shape[1] / input_size))
            y1 = int(y1 * (image.shape[0] / input_size))
            y2 = int(y2 * (image.shape[0] / input_size))
            label = f"{model.names[int(cls)]} {conf:.2f}"
            color = (0, 255, 0) if int(cls) == 2 else (255, 0, 0)
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            objects.append((int(cls), x1, y1, x2, y2))

        return annotated_image, objects

    
    @staticmethod
    def detect_lines(cropped_edges):
        """使用霍夫變換檢測直線。"""
        if cropped_edges.shape[0] == 64 or cropped_edges.shape[0] == 128 or cropped_edges.shape[0] == 256:
            lines = cv2.HoughLinesP( # 64 * 64
                cropped_edges,
                rho=1,
                theta=np.pi / 180,
                threshold=10,
                minLineLength=20,
                maxLineGap=20
            )
        elif cropped_edges.shape[0] == 480:
            lines = cv2.HoughLinesP( # 480 * 960
                cropped_edges,
                rho=1,
                theta=np.pi / 180,
                threshold=100,
                minLineLength=10,
                maxLineGap=120
            )

        return lines



    @staticmethod
    def mean_coordinate(img_shape, lines):
        """
        Calculates the left and right lane line coordinates.
        """
        def make_coordinate(parameter, y_max, y_min):
            x1_mean = int((y_max - parameter[1]) / parameter[0])
            x2_mean = int((y_min - parameter[1]) / parameter[0])
            return np.array([x1_mean, y_max, x2_mean, y_min])

        left_fit = []
        right_fit = []
        y_threshold = 0.4
        y_max = img_shape[0]  # Use height directly
        y_min = round(img_shape[0] * y_threshold)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x1 - x2) < 1e-5 or abs(y1 - y2) < 1e-5:
                    continue
                try:
                    # Try to fit a line to the points
                    parameter = np.polyfit((x1, x2), (y1, y2), 1)
                    slope = parameter[0]
                    intercept = parameter[1]

                    if abs(slope) < 0.5:
                        continue  # Ignore nearly horizontal lines

                    if slope < 0:  # Left lane
                        left_fit.append((slope, intercept))
                    else:  # Right lane
                        right_fit.append((slope, intercept))
                except np.linalg.LinAlgError:
                    continue

            coordinates = []
            if len(left_fit) > 0:
                left_fit_mean = np.mean(left_fit, axis=0)
                left_coordinate = make_coordinate(left_fit_mean, y_max, y_min)
                ImageProcessing._last_left_line = left_coordinate
                coordinates.append(left_coordinate)
            else:
                coordinates.append(ImageProcessing._last_left_line if ImageProcessing._last_left_line is not None else None)

            if len(right_fit) > 0:
                right_fit_mean = np.mean(right_fit, axis=0)
                right_coordinate = make_coordinate(right_fit_mean, y_max, y_min)
                ImageProcessing._last_right_line = right_coordinate
                coordinates.append(right_coordinate)
            else:
                coordinates.append(ImageProcessing._last_right_line if ImageProcessing._last_right_line is not None else None)

            return coordinates
        return None


    @staticmethod
    def draw_lines_on_image(image, lines):
        """在原始影像上繪製偵測到的直線。"""
        line_image = np.zeros_like(image)

        lines = lines or [] # replace None with empty list

        if line_image.ndim == 2:
            for line in lines:
                x1, y1, x2, y2 = line
                cv2.line(line_image, (x1, y1), (x2, y2), 255, 2)
        elif line_image.ndim == 3:
            for line in lines:
                x1, y1, x2, y2 = line
                cv2.line(line_image, (x1, y1), (x2, y2), (255,255,255), 2)
            
        combined_image = cv2.addWeighted(image, 1, line_image, 1, 1)

        return combined_image
        # if len(lines) >= 2 or ImageProcessing._last_image is None:
        #     ImageProcessing._last_image = combined_image
        #     return combined_image
        # return ImageProcessing._last_image if ImageProcessing._last_image is not None else image

    @staticmethod
    def enhance_red_objects(image, red_threshold=30, alpha=2):
        """
        增強圖像中紅色物件的紅色強度。
        :param image: 原始彩色圖像 (BGR 格式)。
        :param red_threshold: 紅色通道的閾值，僅增強R比G或B大於這個閾值的像素。
        :param alpha: 增強係數，默認為 2。
        :return: 增強紅色後的彩色圖像。
        """
        # 分離 B, G, R 通道
        b, g, r = cv2.split(image)

        # 建立紅色掩膜：紅色通道大於綠色和藍色通道
        red_mask = (r > g+red_threshold) & (r > b+red_threshold)

        # 增強紅色通道，只對符合紅色掩膜的像素進行增強
        r[red_mask] = np.clip(r[red_mask] * alpha, 0, 255).astype(np.uint8)

        # 合併 B, G, R 通道
        enhanced_image = cv2.merge([b, g, r])
        return enhanced_image

    @staticmethod
    def lane_detection_pipeline(image):
        """車道線檢測流程。"""
        preprocessed_image = ImageProcessing.preprocess_image(image)
        edges = ImageProcessing.detect_edges(preprocessed_image)
        cropped_edges = ImageProcessing.region_of_interest(edges)
        lines = ImageProcessing.detect_lines(cropped_edges)

        corrdinate = ImageProcessing.mean_coordinate(image, lines)
        lane_image = ImageProcessing.draw_lines_on_image(image, corrdinate)

        return lane_image

    @staticmethod
    def adaptive_canny(image):
        median_intensity = np.median(image)
        lower_threshold = int(max(0, (1.0 - 0.33) * median_intensity))
        upper_threshold = int(min(255, (1.0 + 0.33) * median_intensity))
        edges = cv2.Canny(image, lower_threshold, upper_threshold)
        return edges
   
    @staticmethod
    def warp_perspective(image, src_points, dst_points):
        """
        Applies perspective transform to the input image.
        """
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        return cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))

    @staticmethod
    def mean_coordinate_curve(img_shape, lines):
        """
        Calculates the left and right lane line coordinates.
        """
        def make_coordinate(parameter, y_max, y_min):
            x1_mean = int((y_max - parameter[1]) / parameter[0])
            x2_mean = int((y_min - parameter[1]) / parameter[0])
            return np.array([x1_mean, y_max, x2_mean, y_min])

        left_fit = []
        right_fit = []
        y_threshold = 0.4
        y_max = img_shape[0]
        y_min = round(img_shape[0] * y_threshold)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x1 - x2) < 1e-5 or abs(y1 - y2) < 1e-5:
                    continue
                try:
                    parameter = np.polyfit((x1, x2), (y1, y2), 1)
                    slope = parameter[0]
                    intercept = parameter[1]

                    if abs(slope) < 0.3 or abs(slope) > 2:  # Adjust slope range for valid lanes
                        continue

                    if slope < 0:  # Left lane
                        left_fit.append((slope, intercept))
                    else:  # Right lane
                        right_fit.append((slope, intercept))
                except np.linalg.LinAlgError:
                    continue

            coordinates = []
            if len(left_fit) > 0:
                left_fit_mean = np.mean(left_fit, axis=0)
                left_coordinate = make_coordinate(left_fit_mean, y_max, y_min)
                ImageProcessing._last_left_line = left_coordinate
                coordinates.append(left_coordinate)
            else:
                coordinates.append(ImageProcessing._last_left_line)

            if len(right_fit) > 0:
                right_fit_mean = np.mean(right_fit, axis=0)
                right_coordinate = make_coordinate(right_fit_mean, y_max, y_min)
                ImageProcessing._last_right_line = right_coordinate
                coordinates.append(right_coordinate)
            else:
                coordinates.append(ImageProcessing._last_right_line)

            return coordinates
        return None

    @staticmethod
    def ransac_filter(points_x, points_y):
        """
        Filters points using RANSAC to remove outliers.
        """
        points = np.array([points_x, points_y]).T
        model = RANSACRegressor()
        model.fit(points[:, 1].reshape(-1, 1), points[:, 0])  # x = f(y)
        inlier_mask = model.inlier_mask_
        return points[inlier_mask, 0], points[inlier_mask, 1]
    @staticmethod
    def fit_polynomial(lines, image_shape):
        """
        Fits polynomials to the left and right lane lines and returns them.
        """
        height, width, _ = image_shape
        coordinates = ImageProcessing.mean_coordinate_curve((height, width), lines)
        if coordinates is None or len(coordinates) == 0:
            return None, None

        left_points = []
        right_points = []

        for coord in coordinates:
            if coord is not None:
                x1, y1, x2, y2 = coord
                if x1 < width // 2 and x2 < width // 2:
                    left_points.append((x1, y1))
                    left_points.append((x2, y2))
                elif x1 > width // 2 and x2 > width // 2:
                    right_points.append((x1, y1))
                    right_points.append((x2, y2))

        left_curve = None
        right_curve = None

        if len(left_points) > 0:
            left_y, left_x = zip(*left_points)
            left_x, left_y = ImageProcessing.ransac_filter(left_x, left_y)  # Apply RANSAC
            if len(left_x) > 2:  # Ensure sufficient points for fitting
                left_poly = np.polyfit(left_y, left_x, 2)
                left_y_points = np.linspace(0, height, num=100)
                left_x_points = np.polyval(left_poly, left_y_points)
                left_curve = np.array([np.transpose([left_x_points, left_y_points])], np.int32)

        if len(right_points) > 0:
            right_y, right_x = zip(*right_points)
            right_x, right_y = ImageProcessing.ransac_filter(right_x, right_y)  # Apply RANSAC
            if len(right_x) > 2:  # Ensure sufficient points for fitting
                right_poly = np.polyfit(right_y, right_x, 2)
                right_y_points = np.linspace(0, height, num=100)
                right_x_points = np.polyval(right_poly, right_y_points)
                right_curve = np.array([np.transpose([right_x_points, right_y_points])], np.int32)

        return left_curve, right_curve

    
    @staticmethod
    def lane_detection_pipeline_forYOLO(image):
        """車道線檢測流程。"""
        preprocessed_image = ImageProcessing.preprocess_image(image)
        edges = ImageProcessing.detect_edges(preprocessed_image)
        cropped_edges = ImageProcessing.region_of_interest_1(edges)
        lines = ImageProcessing.detect_lines(cropped_edges)

        #corrdinate = ImageProcessing.mean_coordinate(image, lines)
        #lane_image = ImageProcessing.draw_lines_on_image(image, corrdinate)
        
        #curve = ImageProcessing.fit_polynomial(lines, image.shape)
        lane_image = np.zeros_like(image)

        
        lane_image = cv2.cvtColor(lane_image, cv2.COLOR_BGR2GRAY)
        resized_lane_image = cv2.resize(lane_image, (128, 128))

        image = cv2.resize(image,(128,128))

        # Display the combined lane and car procedure image
        cv2.imshow("Lane Detection and Car Procedure", resized_lane_image)

        # Load the YOLO model
        yolo_model = ImageProcessing.load_yolo_model()
        annotated_image, objects = ImageProcessing.detect_objects_yolo(image, yolo_model)  # Example: Class 0 (cars), Class 1 (road)

        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2GRAY)

        return annotated_image  


    @staticmethod
    def lane_detection_pipeline_forCNN(image):
        """車道線檢測流程。"""
        preprocessed_image = ImageProcessing.preprocess_image(image)
        edges = ImageProcessing.detect_edges(preprocessed_image)
        cropped_edges = ImageProcessing.region_of_interest(edges)
        # return cropped_edges

        lines = ImageProcessing.detect_lines(cropped_edges)
        # empty_image = np.zeros_like(image).squeeze()
        empty_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        corrdinate = ImageProcessing.mean_coordinate(empty_image, lines)
        lane_image = ImageProcessing.draw_lines_on_image(empty_image, corrdinate)
        return lane_image

# 測試 ImageProcessing 類別
if __name__ == "__main__":
    image_size = 128
    # 讀取 .npy 文件
    # loaded_data = np.load('image_process/image2.npy')
    loaded_data = cv2.imread(r"C:\Users\JUN-TING HUANG\JT\PyAutoDriveRL-Env\image_process\image_gallery\car_image_20241128_214907.png")
    # loaded_data = cv2.imread(r"C:\Users\JUN-TING HUANG\JT\PyAutoDriveRL-Env\image_process\image_gallery\car_image_20241128_202233.png")
    # loaded_data = cv2.imread(r"C:\Users\JUN-TING HUANG\JT\PyAutoDriveRL-Env\image_process\image_gallery\car_image_20241128_202152.png")
    # resize_data = cv2.resize(loaded_data, (image_size, image_size))
    resize_data = loaded_data

    # 使用靜態方法進行車道檢測
    # lane_image = ImageProcessing.lane_detection_pipeline(resize_data)
    # enhanced_image = ImageProcessing.enhance_red_objects(lane_image)

    #lane_image = ImageProcessing.lane_detection_pipeline_forCNN(resize_data)
    lane_image = ImageProcessing.lane_detection_pipeline_forYOLO(resize_data)

    # 顯示原始影像和車道檢測結果
    cv2.imshow('Original Image', loaded_data)
    cv2.imshow('Processed Image', lane_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
