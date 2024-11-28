'''
Date: 2024/11/28
Reference: https://hackmd.io/@yoch/ByVdEVZP_

Update:
    Fix some road can't draw the lane line and use the mean of the slope and intercept to draw the lane line.
'''

import numpy as np
import cv2

class ImageProcessing:
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
        return cropped_edges
    
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
        # cv2.imshow('cropped_edges', edges)
        # cv2.waitKey(0)
        return lines



    @staticmethod
    def mean_coordinate(img, lines):
        '''
        計算左右車道線的坐標
        '''
        def make_coordinate(parameter, y_max, y_min):
            x1_mean = int((y_max - parameter[1]) / parameter[0])
            x2_mean = int((y_min - parameter[1]) / parameter[0])
            return np.array([x1_mean, y_max, x2_mean, y_min])
        left_fit = []
        right_fit = []
        y_threshold = 0.4
        y_max = round(img.shape[0])
        y_min = round(img.shape[0]*y_threshold)  # 通常從圖像底部到中間
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                parameter = np.polyfit((x1, x2), (y1, y2), 1)  # 求直線的斜率和截距
                slope = parameter[0]
                intercept = parameter[1]

                if abs(slope) < 0.5:
                    continue  # 去除接近水平的線

                if slope < 0:  # 判斷斜率的正負來區分左右車道
                    left_fit.append((slope, intercept))
                else:
                    right_fit.append((slope, intercept))

            if len(left_fit) > 0 and len(right_fit) > 0:
                left_fit_mean = np.mean(left_fit, axis=0)
                right_fit_mean = np.mean(right_fit, axis=0)

                left_coordinate = make_coordinate(left_fit_mean, y_max, y_min)
                right_coordinate = make_coordinate(right_fit_mean, y_max, y_min)

                return np.array([left_coordinate, right_coordinate])
        return None

    @staticmethod
    def draw_lines_on_image(image, lines):
        """在原始影像上繪製偵測到的直線。"""
        line_image = np.zeros_like(image)

        if lines is not None:
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
    loaded_data = cv2.imread(r"C:\Users\JUN-TING HUANG\JT\PyAutoDriveRL-Env\image_process\image_gallery\car_image_20241128_202453.png")
    # loaded_data = cv2.imread(r"C:\Users\JUN-TING HUANG\JT\PyAutoDriveRL-Env\image_process\image_gallery\car_image_20241128_202233.png")
    # loaded_data = cv2.imread(r"C:\Users\JUN-TING HUANG\JT\PyAutoDriveRL-Env\image_process\image_gallery\car_image_20241128_202152.png")
    # resize_data = cv2.resize(loaded_data, (image_size, image_size))
    resize_data = loaded_data

    # 使用靜態方法進行車道檢測
    # lane_image = ImageProcessing.lane_detection_pipeline(resize_data)
    # enhanced_image = ImageProcessing.enhance_red_objects(lane_image)

    lane_image = ImageProcessing.lane_detection_pipeline_forCNN(resize_data)

    # 顯示原始影像和車道檢測結果
    cv2.imshow('Original Image', loaded_data)
    cv2.imshow('Processed Image', lane_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
