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
        edges = cv2.Canny(image, 50, 150)
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
                minLineLength=10,
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
    def draw_lines_on_color(image, lines):
        """在原始影像上繪製偵測到的直線。"""
        line_image = np.zeros_like(image)
        height, width = image.shape[:2]
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 < width * 0.3:
                    cv2.line(line_image, (x1, y1), (x2, y2), (0,255,0), 1)
                elif x2 > width * 0.7:
                    cv2.line(line_image, (x1, y1), (x2, y2), (255,0,0), 1)

        # combined_image = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), 0.8, line_image, 1, 1)
        combined_image = cv2.addWeighted(image, 0.8, line_image, 1, 1)
        return combined_image

    @staticmethod
    def draw_lines_on_gray(image, lines):
        """在灰階影像上繪製偵測到的直線。"""
        line_image = np.zeros_like(image)

        if lines is not None:
            if line_image.ndim == 2:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(line_image, (x1, y1), (x2, y2), 255, 2)
            elif line_image.ndim == 3:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
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
        lane_image = ImageProcessing.draw_lines_on_gray(image, lines)
        
        # Fuse the lane image with the gray image
        # lane_image = ImageProcessing.draw_lines_on_gray(preprocessed_image, lines)
        # lane_image = np.expand_dims(lane_image, axis=-1)

        return lane_image

# 測試 ImageProcessing 類別
if __name__ == "__main__":
    image_size = 256
    # 讀取 .npy 文件
    loaded_data = np.load('image_process/image2.npy')
    resize_data = cv2.resize(loaded_data, (image_size, image_size))

    # 使用靜態方法進行車道檢測
    lane_image = ImageProcessing.lane_detection_pipeline(resize_data)
    enhanced_image = ImageProcessing.enhance_red_objects(lane_image)

    # 顯示原始影像和車道檢測結果
    cv2.imshow('Original Image', loaded_data)
    cv2.imshow('Processed Image', enhanced_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
