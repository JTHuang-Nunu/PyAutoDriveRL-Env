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
        lines = cv2.HoughLinesP( # 64 * 64
            cropped_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=10,
            minLineLength=10,
            maxLineGap=20
        )
        # lines = cv2.HoughLinesP( # 480 * 960
        #     cropped_edges,
        #     rho=1,
        #     theta=np.pi / 180,
        #     threshold=100,
        #     minLineLength=10,
        #     maxLineGap=120
        # )
        return lines

    @staticmethod
    def draw_lines_on_color(image, lines):
        """在原始影像上繪製偵測到的直線。"""
        line_image = np.zeros_like(image)
        line_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        height, width = image.shape
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 < width * 0.3:
                    cv2.line(line_image, (x1, y1), (x2, y2), (0,255,0), 1)
                elif x2 > width * 0.7:
                    cv2.line(line_image, (x1, y1), (x2, y2), (255,0,0), 1)

        combined_image = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), 0.8, line_image, 1, 1)
        return combined_image

    @staticmethod
    def draw_lines_on_gray(image, lines):
        """在灰階影像上繪製偵測到的直線。"""
        line_image = np.zeros_like(image)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), 255, 1)

        combined_image = cv2.addWeighted(image, 0.8, line_image, 1, 1)
        return combined_image

    @staticmethod
    def lane_detection_pipeline(image):
        """車道線檢測流程。"""
        preprocessed_image = ImageProcessing.preprocess_image(image)
        edges = ImageProcessing.detect_edges(preprocessed_image)
        cropped_edges = ImageProcessing.region_of_interest(edges)
        lines = ImageProcessing.detect_lines(cropped_edges)
        lane_image = ImageProcessing.draw_lines_on_gray(preprocessed_image, lines)
        return lane_image

# 測試 ImageProcessing 類別
if __name__ == "__main__":
    # 讀取 .npy 文件
    loaded_data = np.load('image_process/image2.npy')
    resize_data = cv2.resize(loaded_data, (64, 64))

    # 使用靜態方法進行車道檢測
    lane_image = ImageProcessing.lane_detection_pipeline(resize_data)

    # 顯示原始影像和車道檢測結果
    cv2.imshow('Original Image', loaded_data)
    cv2.imshow('Lane Detection', lane_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
