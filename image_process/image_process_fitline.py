'''
Have Some Problem about the can't correct read the left and right line,
Need to adjust the threshold of the HoughLinesP function or something filter?
'''
import numpy as np
import cv2
def preprocess_image(image):
    # 轉為灰階影像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 使用高斯模糊降噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

def detect_edges(image):
    # 使用 Canny 邊緣檢測
    edges = cv2.Canny(image, 50, 150)
    return edges

def region_of_interest(edges):
    # 定義感興趣的區域 (ROI)
    height, width = edges.shape
    mask = np.zeros_like(edges)
    ROI_threshold = 0.4
    # 定義多邊形遮罩
    polygon = np.array([[
        (0, height),                
        (width, height),            
        (width, int(height * ROI_threshold)),
        (0, int(height * ROI_threshold))     
    ]], np.int32)

    # 填充多邊形區域
    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)
    return cropped_edges

def detect_lines(cropped_edges):
    # 使用霍夫變換檢測直線
    lines = cv2.HoughLinesP( # 480 * 960
        cropped_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=300,
        minLineLength=10,
        maxLineGap=120
    )
    # lines = cv2.HoughLinesP( # 64 * 64
    #     cropped_edges,
    #     rho=1,
    #     theta=np.pi / 180,
    #     threshold=10,
    #     minLineLength=10,
    #     maxLineGap=20
    # )
    return lines
def separate_lines(lines):
    left_lines = []
    right_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
        if slope < 0:  # Left lane
            left_lines.append(line)
        elif slope > 0:  # Right lane
            right_lines.append(line)
    return left_lines, right_lines

def fit_line(lines, image_shape):
    x_coords = []
    y_coords = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        x_coords += [x1, x2]
        y_coords += [y1, y2]

    if len(x_coords) == 0:
        return None

    # Fit a polynomial (linear fit)
    poly = np.polyfit(y_coords, x_coords, deg=1)
    y1 = image_shape[0]
    y2 = int(image_shape[0] * 0.4)
    x1 = int(np.polyval(poly, y1))
    x2 = int(np.polyval(poly, y2))
    return (x1, y1, x2, y2)

def draw_lane_lines(image, lines):
    left_lines, right_lines = separate_lines(lines)
    left_fit = fit_line(left_lines, image.shape)
    right_fit = fit_line(right_lines, image.shape)

    line_image = np.zeros_like(image)

    if left_fit is not None:
        x1, y1, x2, y2 = left_fit
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

    if right_fit is not None:
        x1, y1, x2, y2 = right_fit
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)

    # Draw the triangle (region of interest)
    if left_fit is not None and right_fit is not None:
        pts = np.array([[left_fit[0], left_fit[1]], [right_fit[0], right_fit[1]], [right_fit[2], right_fit[3]], [left_fit[2], left_fit[3]]], np.int32)
        cv2.fillPoly(line_image, [pts], (0, 255, 255))

    combined_image = cv2.addWeighted(image, 0.8, line_image, 1, 1)
    return combined_image


def lane_detection_pipeline(image):
    # 影像預處理
    preprocessed_image = preprocess_image(image)
    # 邊緣檢測
    edges = detect_edges(preprocessed_image)
    # 區域選擇
    cropped_edges = region_of_interest(edges)
    # 偵測直線
    lines = detect_lines(cropped_edges)

    cv2.imshow('', cropped_edges)
    cv2.waitKey(0)
    # # 繪製車道線
    # lane_image = draw_lines(image, lines)
    lane_image = draw_lane_lines(preprocessed_image, lines)

    return lane_image
# 讀取 .npy 文件
loaded_data = np.load('image_process/image2.npy')
resize_data = cv2.resize(loaded_data, (64,64))
lane_image = lane_detection_pipeline(loaded_data)
# lane_image = lane_detection_pipeline(resize_data)

cv2.imshow('image', loaded_data)
cv2.imshow('lane_image', lane_image)
cv2.waitKey(0)



# def measure_curvature_real(actual_fit, ploty):
#     '''
#     Calculates the curvature o
#     f polynomial functions in meters.
#     '''
#     ym_per_pix = 30/700 # meters per pixel in y dimension
#     xm_per_pix = 3.7/700 # meters per pixel in x dimension
#     # Define y-value where we want radius of curvature
#     # We'll choose the maximum y-value, corresponding to the bottom of the image
#     y_eval = np.max(ploty) * ym_per_pix
 
#     ##### Implement the calculation of R_curve (radius of curvature) #####
#     curvature = ((1 + (2*actual_fit[0]*y_eval + actual_fit[1])**2)**1.5) / np.absolute(2*actual_fit[0])
#     return curvature * actual_fit[0]/abs(actual_fit[0])

# def horizontal_distance(left_fit,right_fit,ploty):
#     xm_per_pix = 3.7/700
#     left_fitx = (left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]) * xm_per_pix
#     right_fitx = (right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]) * xm_per_pix
#     average_distance = np.average(right_fitx - left_fitx)
#     std_distance = np.std(right_fitx - left_fitx)
    
#     x_der = right_fitx[0]
#     x_izq = left_fitx[0]
#     center_car = (1280*xm_per_pix/2.0)
#     center_road = ((x_der+x_izq)/2.0)
#     position = center_car-center_road
#     return average_distance, std_distance, position