import threading
import tkinter as tk
import time

class CarDataWindow:
    def __init__(self, shared_dict):
        self.shared_dict = shared_dict
        self.window = tk.Tk()
        self.window.title("Car Data Monitor")
        self.data_label = tk.Label(self.window, text="Initializing...", font=("Courier", 10), justify=tk.LEFT)
        self.data_label.pack()
        self.update_data()

    def update_data(self):
        data = self.shared_dict.get('car_data')
        self.data_label.config(text=data)
        self.window.after(100, self.update_data)

    def run(self):
        self.window.mainloop()

class CarDataWindowBAK:
    def __init__(self, share_dict):
        # 初始化 tkinter 視窗
        self.window = tk.Tk()
        self.window.title("Car Data Monitor")

        # 初始化標籤
        self.data_text = tk.Text(self.window, height=20, width=50, justify=tk.LEFT)
        self.data_text.pack()
        
        # # 建立執行緒
        # self.gui_thread = threading.Thread(target=self.run)
        # self.gui_thread.daemon = True  # 設置為 daemon 執行緒

        # 模擬初始資料
        self.car_data = {
            "Timestamp": 1731480193140,
            "Image shape": "(480, 960, 3)",
            "Steering Angle": -11.7095,
            "Throttle": 0.0,
            "Speed": 3.0338,
            "Velocity (X, Y, Z)": (-0.3057, -0.0004, 1.3213),
            "Acceleration (X, Y, Z)": (6.0780, 0.0854, -2.2506),
            "Angular Velocity (X, Y, Z)": (-0.0031, -0.0985, -0.0003),
            "Wheel Friction (F, S)": (-0.0549, 0.0),
            "Brake Input": 0.2759,
            "Yaw, Pitch, Roll": (353.5719, 359.7080, 359.9931),
            "Y Position": 0.5570,
            "Hit An Obstacle": 0.0,
            "Progress": 0.0122,
            "Time Speed Up Scale": 1.0,
            "FPS": 8.0,
            "Reward": 0.06
        }
        self.share_dict = share_dict

        # # 開始更新資料
        self.update_data()
    # def start(self):
    #     self.gui_thread.start()

    def update_data(self):
        # self.car_data = data
        # 清空視窗文字
        self.data_text.delete(1.0, tk.END)

        # 動態更新資料
        display_text = "CarData\n" + "="*50 + "\n"
        # for key, value in self.car_data.items():
        #     display_text += f"{key:<30}: {value}\n"
        display_text = self.share_dict['car_data']
        # 更新顯示文字
        self.data_text.insert(tk.END, display_text)

        # 模擬資料更新（可以替換為實時資料獲取邏輯）
        # self.car_data["Timestamp"] += 1
        # self.car_data["Speed"] += 0.01
        # self.car_data["Progress"] += 0.001

        # 每 500 毫秒更新一次
        self.window.after(500, self.update_data)

    def run(self):
        self.window.mainloop()