import threading
import tkinter as tk
import time

import tkinter as tk

class CarDataWindow:
    def __init__(self, shared_dict):
        self.shared_dict = shared_dict
        self.window = tk.Tk()
        self.window.title("Car Data Monitor")
        self.data_label = tk.Label(self.window, text="Initializing...", font=("Courier", 10), justify=tk.LEFT)
        self.data_label.pack()

        # bind the window close event
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

        self.update_data()

    def update_data(self):
        data = self.shared_dict.get('car_data', "No data available")
        self.data_label.config(text=data)
        # 每 100 毫秒更新一次数据
        self.window.after(100, self.update_data)

    def on_close(self):
        '''
        Close the window.
        '''
        self.window.destroy()

    def start(self):
        self.window.mainloop()

    def stop(self):
        """
        Stop the GUI thread.
        """
        if self.window is not None and self.window.winfo_exists():
            self.window.destroy()
