import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import numpy as np

# Assuming compute_strehl and airy_2D are defined elsewhere
# from your_module import compute_strehl, airy_2D

# Dummy implementations for the purpose of this example
def compute_strehl(image):
    return np.random.random()

airy_2D = np.random.random((100, 100)) * 255

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Strehl Ratio GUI")

        # Live Camera Stream
        self.camera_label = tk.Label(root)
        self.camera_label.grid(row=0, column=0)
        self.cap = cv2.VideoCapture(0)

        # Zoomed-in Window
        self.zoomed_label = tk.Label(root)
        self.zoomed_label.grid(row=0, column=1)

        # Static Ideal Image
        self.ideal_label = tk.Label(root)
        self.ideal_label.grid(row=0, column=2)

        # Strehl Ratio Label
        self.strehl_label = tk.Label(root, text="Strehl Ratio (0 to 1)")
        self.strehl_label.grid(row=1, column=0, columnspan=2)

        # Strehl Ratio Value
        self.strehl_value_label = tk.Label(root, text="0.0")
        self.strehl_value_label.grid(row=1, column=2)

        # Strehl Ratio Bar
        self.strehl_bar = ttk.Progressbar(root, orient="horizontal", length=200, mode="determinate", maximum=1)
        self.strehl_bar.grid(row=2, column=0, columnspan=2)

        # Max Value Label
        self.max_value_label = tk.Label(root, text="Max Value (0 to 255)")
        self.max_value_label.grid(row=3, column=0, columnspan=2)

        # Max Value Value
        self.max_value_value_label = tk.Label(root, text="0")
        self.max_value_value_label.grid(row=3, column=2)

        # Max Value Bar
        self.max_value_bar = ttk.Progressbar(root, orient="horizontal", length=200, mode="determinate", maximum=255)
        self.max_value_bar.grid(row=4, column=0, columnspan=2)

        self.update()

    def update(self):
        ret, frame = self.cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray)

            # Draw red rectangle around the brightest pixel
            cv2.rectangle(frame, (max_loc[0]-10, max_loc[1]-10), (max_loc[0]+10, max_loc[1]+10), (0, 0, 255), 2)

            # Zoomed-in region
            zoomed_region = gray[max(0, max_loc[1]-10):min(gray.shape[0], max_loc[1]+10), max(0, max_loc[0]-10):min(gray.shape[1], max_loc[0]+10)]
            zoomed_image = Image.fromarray(zoomed_region)
            zoomed_image = zoomed_image.resize((frame.shape[1], frame.shape[0]), Image.NEAREST)
            zoomed_image = ImageTk.PhotoImage(image=zoomed_image)

            # Update GUI elements
            self.zoomed_label.config(image=zoomed_image)
            self.zoomed_label.image = zoomed_image

            strehl_ratio = compute_strehl(zoomed_region)
            self.strehl_bar['value'] = strehl_ratio
            self.strehl_value_label.config(text=f"{strehl_ratio:.2f}")

            self.max_value_bar['value'] = max_val
            self.max_value_value_label.config(text=f"{max_val}")

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.camera_label.config(image=img)
            self.camera_label.image = img

            # Resize the airy disc image to match the frame size
            airy_image = Image.fromarray(airy_2D.astype(np.uint8))
            airy_image = airy_image.resize((frame.shape[1], frame.shape[0]), Image.NEAREST)
            airy_image = ImageTk.PhotoImage(image=airy_image)
            self.ideal_label.config(image=airy_image)
            self.ideal_label.image = airy_image

        self.root.after(10, self.update)

    def on_closing(self):
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()