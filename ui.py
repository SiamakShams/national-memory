import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import g
import os
import configparser

class FaceExtractorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Extractor")
        self.root.minsize(800, 600)

        self.filepath = None
        self.original_image = None
        self.image_with_boxes = None
        self.face_boxes = []
        self.cv2_image = None

        # Settings
        self.config_file = "config.ini"
        self.load_settings()

        # Create Menu Bar
        self.menu_bar = tk.Menu(self.root)
        self.root.config(menu=self.menu_bar)

        # File Menu
        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="Exit", command=self.root.quit)

        # Configuration Menu
        self.config_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Configuration", menu=self.config_menu)
        self.config_menu.add_command(label="Settings", command=self.open_settings)

        # Faces Menu
        self.faces_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Faces", menu=self.faces_menu)
        self.faces_menu.add_command(label="No faces detected", state="disabled")

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self.main_frame = tk.Frame(root, padx=10, pady=10)
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        self.image_label = tk.Label(self.main_frame, text="No image selected", relief="groove", anchor="center")
        self.image_label.grid(row=0, column=0, sticky="nsew", pady=5)
        self.image_label.bind('<Configure>', self.on_resize)

        self.button_frame = tk.Frame(self.main_frame)
        self.button_frame.grid(row=1, column=0, pady=5, sticky="ew")
        self.button_frame.grid_columnconfigure(0, weight=1)
        self.button_frame.grid_columnconfigure(1, weight=1)

        self.select_button = tk.Button(self.button_frame, text="Select Image", command=self.select_image)
        self.select_button.grid(row=0, column=0, padx=5, sticky="e")

        self.extract_button = tk.Button(self.button_frame, text="Extract Faces", command=self.extract_faces, state="disabled")
        self.extract_button.grid(row=0, column=1, padx=5, sticky="w")
        
        self.status_label = tk.Label(self.main_frame, text="", fg="blue")
        self.status_label.grid(row=2, column=0, pady=5, sticky="ew")

    def load_settings(self):
        self.config = configparser.ConfigParser()
        if os.path.exists(self.config_file):
            self.config.read(self.config_file)
            self.input_folder = self.config.get("Settings", "input_folder", fallback=os.getcwd())
            self.output_folder = self.config.get("Settings", "output_folder", fallback=os.path.join(os.getcwd(), "faces"))
        else:
            self.input_folder = os.getcwd()
            self.output_folder = os.path.join(os.getcwd(), "faces")
            self.config["Settings"] = {"input_folder": self.input_folder, "output_folder": self.output_folder}

    def select_image(self):
        self.filepath = filedialog.askopenfilename(
            initialdir=self.input_folder,
            title="Select an Image",
            filetypes=(("jpeg files", "*.jpg"), ("png files", "*.png"), ("all files", "*.*"))
        )
        if self.filepath:
            self.original_image = Image.open(self.filepath).convert("RGB")
            self.image_with_boxes = self.original_image.copy()
            self.face_boxes = []
            self.display_image()
            self.extract_button.config(state="normal")
            self.status_label.config(text=f"Selected: {self.filepath.split('/')[-1]}")

    def display_image(self):
        img_to_display = self.image_with_boxes if self.image_with_boxes else self.original_image
        if not img_to_display:
            return

        label_width = self.image_label.winfo_width()
        label_height = self.image_label.winfo_height()

        if label_width == 1 or label_height == 1:
            self.root.after(50, self.display_image)
            return

        img_copy = img_to_display.copy()
        img_copy.thumbnail((label_width - 10, label_height - 10), Image.Resampling.LANCZOS)

        photo = ImageTk.PhotoImage(img_copy)

        self.image_label.config(image=photo, text="")
        self.image_label.image = photo
        
    def on_resize(self, event):
        self.display_image()

    def extract_faces(self):
        if not self.filepath:
            messagebox.showwarning("Warning", "Please select an image first.")
            return

        self.extract_button.config(state="disabled")
        self.select_button.config(state="disabled")
        self.status_label.config(text="Detecting faces...")
        self.root.update_idletasks()

        self.cv2_image, self.face_boxes = g.get_face_boxes(self.filepath)

        if not self.face_boxes:
            messagebox.showinfo("Info", "No faces detected.")
            self.status_label.config(text="")
            self.update_faces_menu()
            self.extract_button.config(state="normal")
            self.select_button.config(state="normal")
            return

        self.status_label.config(text=f"Found {len(self.face_boxes)} faces. Highlighting...")
        self.draw_boxes() # Draw initial amber boxes
        self.update_faces_menu()
        self.root.after(1000, self.extract_and_save_one_face, 0)

    def draw_boxes(self, highlight_index=None):
        self.image_with_boxes = self.original_image.copy()
        draw = ImageDraw.Draw(self.image_with_boxes)

        for i, box in enumerate(self.face_boxes):
            color = "green" if highlight_index is not None and i <= highlight_index else "orange"
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline=color, width=3)
        
        self.display_image()
        self.root.update_idletasks()

    def extract_and_save_one_face(self, face_index):
        if face_index >= len(self.face_boxes):
            self.status_label.config(text=f"Finished extracting {len(self.face_boxes)} faces.")
            messagebox.showinfo("Success", f"Successfully saved {len(self.face_boxes)} faces.")
            self.extract_button.config(state="normal")
            self.select_button.config(state="normal")
            return

        self.status_label.config(text=f"Extracting face {face_index + 1}/{len(self.face_boxes)}...")
        
        box = self.face_boxes[face_index]
        original_filename = os.path.splitext(os.path.basename(self.filepath))[0]
        g.save_face(self.cv2_image, box, original_filename, face_index, self.output_folder)
        
        self.draw_boxes(highlight_index=face_index) # Redraw with green box

        # Schedule the next face extraction
        self.root.after(500, self.extract_and_save_one_face, face_index + 1)

    def update_faces_menu(self):
        self.faces_menu.delete(0, tk.END)
        if not self.face_boxes:
            self.faces_menu.add_command(label="No faces detected", state="disabled")
        else:
            self.faces_menu.add_command(label="Show Full Image", command=self.display_image)
            self.faces_menu.add_separator()
            for i, box in enumerate(self.face_boxes):
                self.faces_menu.add_command(label=f"Face {i+1}", command=lambda idx=i: self.display_face(idx))

    def display_face(self, index):
        if not self.original_image or index >= len(self.face_boxes):
            return

        box = self.face_boxes[index]
        # Crop the face (convert coordinates to int)
        face_img = self.original_image.crop((int(box[0]), int(box[1]), int(box[2]), int(box[3])))

        label_width = self.image_label.winfo_width()
        label_height = self.image_label.winfo_height()

        if label_width > 1 and label_height > 1:
            face_img.thumbnail((label_width - 10, label_height - 10), Image.Resampling.LANCZOS)

        photo = ImageTk.PhotoImage(face_img)
        self.image_label.config(image=photo, text="")
        self.image_label.image = photo
        self.status_label.config(text=f"Displaying Face {index + 1}")

    def open_settings(self):
        self.settings_win = tk.Toplevel(self.root)
        self.settings_win.title("Settings")
        self.settings_win.geometry("500x200")
        
        # Input Folder
        tk.Label(self.settings_win, text="Input Folder:").pack(anchor="w", padx=10)
        frame_in = tk.Frame(self.settings_win)
        frame_in.pack(fill="x", padx=10, pady=5)
        self.input_entry = tk.Entry(frame_in)
        self.input_entry.insert(0, self.input_folder)
        self.input_entry.pack(side="left", fill="x", expand=True)
        tk.Button(frame_in, text="Browse", command=self.browse_input).pack(side="right", padx=5)
        
        # Output Folder
        tk.Label(self.settings_win, text="Output Folder:").pack(anchor="w", padx=10)
        frame_out = tk.Frame(self.settings_win)
        frame_out.pack(fill="x", padx=10, pady=5)
        self.output_entry = tk.Entry(frame_out)
        self.output_entry.insert(0, self.output_folder)
        self.output_entry.pack(side="left", fill="x", expand=True)
        tk.Button(frame_out, text="Browse", command=self.browse_output).pack(side="right", padx=5)
        
        tk.Button(self.settings_win, text="Save", command=self.save_settings).pack(pady=20)

    def browse_input(self):
        d = filedialog.askdirectory(initialdir=self.input_folder)
        if d:
            self.input_entry.delete(0, tk.END)
            self.input_entry.insert(0, d)

    def browse_output(self):
        d = filedialog.askdirectory(initialdir=self.output_folder)
        if d:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, d)

    def save_settings(self):
        self.input_folder = self.input_entry.get()
        self.output_folder = self.output_entry.get()

        if not self.config.has_section("Settings"):
            self.config.add_section("Settings")
        self.config.set("Settings", "input_folder", self.input_folder)
        self.config.set("Settings", "output_folder", self.output_folder)
        with open(self.config_file, "w") as configfile:
            self.config.write(configfile)
        self.settings_win.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceExtractorApp(root)
    root.mainloop()
