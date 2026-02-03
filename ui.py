import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import face_recognition
import os
import dotenv # Added for loading .env file
import configparser

class FaceExtractorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Extractor")
        self.root.minsize(800, 600)
        self.center_window(self.root) # Center the main window on screen

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
        self.file_menu.add_command(label="Settings", command=self.open_settings)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.root.quit)

        # Faces Menu
        self.faces_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Faces", menu=self.faces_menu)
        self.update_faces_menu()

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
        # Load environment variables from .env file
        dotenv.load_dotenv()

        self.config = configparser.ConfigParser()
        if os.path.exists(self.config_file):
            self.config.read(self.config_file)
        else:
            # Ensure 'Settings' section exists if config.ini is new
            if not self.config.has_section("Settings"):
                self.config.add_section("Settings")

        # Load input_folder and output_folder from config.ini, with fallbacks
        self.input_folder = self.config.get("Settings", "input_folder", fallback=os.getcwd())
        self.output_folder = self.config.get("Settings", "output_folder", fallback=os.path.join(os.getcwd(), "faces"))

        # Construct postgres_dsn using environment variables, with fallbacks
        # Prioritize environment variables from .env (loaded by dotenv)
        pg_user = os.getenv("POSTGRES_USER", "postgres")
        pg_password = os.getenv("POSTGRES_PASSWORD", "postgres") # Use a safe default if not found
        pg_host = os.getenv("POSTGRES_HOST", "localhost")
        pg_port = os.getenv("POSTGRES_PORT", "5433") # Default to 5432 as per docker-compose.yml
        pg_db = os.getenv("POSTGRES_DB", "national_memory")

        self.postgres_dsn = f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_db}"

        # Update config.ini with current values if it was just created or values were missing
        self.config.set("Settings", "input_folder", self.input_folder)
        self.config.set("Settings", "output_folder", self.output_folder)
        with open(self.config_file, "w") as configfile:
            self.config.write(configfile)

        face_recognition.configure_postgres(self.postgres_dsn)
        self.face_match_threshold = float(self.config.get("Settings", "face_match_threshold", fallback=0.4))

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
        # This check prevents recursive calls during initial setup
        # and ensures display_image is only called when the label has valid dimensions.
        if self.image_label.winfo_width() > 1 and self.image_label.winfo_height() > 1:
            self.display_image()

    def center_window(self, win):
        win.update_idletasks()  # Ensure window dimensions are calculated

        screen_width = win.winfo_screenwidth()
        screen_height = win.winfo_screenheight()
        
        win_width = win.winfo_width()
        win_height = win.winfo_height()

        center_x = (screen_width // 2) - (win_width // 2)
        center_y = (screen_height // 2) - (win_height // 2)
        win.geometry(f'+{center_x}+{center_y}')

    def extract_faces(self):
        if not self.filepath:
            messagebox.showwarning("Warning", "Please select an image first.")
            return

        self.extract_button.config(state="disabled")
        self.select_button.config(state="disabled")
        self.status_label.config(text="Detecting faces...")
        self.root.update_idletasks()

        self.cv2_image, self.face_boxes = face_recognition.get_face_boxes(self.filepath)

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
        face_recognition.save_face(self.cv2_image, box, original_filename, face_index, self.output_folder)
        face_recognition.index_extracted_face(self.cv2_image, box, self.filepath, face_index)
        
        self.draw_boxes(highlight_index=face_index) # Redraw with green box

        # Schedule the next face extraction
        self.root.after(500, self.extract_and_save_one_face, face_index + 1)

    def view_faces(self):
        if os.path.exists(self.output_folder):
            os.startfile(self.output_folder)
        else:
            messagebox.showinfo("Info", "Output folder not found.")

    def browse_file_for_entry(self, entry_widget):
        filename = filedialog.askopenfilename(
            initialdir=self.input_folder,
            title="Select Image",
            filetypes=(("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*"))
        )
        if filename:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, filename)

    def match_faces(self):
        match_win = tk.Toplevel(self.root)
        match_win.title("Match Faces")
        match_win.geometry("500x200")
        self.center_window(match_win) # Center the window

        tk.Label(match_win, text="Select Face 1:").grid(row=0, column=0, padx=10, pady=10, sticky="e")
        entry1 = tk.Entry(match_win, width=40)
        entry1.grid(row=0, column=1, padx=10, pady=10)
        tk.Button(match_win, text="Browse", command=lambda: self.browse_file_for_entry(entry1)).grid(row=0, column=2, padx=10, pady=10)

        tk.Label(match_win, text="Select Face 2:").grid(row=1, column=0, padx=10, pady=10, sticky="e")
        entry2 = tk.Entry(match_win, width=40)
        entry2.grid(row=1, column=1, padx=10, pady=10)
        tk.Button(match_win, text="Browse", command=lambda: self.browse_file_for_entry(entry2)).grid(row=1, column=2, padx=10, pady=10)

        def perform_match():
            path1 = entry1.get()
            path2 = entry2.get()
            if path1 and path2:
                result = face_recognition.match_faces(path1, path2)
                messagebox.showinfo("Match Result", result)
                match_win.grab_release() # Release grab before destroying
                match_win.destroy()

        tk.Button(match_win, text="Match", command=perform_match).grid(row=2, column=1, pady=20)
        match_win.lift() # Bring to front
        self.root.wait_window(match_win) # Pause main window until this is closed

    def find_face(self):
            find_win = tk.Toplevel(self.root)
            find_win.title("Find Face in Crowd")
            find_win.geometry("500x200")
            self.center_window(find_win) 
            find_win.transient(self.root)
            find_win.grab_set()
            find_win.focus_set()

            tk.Label(find_win, text="Target Face:").grid(row=0, column=0, padx=10, pady=10, sticky="e")
            entry_target = tk.Entry(find_win, width=40)
            entry_target.grid(row=0, column=1, padx=10, pady=10)
            tk.Button(find_win, text="Browse", command=lambda: self.browse_file_for_entry(entry_target)).grid(row=0, column=2, padx=10, pady=10)

            tk.Label(find_win, text="Crowd Image:").grid(row=1, column=0, padx=10, pady=10, sticky="e")
            entry_crowd = tk.Entry(find_win, width=40)
            entry_crowd.grid(row=1, column=1, padx=10, pady=10)
            tk.Button(find_win, text="Browse", command=lambda: self.browse_file_for_entry(entry_crowd)).grid(row=1, column=2, padx=10, pady=10)

            def perform_search():
                target = entry_target.get()
                crowd = entry_crowd.get()
                
                # Get the threshold from the entry, or use the global setting if empty/invalid
                override_threshold_str = threshold_entry.get()
                current_threshold = self.face_match_threshold
                result_text = ""
                result_image_path = None
                if target and crowd:
                    try:
                        if override_threshold_str:
                            current_threshold = float(override_threshold_str)
                        if not (0.0 <= current_threshold <= 1.0):
                            raise ValueError("Threshold must be between 0.0 and 1.0")
                    except ValueError as e:
                        messagebox.showerror("Invalid Threshold", f"Please enter a valid number between 0.0 and 1.0 for threshold. Error: {e}")
                        return

                    result_text, result_image_path = face_recognition.find_face_in_crowd(
                        target, crowd, threshold=current_threshold
                    )
                    
                    # 2. Handle the "temp" folder logic
                    if result_image_path and os.path.exists(result_image_path):
                        temp_dir = os.path.join(os.getcwd(), "temp")
                        if not os.path.exists(temp_dir):
                            os.makedirs(temp_dir)
                        
                        # Define new path in temp folder
                        new_path = os.path.join(temp_dir, os.path.basename(result_image_path))
                        
                        # Move the file from root to temp
                        try:
                            os.replace(result_image_path, new_path)
                            result_image_path = new_path # Update path for the display method
                        except Exception as e:
                            print(f"Error moving file: {e}")

                    # 3. Show message and update the main application screen
                    messagebox.showinfo("Search Result", result_text)
                    
                    if result_image_path and os.path.exists(result_image_path):
                        self.display_result_image(result_image_path)
                    
                    find_win.grab_release()
                    find_win.destroy()

            # Add threshold entry to the dialog
            tk.Label(find_win, text="Override Threshold:").grid(row=2, column=0, padx=10, pady=10, sticky="e")
            threshold_entry = tk.Entry(find_win, width=10)
            threshold_entry.insert(0, str(self.face_match_threshold)) # Default to current global threshold
            threshold_entry.grid(row=2, column=1, padx=10, pady=10, sticky="w")

            tk.Button(find_win, text="Find", command=perform_search).grid(row=3, column=1, pady=20)
            find_win.lift()
            self.root.wait_window(find_win)

    def update_faces_menu(self):
        self.faces_menu.delete(0, tk.END)
        self.faces_menu.add_command(label="View faces", command=self.view_faces)
        self.faces_menu.add_command(label="Match faces", command=self.match_faces)
        self.faces_menu.add_command(label="Find a face", command=self.find_face)

        if self.face_boxes:
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

    def display_result_image(self, image_path):
        """Loads and displays a result image in the main application window."""
        if not os.path.exists(image_path):
            messagebox.showerror("Error", f"Result image not found: {image_path}")
            return
        
        self.filepath = image_path # Set filepath to the result image
        self.original_image = Image.open(self.filepath).convert("RGB")
        self.image_with_boxes = self.original_image.copy() # No boxes initially, but keeps consistent
        self.face_boxes = [] # Clear face boxes as this is a result image
        self.display_image()
        self.status_label.config(text=f"Displayed result: {os.path.basename(image_path)}")

    def open_settings(self):
        self.settings_win = tk.Toplevel(self.root)
        self.settings_win.title("Settings")
        self.settings_win.geometry("500x350")
        self.center_window(self.settings_win) # Center the window
        # Make the settings window modal and on top
        self.settings_win.transient(self.root)
        self.settings_win.grab_set()
        self.settings_win.focus_set()
        self.settings_win.lift()
        
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
        
        # Postgres DSN
        tk.Label(self.settings_win, text="Postgres DSN:").pack(anchor="w", padx=10)
        # Display the resolved DSN, but make it read-only as it's controlled by .env
        # This prevents saving sensitive credentials from the UI to config.ini.
        self.dsn_entry = tk.Entry(self.settings_win, state='readonly')
        self.dsn_entry.insert(0, self.postgres_dsn)
        self.dsn_entry.pack(fill="x", padx=10, pady=5)

        # Face Match Threshold
        tk.Label(self.settings_win, text="Face Match Threshold:").pack(anchor="w", padx=10)
        self.threshold_entry = tk.Entry(self.settings_win)
        self.threshold_entry.insert(0, str(self.face_match_threshold))
        self.threshold_entry.pack(fill="x", padx=10, pady=5)
        
        tk.Button(self.settings_win, text="Save", command=self.save_settings).pack(pady=20)

    def browse_input(self):
        # Make the settings window modal and on top
        self.settings_win.transient(self.root)
        self.settings_win.grab_set()
        self.settings_win.focus_set()
        self.settings_win.lift()
        d = filedialog.askdirectory(initialdir=self.input_folder)
        if d:
            self.input_entry.delete(0, tk.END)
            self.input_entry.insert(0, d)

    def browse_output(self):
        d = filedialog.askdirectory(initialdir=self.output_folder)
        # Make the settings window modal and on top
        self.settings_win.transient(self.root)
        self.settings_win.grab_set()
        self.settings_win.focus_set()
        if d:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, d)

    def save_settings(self):
        # Release grab and destroy window
        self.settings_win.grab_release()
        self.settings_win.destroy()

        # Save settings logic (moved from open_settings)
        self.input_folder = self.input_entry.get()
        self.output_folder = self.output_entry.get()
        self.face_match_threshold = float(self.threshold_entry.get()) # Get updated threshold
        if not self.config.has_section("Settings"):
            self.config.add_section("Settings")
        self.config.set("Settings", "input_folder", self.input_folder)
        self.config.set("Settings", "output_folder", self.output_folder)
        self.config.set("Settings", "face_match_threshold", str(self.face_match_threshold))
        with open(self.config_file, "w") as configfile:
            self.config.write(configfile)
        # Release grab and destroy window
        self.settings_win.grab_release()
        self.settings_win.destroy()
        # Reconfigure postgres with potentially new DSN (though ideally from .env)
        # This line was moved from the end of open_settings to here.
        face_recognition.configure_postgres(self.postgres_dsn)
        self.settings_win.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceExtractorApp(root)
    root.mainloop()
