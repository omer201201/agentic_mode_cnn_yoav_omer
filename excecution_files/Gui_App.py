import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox

# Get the directory of this script (excecution_files) and find its parent (Organized_project)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# Add the project root to sys.path so Python can find 'basic_pipeline' and 'excecution_files'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now these imports will work regardless of where the script is executed from
try:
    from basic_pipeline.pipline_yolo_resnet import BaselinePipeline
    from excecution_files.main import IntegratedGate
except ImportError as e:
    messagebox.showerror("Import Error", f"Failed to import modules:\n{e}\n\nMake sure paths are correct.")
    sys.exit(1)


class PipelineGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Pipeline System")
        self.root.geometry("350x300")
        self.root.eval('tk::PlaceWindow . center')

        self.main_menu()

    def clear_window(self):
        """Removes all widgets from the current window."""
        for widget in self.root.winfo_children():
            widget.destroy()

    # ==========================
    # MENU VIEWS
    # ==========================

    def main_menu(self):
        self.clear_window()
        self.root.title("Main Menu")

        tk.Label(self.root, text="Select Operation Mode", font=("Helvetica", 14, "bold")).pack(pady=20)

        tk.Button(self.root, text="Live Feed", width=20, height=2, command=self.live_feed_menu).pack(pady=10)
        tk.Button(self.root, text="Folder", width=20, height=2, command=self.folder_menu).pack(pady=10)
        tk.Button(self.root, text="Exit", width=20, height=2, fg="white", bg="#d9534f", command=self.root.quit).pack(
            pady=10)

    def live_feed_menu(self):
        self.clear_window()
        self.root.title("Live Feed")

        tk.Label(self.root, text="Live Feed pipelines", font=("Helvetica", 14, "bold")).pack(pady=20)

        tk.Button(self.root, text="Basic pipeline", width=20, height=2, command=self.run_basic_camera).pack(pady=10)
        tk.Button(self.root, text="Gated pipeline", width=20, height=2, command=self.run_gated_camera).pack(pady=10)
        tk.Button(self.root, text="Exit", width=20, height=2, command=self.main_menu).pack(pady=10)

    def folder_menu(self):
        self.clear_window()
        self.root.title("Folder Processing")

        tk.Label(self.root, text="Folder Pipelines", font=("Helvetica", 14, "bold")).pack(pady=20)

        tk.Button(self.root, text="Basic pipeline", width=20, height=2, command=self.run_basic_folder).pack(pady=10)
        tk.Button(self.root, text="Gated pipeline", width=20, height=2, command=self.run_gated_folder).pack(pady=10)
        tk.Button(self.root, text="Exit", width=20, height=2, command=self.main_menu).pack(pady=10)

    # ==========================
    # EXECUTION LOGIC
    # ==========================

    def run_basic_camera(self):
        try:
            pipeline = BaselinePipeline()
            pipeline.run_on_camera()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run Basic Pipeline on camera:\n{e}")

    def run_gated_camera(self):
        try:
            system = IntegratedGate()
            system.run_on_camera()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run Gated Pipeline on camera:\n{e}")

    def process_folder_logic(self, pipeline_class):
        source_folder = filedialog.askdirectory(title="Select Source Folder for Processing")
        if not source_folder:
            return

        output_folder = os.path.join(source_folder, "output")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        try:
            system = pipeline_class()
            system.run_on_folder(source_folder, output_folder)

            messagebox.showinfo(
                "Processing Complete",
                f"Successfully processed images!\nAnswers and certainty percentages saved to:\n{output_folder}"
            )
        except TypeError as e:
            messagebox.showerror("Signature Error", f"Parameter error. Does run_on_folder take (input, output)?\n{e}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process folder:\n{e}")

    def run_basic_folder(self):
        self.process_folder_logic(BaselinePipeline)

    def run_gated_folder(self):
        self.process_folder_logic(IntegratedGate)


if __name__ == "__main__":
    root = tk.Tk()
    app = PipelineGUI(root)
    root.mainloop()