import os
import subprocess
import numpy as np
from skimage.io import imread
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
import platform

software_config = {
    "meshroom": {
        "windows": {
            "install_command": "msiexec /i meshroom_installer.msi /quiet",
            "executable": r"C:\Program Files\Meshroom\meshroom_photogrammetry.exe",
            "command": "--input {image_dir} --output {output_dir}"
        },
        "linux": {
            "install_command": "sudo apt install meshroom",
            "executable": "meshroom_photogrammetry",
            "command": "--input {image_dir} --output {output_dir}"
        }
    },
    "visualsfm": {
        "windows": {
            "install_command": "visualsfm_installer.exe /S",
            "executable": r"C:\Program Files\VisualSFM\visualsfm.exe",
            "command": "{image_dir} {output_dir}"
        },
        "linux": {
            "install_command": "sudo apt install visualsfm",
            "executable": "visualsfm",
            "command": "{image_dir} {output_dir}"
        }
    },
    "colmap": {
        "windows": {
            "install_command": "colmap_installer.exe /S",
            "executable": r"C:\Program Files\COLMAP\colmap.exe",
            "command": "automatic_reconstructor --image_path {image_dir} --workspace_path {output_dir}"
        },
        "linux": {
            "install_command": "sudo apt install colmap",
            "executable": "colmap",
            "command": "automatic_reconstructor --image_path {image_dir} --workspace_path {output_dir}"
        }
    }
}

def get_os():
    return "windows" if platform.system() == "Windows" else "linux"

def install_if_not_exists(software):
    if not os.path.exists(software_config[software][get_os()]["executable"]):
        subprocess.run(software_config[software][get_os()]["install_command"], shell=True, check=True)

def create_3d_model(image_dir, output_dir, software):
    install_if_not_exists(software)
    command = software_config[software][get_os()]["command"].format(image_dir=image_dir, output_dir=output_dir)
    executable = software_config[software][get_os()]["executable"]
    if get_os() == "windows":
        executable = f"\"{executable}\""
    subprocess.run(f"{executable} {command}", shell=True, check=True)

def evaluate_3d_model(image_dir, output_dir):
    ref_image = imread(os.path.join(image_dir, os.listdir(image_dir)[0]))
    generated_images = [
        resize(imread(os.path.join(output_dir, filename)), ref_image.shape)
        for filename in os.listdir(output_dir)
        if filename.endswith((".jpg", ".png")) and not filename.startswith(("depth", "normal"))
    ]
    ssim_scores = [ssim(ref_image, img, multichannel=True) for img in generated_images]
    return np.mean(ssim_scores)

def run_photogrammetry(image_dir, output_dir, software_list):
    for software in software_list:
        create_3d_model(image_dir, output_dir, software)
        mean_ssim = evaluate_3d_model(image_dir, output_dir)
        print(f"3D model created using {software}. Mean SSIM score: {mean_ssim:.4f}")
