import yaml
from dataclasses import dataclass
from typing import List, Optional
import os

@dataclass
class Config:
    outdir: str
    device: str
    silent: bool
    image_size: int
    filelist: List[str]
    schedule: str
    niter: int
    min_conf_thr: float
    as_pointcloud: bool
    mask_sky: bool
    clean_depth: bool
    transparent_cams: bool
    cam_size: float
    scenegraph_type: str
    winsize: int
    refid: int

    def get_default_params(self):
        return (
            self.outdir,
            self.device,
            self.silent,
            self.image_size,
            self.filelist,
            self.schedule,
            self.niter,
            self.min_conf_thr,
            self.as_pointcloud,
            self.mask_sky,
            self.clean_depth,
            self.transparent_cams,
            self.cam_size,
            self.scenegraph_type,
            self.winsize,
            self.refid
        )

    def update_filelist(self, img_dir: str, start_frame: int = -1, end_frame: int = -1, stride: int = 1):
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        self.filelist = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)
                         if f.lower().endswith(image_extensions)])
        if start_frame == -1:
            start_frame = 0
            end_frame = len(self.filelist)
        self.filelist = self.filelist[start_frame:end_frame:stride]

def get_dust3r_config(config_yaml_path: str) -> Config:
    with open(config_yaml_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return Config(**config_dict)