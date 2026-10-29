import threading
from tqdm import tqdm
import os
import glob
import argparse
from PIL import Image, ImageOps
import subprocess as subp
import sys
from time import sleep

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ENCODER_APP = os.path.join(PROJECT_ROOT, "EncoderAppStatic")
DECODER_APP = os.path.join(PROJECT_ROOT, "DecoderAppStatic")
ENCODER_CONFIG = os.path.join(PROJECT_ROOT, "encoder_intra_vtm.cfg")
BQMALL_CONFIG = os.path.join(PROJECT_ROOT, "BQMall.cfg")


def vtm_stage_dir(path, qp, stage):
    """Return the QP-specific VTM output directory for a feature directory.

    The feature directory may already contain ``_<qp>_ori`` (for example,
    ``ourseg.s2_an_30_ori``).  In that case the QP must not be appended a
    second time.  For an older ``_<dataset>_ori`` directory we still append
    the QP so both naming conventions remain usable.
    """
    normalized = os.path.normpath(os.fspath(path))
    if normalized.endswith("_ori"):
        stem = normalized[:-len("_ori")]
    elif normalized.endswith("ori"):
        stem = normalized[:-len("ori")]
    else:
        raise ValueError(f"VTM feature path must end with '_ori': {path}")

    qp_suffix = f"_{qp}"
    if not stem.endswith(qp_suffix):
        stem = f"{stem}{qp_suffix}"
    return f"{stem}_{stage}{os.sep}"


class Worker(threading.Thread):
    def __init__(self, file_path, path, qp):
        super().__init__()
        self.file_path = file_path
        self.path = path
        self.qp = qp

    def run(self):

        file_name = os.path.basename(self.file_path)[:-4]
        
        img = Image.open(self.file_path)

        width = img.size[0]
        height = img.size[1] 

        # bitstream_path = self.path.replace("ori", "bit/")
        # recon_path = self.path.replace("ori", "rec/")
        # temp_path = self.path.replace("ori", "tmp/")

        bitstream_path = vtm_stage_dir(self.path, self.qp, "bit")
        recon_path = vtm_stage_dir(self.path, self.qp, "rec")
        temp_path = vtm_stage_dir(self.path, self.qp, "tmp")

        stdout_vtm = open(f"{temp_path}vtm_log.txt", 'a')
        stdout_fmp = open(f"{temp_path}ffmpeg_log.txt", 'a')

        #-------------------1. Convert png to yuv-------------------
        if (os.path.exists(f"{temp_path}{file_name}_yuv.yuv")): os.remove(f"{temp_path}{file_name}_yuv.yuv")
        subp.run(f"/ffmpeg-4.2.2/bin/ffmpeg -i {self.file_path} -f "
                 f"rawvideo -pix_fmt gray16le -dst_range 1 {temp_path}{file_name}_yuv.yuv", shell = True, stdout = stdout_fmp, stderr = stdout_fmp)
        # Encoding
        #-------------------2.Encoding and decoding----------------
        subp.run(f'"{ENCODER_APP}" -c "{ENCODER_CONFIG}" -c "{BQMALL_CONFIG}" '
                 f"-i {temp_path}{file_name}_yuv.yuv -wdt {width} -hgt" \
                 f" {height} -b {bitstream_path}{file_name}.bin -o {temp_path}{file_name}_rec.yuv -q {self.qp} -f 1 " \
                 f"-fr 1", stdout=stdout_vtm, shell=True)
        
        # # Decoding
        subp.run(f'"{DECODER_APP}" -b {bitstream_path}{file_name}.bin -o {temp_path}{file_name}_rec.yuv', stdout = stdout_vtm, shell = True)

        #-------------------3. Convert yuv to png-------------------
        if (os.path.exists(f"{recon_path}{file_name}.png")): os.remove(f"{recon_path}{file_name}.png")
        subp.run(f"/ffmpeg-4.2.2/bin/ffmpeg -f rawvideo -pix_fmt gray16le -s"
                 f" {width}x{height} -src_range 1 -i {temp_path}{file_name}_rec.yuv -frames 1 -pix_fmt gray16le {recon_path}{file_name}.png", shell = True, stdout = stdout_fmp, stderr = stdout_fmp)
        # Remove tmp files
        try:
            os.remove(f"{temp_path}{file_name}_yuv.yuv")
        except OSError:
            pass
        try:
            os.remove(f"{temp_path}{file_name}_rec.yuv")
        except OSError:
            pass

def run_vtm(path, qp, threads):
    FileList = glob.glob(f"{path}/*.png")

    bitstream_path = vtm_stage_dir(path, qp, "bit")
    recon_path = vtm_stage_dir(path, qp, "rec")
    temp_path = vtm_stage_dir(path, qp, "tmp")

    os.makedirs(bitstream_path, exist_ok = True)
    os.makedirs(recon_path, exist_ok = True)
    os.makedirs(temp_path,   exist_ok = True)

    workers = []
    for file_path in tqdm(FileList, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):

        file_name = os.path.basename(file_path)[:-4] 
        
        if os.path.isfile(f"{recon_path}{file_name}.png"):
            print(f"{file_name} skip (exist)")
            continue

        while (threads + 1 < threading.active_count()): sleep(1)
        
        file_path = file_path.encode('utf-8','backslashreplace').decode().replace("\\","/")     

        t = Worker(file_path, path, qp)
        workers.append(t)
        t.start()

    # 等待所有线程完成
    for worker in workers:
        worker.join()

