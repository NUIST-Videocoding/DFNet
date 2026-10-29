import os

import torch
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator
from tqdm import tqdm
import cv2
import numpy as np

import utils
from quantizer import quant_fix_dsslic, dequant_fix_dsslic
from VTM_encoder import run_vtm
import re
from dataset_path_utils import coco_metric_name, compress_paths

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SIMO_COMPRESS_ROOT = os.path.join(PROJECT_ROOT, "compress")
class Eval:
    def __init__(self, settings, index) -> None:
        self.settings = settings
        self.set_idx = index
        self.VTM_param = settings["VTM"]
        self.model, self.cfg = utils.model_loader(settings)
        utils.print_settings(settings, index)

    def forward_front(self, inputs, images, features):
        proposals, _ = self.model.proposal_generator(images, features, None)
        results, _ = self.model.roi_heads(images, features, proposals, None)
        return self.model._postprocess(results, inputs, images.image_sizes)

    def feature_quant(self,output_path):
        print("Quantizer of features maps...")
        filenames = []
        max_pixel_value = -float('inf')
        min_pixel_value = float('inf')
        with tqdm(total=len(self.data_loader.dataset), bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
            for inputs in iter(self.data_loader):
                # 自己加入的5行，断了之后重新跑，提过feature的不用再提
                # fname_temp = utils.simple_filename(inputs[0]["file_name"])
                # fea_path = f"feature/{self.set_idx}_ori/"
                # if os.path.isfile(f"{fea_path}{fname_temp}.png"):
                #     print(f"feature extraction: {fname_temp} skip (exist)")
                #     continue
                # ---------------------------------------------
                # 完整流程
                feature_map = self._feature_quant(inputs)
                # print('quant feature_map',feature_map )
                np_feature_map=feature_map.detach().cpu().numpy()
                # print(feature_map.shape)
                # filenames.append(filename)
                pbar.update()
                # Update max and min pixel values
                max_pixel_value = max(max_pixel_value, np.amax(np_feature_map))
                min_pixel_value = min(min_pixel_value, np.amin(np_feature_map))
            scale= 1023/ (max_pixel_value - min_pixel_value)
            print(f"Max pixel value in feature maps: {max_pixel_value}")
            print(f"Min pixel value in feature maps: {min_pixel_value}")
            print(f" scale in feature maps: {scale}")

        with open(f'{output_path}values.txt', 'a') as f:
            f.write(f"Max pixel value in feature maps: {max_pixel_value}\n")
            f.write(f"Min pixel value in feature maps: {min_pixel_value}\n")
            f.write(f"Scale in feature maps: {scale}\n")

        return filenames

    def _feature_quant(self, inputs):
        images = self.model.preprocess_image(inputs)
        features = self.model.backbone(images.tensor)
        # ------------MSFF、quantization----------------

        Lehance_p2,Lehance_p3,Lehance_p4,Lehance_p5=self.model.Lenhance(features)

        comp_image= self.model.all_fusion_L(Lehance_p2, Lehance_p3,Lehance_p4, Lehance_p5)
       

        return comp_image
    
    def vtm_coding(self,fea_path):
        print("VVC coding features maps...")
        filenames = []

        run_vtm(fea_path , self.VTM_param["QP"], self.VTM_param["threads"])
        return filenames

    def feature_coding_novtm(self,fea_path):
        print("Saving features maps...")
        filenames = []
        with tqdm(total=len(self.data_loader.dataset), bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
            for inputs in iter(self.data_loader):
                filenames.append(self._feature_coding(inputs,fea_path))
                print(filenames)
                pbar.update()
        return filenames

    def _feature_coding(self, inputs,fea_path):
        images = self.model.preprocess_image(inputs)
        features = self.model.backbone(images.tensor)
        #------------MSFF、quantization----------------

        Lehance_p2,Lehance_p3,Lehance_p4,Lehance_p5=self.model.Lenhance(features)

        comp_image= self.model.all_fusion_L(Lehance_p2, Lehance_p3,Lehance_p4, Lehance_p5)
       
      
        quant_path = os.path.dirname(fea_path)
        with open(f"{quant_path}/values.txt", "rt") as ap_f:
            lines = ap_f.readlines()
            ap1 = lines[1].split(":")[1][:-1]
            ap2 = lines[2].split(":")[1][:-1]

            # ap = ap.
            # ap = ap.split(":")
            # print(ap)
            min_value = float(ap1)
            scale_value = float(ap2)
            print(min_value, scale_value)
        ds_feat = quant_fix_dsslic(comp_image, min_value, scale_value)
    
        # --------------------------------------------
        fname = utils.simple_filename(inputs[0]["file_name"])
        fname_feat = f"{fea_path}/{fname}.png"
        info_path = compress_paths(SIMO_COMPRESS_ROOT, self.set_idx)["info_dir"]
        os.makedirs(info_path, exist_ok=True)
        with open(os.path.join(info_path, f"{fname}_inputs.bin"), "wb") as inputs_f:
            torch.save(inputs, inputs_f)
        utils.save_feature_map_msff_192(fname_feat, ds_feat)

        return fname_feat

    def evaluation(self, inputs,output_path):
        for fname in tqdm(inputs, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):
            self._evaluation(fname, output_path)
        
        results = self.evaluator.evaluate()

        print(results)

        # data=results
        with open(f"{output_path}{self.set_idx}_AP.txt", 'w') as f:
            # 遍历 bbox 和 segm 两部分
           
            f.write(str(results))

        self.summary_coco(output_path)        
        return 

  

    def _evaluation(self, fname,out_path):

        fname_simple = utils.simple_filename(fname)
        print(fname)
        info_path = compress_paths(SIMO_COMPRESS_ROOT, self.set_idx)["info_dir"]
        inputs_path = os.path.join(info_path, f"{fname_simple}_inputs.bin")
        with open(inputs_path, "rb") as inputs_f:
            inputs = torch.load(inputs_f)

        images = self.model.preprocess_image(inputs)
        p5_h, p5_w = self.model.backbone(images.tensor)["p5"].shape[-2:]
        # fname是VTM解压后的图片路径,features是对应的多尺度特征pyramid
        features = self.feat2feat_msfr(fname, out_path, (p5_h, p5_w))

        outputs = self.forward_front(inputs, images, features)
        self.evaluator.process(inputs, outputs)

        return outputs

    def summary_coco(self, output_path):
        with open(f"{output_path}test_ablation.csv", "a") as result_f:
            with open(f"{output_path}{self.set_idx}_AP.txt", "rt") as ap_f:
                # 读取文本 + 正则匹配bbox下的AP50数值
                text = ap_f.read()
                metric = coco_metric_name(self.set_idx)
                match = re.search(
                    rf"'{metric}'.*?'AP50':\s*(\d+\.\d+)",
                    text,
                    re.DOTALL,
                )
                
                # 提取数值并格式化（失败直接用0.0000）
                ap = format(float(match.group(1)), '.4f') if match else "0.0000"

            bpp = "0.0000"
            bpp_path = f"{output_path}{self.set_idx}_bpp.txt"
            if os.path.exists(bpp_path):
                with open(bpp_path, "rt", encoding="utf-8") as bpp_f:
                    bpp_text = bpp_f.read()
                    bpp_match = re.search(r"bpp:\s*([0-9]*\.?[0-9]+)", bpp_text)
                    if bpp_match:
                        bpp = format(float(bpp_match.group(1)), '.4f')

            result_f.write(f"{self.set_idx},{bpp},{ap}\n")
            print(f"BPP数值：{bpp}")
            print(f"AP50数值：{ap}")

                  
    def feat2feat_msfr(self, fname, fea_path, p5_size):
        # pyramid = {}
        png = cv2.imread(fname, -1).astype(np.float32)

        # single_fuse= self.feature_slice(png, [png.shape[0] // 16, png.shape[1] // 32])

        single_fuse = self.feature_slice(png, [png.shape[0] // 12, png.shape[1] // 16])
        print(single_fuse.shape)

        values_path = os.path.join(
            compress_paths(SIMO_COMPRESS_ROOT, self.set_idx)["experiment_dir"],
            "values.txt",
        )
        with open(values_path, "rt") as ap_f:
            lines = ap_f.readlines()
            ap1 = lines[1].split(":")[1][:-1]
            ap2 = lines[2].split(":")[1][:-1]
            min_value = float(ap1)
            scale_value = float(ap2)
        print(min_value, scale_value)

        dequant_fuse = dequant_fix_dsslic(single_fuse,min_value,scale_value)
        #unsqueeze 函数是 PyTorch 中的一个函数，用于在输入张量的形状中增加一个新的维度。这个新的维度的大小为1
        dequant_single_feature = torch.unsqueeze(dequant_fuse, 0)

        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dequant_single_feature = dequant_single_feature.cuda()

        Fuse_L = dequant_single_feature
        Fuse_H = torch.zeros_like(Fuse_L).repeat(1, 3, 1, 1)
        fuse = self.model.IWT(torch.cat([Fuse_L, Fuse_H], dim=1))
        p5_h, p5_w = p5_size
        fuse = fuse[:, :, :p5_h, :p5_w]
        pyramid = self.model.MSFR(fuse)


        pyramid["p2"] = pyramid["p2"].cuda()
        pyramid["p3"] = pyramid["p3"].cuda()
        pyramid["p4"] = pyramid["p4"].cuda()
        pyramid["p5"] = pyramid["p5"].cuda()
        pyramid["p6"] = pyramid["p6"].cuda()

        return pyramid

    def feature_slice(self, image, shape):
        height = image.shape[0]
        width = image.shape[1]

        blk_height = shape[0]
        blk_width = shape[1]
        blk = []

        for y in range(height // blk_height):
            for x in range(width // blk_width):
                y_lower = y * blk_height
                y_upper = (y + 1) * blk_height
                x_lower = x * blk_width
                x_upper = (x + 1) * blk_width
                blk.append(image[y_lower:y_upper, x_lower:x_upper])
        feature = torch.from_numpy(np.array(blk))
        return feature

class DetectEval(Eval):
    def prepare_part(self, myarg, data_name="pick"):
        print("Loading", data_name, "...")
        utils.pick_coco_exp(data_name, myarg)
   
        self.data_loader = build_detection_test_loader(self.cfg, data_name)
        self.evaluator = COCOEvaluator(data_name, self.cfg, False)
            
        self.evaluator.reset()
        print(data_name, "Loaded")
