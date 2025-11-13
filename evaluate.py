import pyiqa
import os
import argparse
from pathlib import Path
import torch
from utils import util_image
import tqdm
from pytorch_msssim import ms_ssim


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def evaluate(in_path, ref_path, ntest):
    metric_dict = {}
    metric_dict["clipiqa"] = pyiqa.create_metric('clipiqa').to(device)
    metric_dict["musiq"] = pyiqa.create_metric('musiq').to(device)
    metric_paired_dict = {}
    
    in_path = Path(in_path) if not isinstance(in_path, Path) else in_path
    assert in_path.is_dir()
    
    ref_path_list = None
    if ref_path is not None:
        ref_path = Path(ref_path) if not isinstance(ref_path, Path) else ref_path
        ref_path_list = sorted([x for x in ref_path.glob("*.[jpJP][pnPN]*[gG]")])
        if ntest is not None: ref_path_list = ref_path_list[:ntest]
        
        metric_paired_dict["psnr"]=pyiqa.create_metric('psnr', test_y_channel=True, color_space='rgb').to(device)
        metric_paired_dict["lpips"]=pyiqa.create_metric('lpips-vgg').to(device)
        metric_paired_dict["ssim"]=pyiqa.create_metric('ssim', test_y_channel=True, color_space='ycbcr' ).to(device)
        metric_paired_dict["ms-ssim"]=None
        metric_paired_dict["dists"]=pyiqa.create_metric('dists', test_y_channel=True, color_space='ycbcr' ).to(device)
        
    lr_path_list = sorted([x for x in in_path.glob("*.[jpJP][pnPN]*[gG]")])
    if ntest is not None: lr_path_list = lr_path_list[:ntest]
    
    print(f'Find {len(lr_path_list)} images in {in_path}')
    result = {}
    # 파일 저장을 위한 경로 지정
    with open("/root/workspace/DiffEIC/metric_detail.txt", "w") as f_out:
        for i in tqdm.tqdm(range(len(lr_path_list))):
            in_path = lr_path_list[i]
            ref_path = ref_path_list[i] if ref_path_list is not None else None

            im_in = util_image.imread(in_path, chn='rgb', dtype='float32')
            im_in_tensor = util_image.img2tensor(im_in).to(device)

            f_out.write(f"Image: {Path(in_path).name}\n")  # 이미지 이름 작성

            for key, metric in metric_dict.items():
                with torch.cuda.amp.autocast():
                    val = metric(im_in_tensor).item()
                    result[key] = result.get(key, 0) + val

            if ref_path is not None:
                im_ref = util_image.imread(ref_path, chn='rgb', dtype='float32')
                im_ref_tensor = util_image.img2tensor(im_ref).to(device)

                for key, metric in metric_paired_dict.items():
                    if key == "ms-ssim":
                        with torch.cuda.amp.autocast():
                            val = ms_ssim(im_in_tensor, im_ref_tensor, data_range=1.0).item()
                    else:
                        val = metric(im_in_tensor, im_ref_tensor).item()

                    result[key] = result.get(key, 0) + val

                    if key in ["lpips", "dists"]:
                        f_out.write(f"  {key.upper()}: {val:.5f}\n")  # 보기 좋게 정렬

            f_out.write("\n")  # 이미지 간 줄 바꿈


    # 평균 출력
    for key, res in result.items():
        print(f"{key}: {res/len(lr_path_list):.5f}")

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',"--in_path", type=str, required=True)
    parser.add_argument("-r", "--ref_path", type=str, default=None)
    parser.add_argument("--ntest", type=int, default=None)
    args = parser.parse_args()
    evaluate(args.in_path, args.ref_path, args.ntest)
    


