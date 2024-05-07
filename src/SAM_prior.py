import numpy as np
import skimage
from matplotlib import pyplot as plt
from skimage.segmentation import find_boundaries
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from PIL import Image
import os


image_folder = "../../Datasets/Nuclei_1000/train/images/"
output_folder_sg = "../../Datasets/Nuclei_1000/train/seg_priors/"
output_folder_bo = "../../Datasets/Nuclei_1000/train/boundary_priors/"
os.makedirs(output_folder_sg, exist_ok=True)
os.makedirs(output_folder_bo, exist_ok=True)

# Initialize the Segment Anything model and mask generator
device = 'cpu'
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam, crop_nms_thresh=0.5, box_nms_thresh=0.5, pred_iou_thresh=0.5)


def SAMP(tI, mask_generator):
    masks = mask_generator.generate(tI)
    tI = skimage.img_as_float(tI)
    SegPrior = np.zeros_like(tI[:, :, 0])  # 
    BoundaryPrior = np.zeros_like(tI[:, :, 0])
    for maskindex in range(len(masks)):
        thismask = masks[maskindex]['segmentation']
        stability_score = masks[maskindex]['stability_score']
        thismask_ = thismask.astype(np.uint8)  # 
        SegPrior[thismask_ == 1] += stability_score
        BoundaryPrior += find_boundaries(thismask_, mode='thick').astype(np.uint8)
        BoundaryPrior[BoundaryPrior > 0] = 1

        # 
    # tI[:, :, 1] = tI[:, :, 1] + SegPrior
    # tI[:, :, 2] = tI[:, :, 2] + BoundaryPrior

    return SegPrior, BoundaryPrior  # 


# 示例使用
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.tif'))]

for filename in image_files:
    image_path = os.path.join(image_folder, filename)
    image = Image.open(image_path)
    image_array = np.array(image)

    seg_prior, boundary_prior = SAMP(image_array, mask_generator)

    # 
    seg_prior_normalized = (seg_prior - seg_prior.min()) / (seg_prior.max() - seg_prior.min())
    seg_prior_colored = plt.cm.viridis(seg_prior_normalized)[:, :, :3] * 255
    seg_prior_colored = Image.fromarray(seg_prior_colored.astype(np.uint8))

    # Save a priori area map
    seg_prior_output_path = os.path.join(output_folder_sg, f"{os.path.splitext(filename)[0]}_colored_seg_prior.png")
    seg_prior_colored.save(seg_prior_output_path)

    # Save a priori contour map
    boundary_prior_image = Image.fromarray((boundary_prior * 255).astype(np.uint8))
    boundary_prior_output_path = os.path.join(output_folder_bo, f"{os.path.splitext(filename)[0]}_boundary_prior.png")
    boundary_prior_image.save(boundary_prior_output_path)

print("Save to your path", output_folder_sg, output_folder_bo)
