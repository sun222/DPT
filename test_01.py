"""Compute segmentation maps for images in the input folder.
"""
import os
import glob
import cv2
import argparse

import torch
import torch.nn.functional as F

import util.io

from torchvision.transforms import Compose
from dpt.models import DPTSegmentationModel
from dpt.transforms import Resize, NormalizeImage, PrepareForNet


from torch.utils.data import Dataset, DataLoader


class ledchangeDataset(Dataset):
    
   def __init__(self, root, transform=None):
        self.root = root
        self.transforms = transform
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs1 = list(sorted(os.listdir(os.path.join(root, "images"))))
        
        
        self.masks = list(sorted(os.listdir(os.path.join(root, "annotations"))))
   def __getitem__(self, idx):
        # load images ad masks
        
        img_path1 = os.path.join(self.root, "A", self.imgs1[idx])
        img_path2 = os.path.join(self.root, "B", self.imgs2[idx])
        
        mask_path = os.path.join(self.root, "OUT", self.masks[idx])
        
        img1 = Image.open(img_path1).convert("RGB")
        img1= np.array(img1)
    
        
        img2 = Image.open(img_path2).convert("RGB")
        img2= np.array(img2)
      
        mask = Image.open(mask_path)
        mask = np.array(mask)
        mask=mask/255
#         mask=mask.reshape(768,768,1)
#         mask=mask.transpose(2,0,1)
        
        mask = torch.from_numpy(mask).long()

        if self.transforms is not None:
            img1 = self.transforms(img1)
            img2 = self.transforms(img2)


        return img1,img2,mask

   def __len__(self):
        return len(self.imgs1)  
        
        
  
# use the same transformations for train/val in this example
trans = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
])

train_set = ledchangeDataset(pre_url+'train/', transform = trans)
val_set = ledchangeDataset(pre_url+'test/', transform = trans)

image_datasets = {
  'train': train_set, 'val': val_set
}

batch_size = 16

dataloaders = {
  'train': DataLoader(train_set, batch_size=batch_size, shuffle=True),
  'val': DataLoader(val_set, batch_size=batch_size, shuffle=True)
}










def mIoU(pred_mask, mask, smooth=1e-10, n_classes=151):
    with torch.no_grad():
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(0, n_classes): #loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas
            intersect = torch.logical_and(true_class, true_label).sum().float().item()
            union = torch.logical_or(true_class, true_label).sum().float().item()
            iou_per_class.append([intersect,union])
        return iou_per_class     
#         return np.nanmean(iou_per_class)



def run(input_path, output_path, model_path, model_type="dpt_hybrid", optimize=True):
    """Run segmentation network

    Args:
        input_path (str): path to input folder
        output_path (str): path to output folder
        model_path (str): path to saved model
    """
    print("initialize")

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    net_w = net_h = 480

    # load network
    if model_type == "dpt_large":
        model = DPTSegmentationModel(
            150,
            path=model_path,
            backbone="vitl16_384",
        )
    elif model_type == "dpt_hybrid":
        model = DPTSegmentationModel(
            150,
            path=model_path,
            backbone="vitb_rn50_384",
        )
    else:
        assert (
            False
        ), f"model_type '{model_type}' not implemented, use: --model_type [dpt_large|dpt_hybrid]"

    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            PrepareForNet(),
        ]
    )

    model.eval()

    if optimize == True and device == torch.device("cuda"):
        model = model.to(memory_format=torch.channels_last)
        model = model.half()

    model.to(device)

    # get input
    img_names = glob.glob(os.path.join(input_path, "*"))
    num_images = len(img_names)

    # create output folder
    os.makedirs(output_path, exist_ok=True)

    print("start processing")

    iou_score=[[0.0,0.0] for d in range(151)]
    iou_score = np.array(iou_score)
    
    
    for ind, img_name in enumerate(img_names):

        print("  processing {} ({}/{})".format(img_name, ind + 1, num_images))

        # input
        img = util.io.read_image(img_name)
        img_input = transform({"image": img})["image"]

        # compute
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
            if optimize == True and device == torch.device("cuda"):
                sample = sample.to(memory_format=torch.channels_last)
                sample = sample.half()

            out = model.forward(sample)

            prediction = torch.nn.functional.interpolate(
                out, size=img.shape[:2], mode="bicubic", align_corners=False
            )
            prediction = torch.argmax(prediction, dim=1) + 1
            prediction = prediction.squeeze().cpu()
#             .numpy()

        # output
        filename = os.path.join(
            output_path, os.path.splitext(os.path.basename(img_name))[0]
        )
        
        mask=Image.open(filename)
        mask= np.array(mask)
        ss=mIoU(prediction, mask)
        mid=np.array(ss )
        iou_score +=ss
#         util.io.write_segm_img(filename, img, prediction, alpha=0.5)


    iou_score_final=[0 for d in range(150)]
    smooth=0
    for i in range(150): 
                intersect,union=iou_score[i+1]
                iou = (intersect + smooth) / (union +smooth)
                iou_score_final[i]=iou
    miou =  np.nanmean(iou_score_final)
    print(miou)
    print("finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--input_path", default="input", help="folder with input images"
    )

    parser.add_argument(
        "-o", "--output_path", default="output_semseg", help="folder for output images"
    )

    parser.add_argument(
        "-m",
        "--model_weights",
        default=None,
        help="path to the trained weights of model",
    )

    # 'vit_large', 'vit_hybrid'
    parser.add_argument("-t", "--model_type", default="dpt_hybrid", help="model type")

    parser.add_argument("--optimize", dest="optimize", action="store_true")
    parser.add_argument("--no-optimize", dest="optimize", action="store_false")
    parser.set_defaults(optimize=True)

    args = parser.parse_args()

    default_models = {
        "dpt_large": "weights/dpt_large-ade20k-b12dca68.pt",
        "dpt_hybrid": "weights/dpt_hybrid-ade20k-53898607.pt",
    }

    if args.model_weights is None:
        args.model_weights = default_models[args.model_type]

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # compute segmentation maps
    run(
        args.input_path,
        args.output_path,
        args.model_weights,
        args.model_type,
        args.optimize,
    )
