import random
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
import torch
import numpy as np
from albumentations.core.transforms_interface import BasicTransform

class Cut(ImageOnlyTransform):
    def __init__(self, 
                 cutting=None,
                 always_apply=False,
                 p=1.0):
        
        super(Cut, self).__init__(always_apply, p)
        self.cutting = cutting
    
    
    def apply(self, image, **params):
        
        if self.cutting:
            image = image[self.cutting:-self.cutting,:,:]
            
        return image
            
    def get_transform_init_args_names(self):
        return ("size", "cutting")
# class LimitedFoV(object):
#     def __init__(self, fov=360.):
#         self.fov = fov

#     def __call__(self, x):
#         # print(x.shape)
#         angle = random.randint(0, 359)
#         rotate_index = int(angle / 360. * x.shape[2])
#         fov_index = int(self.fov / 360. * x.shape[2])
#         if rotate_index > 0:
#             img_shift = torch.zeros(x.shape)
#             img_shift[:,:, :rotate_index] = x[:,:, -rotate_index:]
#             img_shift[:,:, rotate_index:] = x[:,:, :(x.shape[2] - rotate_index)]
#         else:
#             img_shift = x
#         return img_shift[:,:,:fov_index]
class LimitedFoV(BasicTransform):
    def __init__(self, fov=360., always_apply=False, p=1.0):
        super().__init__(always_apply=always_apply, p=p)
        self.fov = fov

    def apply(self, img, **params):
        # 转换为 numpy 数组处理
        img = np.array(img)
        angle = random.randint(0, 359)
        rotate_index = int(angle / 360. * img.shape[1])  # 注意：numpy中的shape与torch不同
        fov_index = int(self.fov / 360. * img.shape[1])
        
        if rotate_index > 0:
            img_shift = np.zeros_like(img)
            img_shift[:, :rotate_index] = img[:, -rotate_index:]
            img_shift[:, rotate_index:] = img[:, :(img.shape[1] - rotate_index)]
        else:
            img_shift = img
            
        return img_shift[:, :fov_index]

    def get_transform_init_args_names(self):
        return ("fov",)

    @property
    def targets(self):
        return {"image": self.apply}
def get_transforms_train(image_size_sat,
                         img_size_ground,
                         mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225],
                         ground_cutting=0):
    
    
    satellite_transforms = A.Compose([

                                      A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                      A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                      A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                      A.OneOf([
                                               A.AdvancedBlur(p=1.0),
                                               A.Sharpen(p=1.0),
                                              ], p=0.3),
                                      A.OneOf([
                                               A.GridDropout(ratio=0.4, p=1.0),
                                               A.CoarseDropout(max_holes=25,
                                                               max_height=int(0.2*image_size_sat[0]),
                                                               max_width=int(0.2*image_size_sat[0]),
                                                               min_holes=10,
                                                               min_height=int(0.1*image_size_sat[0]),
                                                               min_width=int(0.1*image_size_sat[0]),
                                                               p=1.0),
                                              ], p=0.3),
                                      A.Normalize(mean, std),
                                      ToTensorV2(),
                                     ])
            

    ground_transforms = A.Compose([Cut(cutting=ground_cutting, p=1.0),
                                   LimitedFoV(fov = 90,p=1.0),
                                   A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                   A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                   A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                   A.OneOf([
                                            A.AdvancedBlur(p=1.0),
                                            A.Sharpen(p=1.0),
                                           ], p=0.3),
                                   A.OneOf([
                                            A.GridDropout(ratio=0.5, p=1.0),
                                            A.CoarseDropout(max_holes=25,
                                                            max_height=int(0.2*img_size_ground[0]),
                                                            max_width=int(0.2*img_size_ground[0]),
                                                            min_holes=10,
                                                            min_height=int(0.1*img_size_ground[0]),
                                                            min_width=int(0.1*img_size_ground[0]),
                                                            p=1.0),
                                           ], p=0.3),
                                   A.Normalize(mean, std),
                                   ToTensorV2(),
                                   ])

    return satellite_transforms, ground_transforms



def get_transforms_val(image_size_sat,
                       img_size_ground,
                       mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225],
                       ground_cutting=0):
    
    satellite_transforms = A.Compose([
                                      A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                      A.Normalize(mean, std),
                                      ToTensorV2(),
                                     ])

    ground_transforms = A.Compose([Cut(cutting=ground_cutting, p=1.0),
                                   LimitedFoV(fov = 90,p=1.0),
                                   A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                   A.Normalize(mean, std),
                                   ToTensorV2(),
                                  ])

    return satellite_transforms, ground_transforms