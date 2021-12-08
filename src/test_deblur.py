import os
import sys
import torch
import argparse
import numpy as np

from Edataloader import *
from net_deblur_RDN import *

from metrics import *

def test(deblur_net, opts):

    data = EventData(opts, opts.load_dataset_name, opts.load_bag_name, opts.is_load_sharp_image)
    dataloader = torch.utils.data.DataLoader(dataset=data, batch_size=opts.batch_sz,shuffle=False)

    if opts.is_load_sharp_image:
        PSNR_CENTER = 0
        PSNR_SEQ = 0
        SSIM_CENTER = 0
        SSIM_SEQ = 0

    iters = 0

    for results in dataloader:
        if opts.is_load_sharp_image:
            event_image01 = results[0].cuda()
            event_image12 = results[1].cuda()
            event_image23 = results[2].cuda()
            event_image34 = results[3].cuda()
            event_image45 = results[4].cuda()
            event_image56 = results[5].cuda()
            blurry = results[6].cuda()
            sharp_img0 = results[7].cuda()
            sharp_img1 = results[8].cuda()
            sharp_img2 = results[9].cuda()
            sharp_img3 = results[10].cuda()
            sharp_img4 = results[11].cuda()
            sharp_img5 = results[12].cuda()
            sharp_img6 = results[13].cuda()
            bag_name = np.array(results[14])
            image_iter = np.array(results[15])
        else:
            event_image01 = results[0].cuda()
            event_image12 = results[1].cuda()
            event_image23 = results[2].cuda()
            event_image34 = results[3].cuda()
            event_image45 = results[4].cuda()
            event_image56 = results[5].cuda()
            blurry = results[6].cuda()
            bag_name = np.array(results[7])
            image_iter = np.array(results[8])

        pred_shape_images = deblur_net(blurry, event_image01, event_image12, event_image23, event_image34, event_image45, event_image56)

        if opts.is_load_sharp_image:
            psnr_center = PSNR(pred_shape_images[3], sharp_img3)
            ssim_center = SSIM(pred_shape_images[3], sharp_img3)
            psnr_seq = (PSNR(pred_shape_images[0], sharp_img0)
                        +PSNR(pred_shape_images[1], sharp_img1)
                        +PSNR(pred_shape_images[2], sharp_img2)
                        +PSNR(pred_shape_images[3], sharp_img3)
                        +PSNR(pred_shape_images[4], sharp_img4)
                        +PSNR(pred_shape_images[5], sharp_img5)
                        +PSNR(pred_shape_images[6], sharp_img6))/7
            ssim_seq = (SSIM(pred_shape_images[0], sharp_img0)
                        +SSIM(pred_shape_images[1], sharp_img1)
                        +SSIM(pred_shape_images[2], sharp_img2)
                        +SSIM(pred_shape_images[3], sharp_img3)
                        +SSIM(pred_shape_images[4], sharp_img4)
                        +SSIM(pred_shape_images[5], sharp_img5)
                        +SSIM(pred_shape_images[6], sharp_img6))/7

            PSNR_CENTER += psnr_center
            PSNR_SEQ += psnr_seq
            SSIM_CENTER += ssim_center
            SSIM_SEQ += ssim_seq

            print(iters, '  psnr_center:', format(psnr_center,'.4f'), '  psnr_seq:', format(psnr_seq,'.4f'), '  ssim_center:', format(ssim_center,'.4f'), '  ssim_seq:', format(ssim_seq,'.4f'))

        if opts.is_save_deblurred_image:
            save_path = os.path.join(opts.save_deblurred_image_folder, opts.load_dataset_name, bag_name[0])
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            pred_shape_image0_toshow = show_img(pred_shape_images[0])
            pred_shape_image1_toshow = show_img(pred_shape_images[1])
            pred_shape_image2_toshow = show_img(pred_shape_images[2])
            pred_shape_image3_toshow = show_img(pred_shape_images[3])
            pred_shape_image4_toshow = show_img(pred_shape_images[4])
            pred_shape_image5_toshow = show_img(pred_shape_images[5])
            pred_shape_image6_toshow = show_img(pred_shape_images[6])
            cv2.imwrite(os.path.join(save_path, str(image_iter[0]).rjust(5,'0') + '_0' + '.png'), pred_shape_image0_toshow)
            cv2.imwrite(os.path.join(save_path, str(image_iter[0]).rjust(5,'0') + '_1' + '.png'), pred_shape_image1_toshow)
            cv2.imwrite(os.path.join(save_path, str(image_iter[0]).rjust(5,'0') + '_2' + '.png'), pred_shape_image2_toshow)
            cv2.imwrite(os.path.join(save_path, str(image_iter[0]).rjust(5,'0') + '_3' + '.png'), pred_shape_image3_toshow)
            cv2.imwrite(os.path.join(save_path, str(image_iter[0]).rjust(5,'0') + '_4' + '.png'), pred_shape_image4_toshow)
            cv2.imwrite(os.path.join(save_path, str(image_iter[0]).rjust(5,'0') + '_5' + '.png'), pred_shape_image5_toshow)
            cv2.imwrite(os.path.join(save_path, str(image_iter[0]).rjust(5,'0') + '_6' + '.png'), pred_shape_image6_toshow)

        iters += 1

    print('Testing done. ')
    if opts.is_load_sharp_image:
        PSNR_CENTER = PSNR_CENTER/iters
        PSNR_SEQ = PSNR_SEQ/iters
        SSIM_CENTER = SSIM_CENTER/iters
        SSIM_SEQ = SSIM_SEQ/iters
        print('PSNR_CENTER:', format(PSNR_CENTER,'.4f'), '  PSNR_SEQ:', format(PSNR_SEQ,'.4f'), '  SSIM_CENTER:', format(SSIM_CENTER,'.4f'), '  SSIM_SEQ:', format(SSIM_SEQ,'.4f'))

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_sz', type=int, default=1, help='batch size used for training')
    parser.add_argument('--crop_sz_H', type=int, default=180, help='cropped image size height')
    parser.add_argument('--crop_sz_W', type=int, default=240, help='cropped image size width')

    parser.add_argument('--data_folder_path', type=str, default='../data')
    parser.add_argument('--load_dataset_name', type=str, default='HQF')
    parser.add_argument('--load_bag_name', type=str, default='test_bags.txt')
    parser.add_argument('--is_load_sharp_image', type=bool, default=True)

    parser.add_argument('--is_test', type=bool, default=True)
    
    parser.add_argument('--is_save_deblurred_image', type=bool, default=False)
    parser.add_argument('--save_deblurred_image_folder', type=str, default='../results')

    opts = parser.parse_args()

    deblur_net = RDN_residual_deblur().cuda()
    checkpoint = torch.load('../pretrained_model/deblur.pth')
    deblur_net.load_state_dict(checkpoint)

    deblur_net.eval()
    for _,param in deblur_net.named_parameters():
        param.requires_grad = False

    test(deblur_net, opts)

if __name__ == "__main__":
    main()