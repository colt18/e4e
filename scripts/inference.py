import argparse
import os
import shutil
import sys

import dlib
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.append(".")

from configs import data_configs, paths_config
from datasets.inference_dataset import InferenceDataset
from utils.alignment import align_face
from utils.common import tensor2im
from utils.model_utils import setup_model
from utils.mp_headpose import HeadPoseDetector

sys.path.append("..")

def main(args):
    net, opts = setup_model(args.ckpt, device)
    is_cars = 'cars_' in opts.dataset_type
    generator = net.decoder
    generator.eval()
    args, data_loader = setup_data_loader(args, opts)

    # Check if latents exist
    latents_file_path = os.path.join(args.save_dir, 'latents.pt')
    if os.path.exists(latents_file_path):
        latent_codes = torch.load(latents_file_path).to(device)
    else:
        latent_codes = get_all_latents(net, data_loader, args.n_sample, is_cars=is_cars)
        torch.save(latent_codes, latents_file_path)

    if not args.latents_only:
        generate_inversions(args, generator, latent_codes, is_cars=is_cars)

    # Load the images and latents
    latents = torch.load(os.path.join('images','output','latents.pt'))
    img1 = np.array(Image.open(os.path.join('images','output','inversions','00001.jpg')))
    lat2 = latents[1]

    # Load the pose latent 
    pose_latent = torch.load(args.direction_path)
    pose_latent = pose_latent.squeeze(0)
         
    # Instantiate head pose detector
    detector = HeadPoseDetector()   

    ################################# START OPTIMIZATION #############################
    # Define optimization parameters
    learning_rate = 0.1
    max_iterations = 100
    tolerance = 1.0  # Tolerance for the yaw difference
    frames = []
    # Define initial values
    lat2_aligned = lat2.clone()  # Initialize aligned latent as lat2
    img2_aligned, _ = generator([lat2_aligned.unsqueeze(0)], input_is_latent=True, randomize_noise=False, return_latents=True)
    img2_aligned = np.array(tensor2im(img2_aligned[0]))
    img2_initial = img2_aligned.copy()

    
    _, _, yaw1 = detector.process_image(img1)  # Calculate yaw for img1

    for iteration in range(max_iterations):
        # Calculate yaw for the aligned img2
        _, _, yaw2_aligned = detector.process_image(img2_aligned)
        
        # Calculate the yaw difference between img1 and aligned img2
        yaw_diff = yaw1 - yaw2_aligned

        print (yaw_diff)
        
        if abs(yaw_diff) < tolerance:
            break  # If yaw is aligned within tolerance, exit the loop
        
        # Update the aligned latent based on yaw difference and latent_dir
        if yaw_diff > 0:
            lat2_aligned -= learning_rate * pose_latent  # Add latent_dir
        else:
            lat2_aligned += learning_rate * pose_latent  # Subtract latent_dir
        
        # Generate the aligned image from the updated latent
        img2_aligned, _ = generator([lat2_aligned.unsqueeze(0)], input_is_latent=True, randomize_noise=False, return_latents=True)
        img2_aligned = np.array(tensor2im(img2_aligned[0]))
        frames.append(Image.fromarray(img2_aligned))
        
    # At this point, img2_aligned should have a similar yaw to img1
    # imageio.mimsave("alignment_animation.gif", frames, duration=0.1)  # Adjust duration as needed
    stacked = np.hstack((img2_initial, img2_aligned))
    Image.fromarray(stacked.astype('uint8')).save('aligned.png')

def main_30degrees(args):
    if os.path.exists('images/output/'):
        shutil.rmtree('images/output/')
    os.makedirs('images/output/', exist_ok=True)

        

    net, opts = setup_model(args.ckpt, device)
    is_cars = 'cars_' in opts.dataset_type
    generator = net.decoder
    generator.eval()
    args, data_loader = setup_data_loader(args, opts)

    # Check if latents exist
    latents_file_path = os.path.join(args.save_dir, 'latents.pt')
    if os.path.exists(latents_file_path):
        latent_codes = torch.load(latents_file_path).to(device)
    else:
        latent_codes = get_all_latents(net, data_loader, args.n_sample, is_cars=is_cars)
        torch.save(latent_codes, latents_file_path)

    if not args.latents_only:
        generate_inversions(args, generator, latent_codes, is_cars=is_cars)

    # Load the image and latent
    latent = torch.load(os.path.join('images','output','latents.pt'))
    # img = np.array(Image.open(os.path.join('images','output','inversions','00001.jpg')))
    lat = latent[0]

    # Load the pose latent 
    pose_latent = torch.load(args.direction_path)
    pose_latent = pose_latent.squeeze(0)
         
    # Instantiate head pose detector
    detector = HeadPoseDetector()   

    ################################# START OPTIMIZATION #############################
    # Define optimization parameters
    learning_rate = 0.1
    max_iterations = 100
    tolerance = 1.0  # Tolerance for the yaw difference
    frames = []
    # Define initial values
    lat_aligned = lat.clone()  # Initialize aligned latent
    img_aligned, _ = generator([lat_aligned.unsqueeze(0)], input_is_latent=True, randomize_noise=False, return_latents=True)
    img_aligned = np.array(tensor2im(img_aligned[0]))
    img_initial = img_aligned.copy()

    
    _, _, yaw_init = detector.process_image(img_aligned)  # Calculate yaw for img

    for iteration in range(max_iterations):
        # Calculate yaw for the aligned img
        _, _, yaw_aligned = detector.process_image(img_aligned)
        
        # Calculate the yaw difference between img1 and aligned img2
        yaw_diff = yaw_init - yaw_aligned + args.degree

        print (yaw_diff)
        
        if abs(yaw_diff) < tolerance:
            break  # If yaw is aligned within tolerance, exit the loop
        
        # Update the aligned latent based on yaw difference and latent_dir
        if yaw_diff > 0:
            lat_aligned -= learning_rate * pose_latent  # Add latent_dir
        else:
            lat_aligned += learning_rate * pose_latent  # Subtract latent_dir
        
        # Generate the aligned image from the updated latent
        img_aligned, _ = generator([lat_aligned.unsqueeze(0)], input_is_latent=True, randomize_noise=False, return_latents=True)
        img_aligned = np.array(tensor2im(img_aligned[0]))
        frames.append(Image.fromarray(img_aligned))
        
    # At this point, img2_aligned should have a similar yaw to img1
    # imageio.mimsave("alignment_animation.gif", frames, duration=0.1)  # Adjust duration as needed
    stacked = np.hstack((img_initial, img_aligned))
    Image.fromarray(stacked.astype('uint8')).save('aligned.png')

def setup_data_loader(args, opts):
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    images_path = args.images_dir if args.images_dir is not None else dataset_args['test_source_root']
    print(f"images path: {images_path}")
    align_function = None
    if args.align:
        align_function = run_alignment
    test_dataset = InferenceDataset(root=images_path,
                                    transform=transforms_dict['transform_test'],
                                    preprocess=align_function,
                                    opts=opts)

    data_loader = DataLoader(test_dataset,
                             batch_size=args.batch,
                             shuffle=False,
                             num_workers=2,
                             drop_last=True)

    print(f'dataset length: {len(test_dataset)}')

    if args.n_sample is None:
        args.n_sample = len(test_dataset)
    return args, data_loader


def get_latents(net, x, is_cars=False):
    # print (torch.mean(x))
    # exit()
    codes = net.encoder(x)
    if net.opts.start_from_latent_avg:
        if codes.ndim == 2:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
        else:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)
    if codes.shape[1] == 18 and is_cars:
        codes = codes[:, :16, :]
    return codes


def get_all_latents(net, data_loader, n_images=None, is_cars=False):
    all_latents = []
    i = 0
    with torch.no_grad():
        for batch in data_loader:
            if n_images is not None and i > n_images:
                break
            x = batch
            # print (x[0][0])
            inputs = x.to(device).float()
            # print (inputs)
            latents = get_latents(net, inputs, is_cars)
            # print(latents)
            all_latents.append(latents)
            i += len(latents)
    return torch.cat(all_latents)


def save_image(img, save_dir, idx):
    result = tensor2im(img)
    im_save_path = os.path.join(save_dir, f"{idx:05d}.jpg")
    Image.fromarray(np.array(result)).save(im_save_path)


@torch.no_grad()
def generate_inversions(args, g, latent_codes, is_cars):
    print('Saving inversion images')
    inversions_directory_path = os.path.join(args.save_dir, 'inversions')
    os.makedirs(inversions_directory_path, exist_ok=True)
    for i in range(min(args.n_sample, len(latent_codes))):  
        imgs, _ = g([latent_codes[i].unsqueeze(0)], input_is_latent=True, randomize_noise=False, return_latents=True)
        # print(imgs.shape)
        if is_cars:
            imgs = imgs[:, :, 64:448, :]
        save_image(imgs[0], inversions_directory_path, i + 1)


def run_alignment(image_path):
    predictor = dlib.shape_predictor(paths_config.model_paths['shape_predictor'])
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--images_dir", type=str, default="images/input",
                        help="The directory of the images to be inverted")
    parser.add_argument("--save_dir", type=str, default="images/output",
                        help="The directory to save the latent codes and inversion images. (default: images_dir")
    parser.add_argument("--batch", type=int, default=1, help="batch size for the generator")
    parser.add_argument("--n_sample", type=int, default=None, help="number of the samples to infer.")
    parser.add_argument("--latents_only", action="store_true", help="infer only the latent codes of the directory")
    parser.add_argument("--align", action="store_true", help="align face images before inference")
    parser.add_argument("--ckpt", type=str, default= "checkpoints/e4e_ffhq_encode.pt", help="path to generator checkpoint")
    parser.add_argument("--direction_path", help=" Direction paths are defined by code" )
    parser.add_argument("--direction", type=str, default="pose", \
                        help="which direction to apply. opts: pose, smile, age" )
    parser.add_argument("--degree", type=int, default=0, help="degree to rotate")

    
    args = parser.parse_args()

    direction_paths = {
        "pose": "editings/interfacegan_directions/pose.pt",
        "smile": "editings/interfacegan_directions/smile.pt",
        "age": "editings/interfacegan_directions/age.pt",
        }
    
    args.direction_path = direction_paths[args.direction]

    # main(args)
    main_30degrees(args)
