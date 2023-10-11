import sys
import argparse
from pathlib import Path
import torch
import numpy as np
import sys
import os
import dlib

sys.path.append(".")
sys.path.append("..")

from configs import data_configs, paths_config
from datasets.inference_dataset import InferenceDataset
from torch.utils.data import DataLoader
from utils.model_utils import setup_model
from utils.common import tensor2im
from utils.alignment import align_face
from utils.mp_headpose import HeadPoseDetector
from PIL import Image
import torchvision.transforms as transforms


def main(args):
    # Get the directory of the current script
    script_dir = Path(__file__).resolve().parent

    args.ckpt = script_dir / '..' / args.ckpt
    net, opts = setup_model(args.ckpt, device)
    is_cars = 'cars_' in opts.dataset_type
    generator = net.decoder
    generator.eval()
    args, data_loader = setup_data_loader(args, opts)

    # Create latent codes
    with torch.no_grad():
        for batch in data_loader:          
            x = batch
            inputs = x.to(device).float()
            latents = get_latents(net, inputs, is_cars)

    latent_codes = get_all_latents(net, data_loader, args.n_sample, is_cars=is_cars)
    torch.save(latent_codes, latents_file_path)

    if not args.latents_only:
        generate_inversions(args, generator, latent_codes, is_cars=is_cars)

    # Get the directory of the current script
    script_dir = Path(__file__).resolve().parent

    # Load the pose latent 
    pose_latent = torch.load(script_dir / '..' / 'editings' / 'interfacegan_directions' / 'pose.pt')
    pose_latent = pose_latent.squeeze(0)

    # Load the latents 
    latents = torch.load(script_dir / '..' / 'images' / 'output' / 'latents.pt')

    # Get a latent from latents file
    latent = latents[0]
    print ('lat1'*10)
    print (torch.mean(latents))

    # Open the image
    img = Image.open(script_dir / '..' / 'images' / 'input' / 'asian.png')
        
    # Resize the image array to shape (256, 256, 3)
    # resized_array = np.resize(img, (256, 256, 3))/255

    with torch.no_grad():        
        # Convert the resized NumPy array to a PyTorch tensor
        # x = torch.from_numpy(resized_array).permute(2, 0, 1).unsqueeze(0).to(device).float()
        # Define a sequence of transformations
        transform = transforms.Compose([
				transforms.ToTensor(),
                transforms.Resize((256, 256)),				
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        print (np.array(img).shape)
        x = transform(np.asarray(img))
        # print ("result transfomr"*10)
        # print (x)
        # exit ()
        codes = net.encoder(x)

        if net.opts.start_from_latent_avg:
            if codes.ndim == 2:
                codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
            else:
                codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)
        if codes.shape[1] == 18 and is_cars:
            codes = codes[:, :16, :]
   
    latent = codes
    print ('lat2'*10)

    print (np.mean(latent))
    exit()
    # Edit the latent with the direction
    rotated_latent = latent + pose_latent*0

    # Generate the image
    img, _ = generator([rotated_latent.unsqueeze(0)], input_is_latent=True, randomize_noise=False, return_latents=True)

    # Get np.array for generated image
    result = tensor2im(img[0])

    # Save the result for generated image 
    Image.fromarray(np.array(result)).save('imggggg.png')

    # Instantiate head pose detector
    detector = HeadPoseDetector()
    
    # Get roll, pitch, yaw vales
    roll, pitch, yaw = detector.process_image(np.array(result))

    # Print roll, pitch, yaw values    
    print (f"roll: {roll}, pitch: {pitch}, yaw: {yaw}")


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
            inputs = x.to(device).float()
            latents = get_latents(net, inputs, is_cars)
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
    parser.add_argument("--direction", type=str, default="pose", \
                        help="which direction to apply. opts: pose, smile, age" )
    parser.add_argument("--direction_path", help=" Direction paths are defined by code" )
    
    args = parser.parse_args()

    direction_paths = {
        "pose": "../editings/interfacegan_directions/pose.pt",
        "smile": "../editings/interfacegan_directions/smile.pt",
        "age": "../editings/interfacegan_directions/age.pt",
        }

    args.direction_path = direction_paths[args.direction]
    main(args)
