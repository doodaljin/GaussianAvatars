#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
from mesh_renderer import NVDiffRenderer
import sys
from scene import Scene, GaussianModel, FlameGaussianModel, CameraDataset
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, error_map
from lpipsPyTorch import lpips
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.general_utils import PILtoTorch
from PIL import Image
# try:
#     from torch.utils.tensorboard import SummaryWriter
#     TENSORBOARD_FOUND = True
# except ImportError:
#     TENSORBOARD_FOUND = False
from skimage.feature import hog
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import numpy as np
from pathlib import Path
from torchvision.utils import save_image
from omegaconf import OmegaConf
from threestudio.models.guidance.dge_guidance import DGEGuidance
from threestudio.models.prompt_processors.stable_diffusion_prompt_processor import StableDiffusionPromptProcessor
from threestudio.utils.perceptual import PerceptualLoss
from threestudio.utils.misc import get_device
import json
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

def extract_feature(image):
    img_gray = np.mean(image.detach().numpy(), axis=0) # Convert to grayscale
    hog_features = hog(img_gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), channel_axis=None) # Adjust parameters if needed
    return hog_features

def extract_cnn_features(image_tensor, model, layer_name='avgpool'):
    """
    Extracts features from a specified layer of a pre-trained CNN.

    Args:
        image_tensor (torch.Tensor): A 3D image tensor (C x H x W) normalized to [0-1], RGB (C=3).
        model (torchvision.models.ResNet): A pre-trained ResNet model.
        layer_name (str): Name of the layer of the ResNet to extract features from. Defaults to 'avgpool'.

    Returns:
        torch.Tensor: CNN feature vector.
    """
    if image_tensor.is_cuda:
        image_tensor = image_tensor.cpu()
    image_numpy = image_tensor.detach().numpy()
    # Check if image has 3 dimensions. If not, return 0
    if len(image_numpy.shape) != 3 or image_numpy.shape[0] != 3:
        return torch.zeros(512, dtype=torch.float32)

    # Make sure the image is resized to 224x224 for the ResNet.
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    image_tensor = transform(image_tensor)
    image_tensor = image_tensor.unsqueeze(0)  # ResNet models expect batches

    # Extract feature from the layer name specified
    feature_vector = None
    if layer_name == 'avgpool':  # Avgpool layer
        with torch.no_grad():
            feature_vector = model(image_tensor).squeeze()
    else:  # Other layers
        feature_extractor = nn.Sequential(*list(model.children())[:])
        layer_features = []

        def hook(module, input, output):
            layer_features.append(output.flatten(start_dim=1))

        target_layer = None
        for name, module in feature_extractor.named_modules():
            if name == layer_name:
                target_layer = module
                break
        target_layer.register_forward_hook(hook)
        with torch.no_grad():
            model(image_tensor)
            feature_vector = torch.cat(layer_features, dim=1).squeeze()
    return feature_vector

def calculate_similarity(feature1, feature2):
    feature_vectors = np.array([feature1, feature2])
    feature_vectors = normalize(feature_vectors)
    similarity_matrix = cosine_similarity(feature_vectors)
    return similarity_matrix[0][1]

def calculate_similarity_sklearn(feature1, feature2):
    """
    Calculates cosine similarity using sklearn.

    Args:
        feature1 (torch.Tensor): First feature vector.
        feature2 (torch.Tensor): Second feature vector.

    Returns:
        float: Cosine similarity score.
    """
    # Convert PyTorch tensors to NumPy arrays
    feature1_np = feature1.detach().cpu().numpy().reshape(1, -1)
    feature2_np = feature2.detach().cpu().numpy().reshape(1, -1)
    # Calculate cosine similarity using sklearn
    similarity_score = cosine_similarity(feature1_np, feature2_np)[0][0]
    return similarity_score

def edit_dataset(edit_cameras, guidance, prompt_utils, gaussians, pipeline, edit_round, background, save_path):
    save_path = Path(save_path) / str(edit_round)
    os.makedirs(save_path, exist_ok = True)
    images = []
    original_frames = []
    for i in tqdm(range(len(edit_cameras)), desc="Editing progress"):
        view = edit_cameras[i]
        if gaussians.binding != None:
            gaussians.select_mesh_by_timestep(view.timestep)
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt_image = view.original_image.cuda()
        original_frames.append(gt_image.unsqueeze(0).permute(0, 2, 3, 1))
        images.append(rendering.unsqueeze(0).permute(0, 2, 3, 1))
    images = torch.cat(images, dim=0)
    original_frames = torch.cat(original_frames, dim=0)
    edited_images = guidance(images, original_frames, prompt_utils, cams = edit_cameras)
    for view_index in range(len(edit_cameras)):
        view = edit_cameras[view_index]
        edit_image = edited_images["edit_images"][view_index].detach().clone().permute(2, 0, 1)
        save_edited_path = os.path.normpath(view.image_path)
        split_path = save_edited_path.split(os.sep)
        save_edited_path = Path(save_path) / (str(view.timestep) + "_" + split_path[-1])
        # view.image_edited_path = save_edited_path
        edit_cameras.update_edit(view_index, save_edited_path)
        save_image(edit_image, save_edited_path)
    print("Done editing ", edit_round, "\n")
    return edit_cameras

def find_timesteps_for_editing(scene, threshold = 0.995):
    sample_cams = scene.getSampleCameras()
    print("Length timesteps: ", len(sample_cams))
    features = []
    timesteps = []
    d_timesteps = []
    resnet_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet_model.eval()
    for i in tqdm(range(len(sample_cams)), desc = "Calculating cnn features"):
        temp_feature = extract_cnn_features(sample_cams[i].original_image, resnet_model)
        features.append(temp_feature)
    for i in tqdm(range(len(sample_cams)), desc = "Filtering timesteps"):
        if i in d_timesteps:
            continue
        # print("Timesteps similar to ", i, ": ")
        cur_feature = features[i]
        timesteps.append(i)
        for j in range(i + 1, len(sample_cams)):
            cam = sample_cams[j]
            assert j == cam.timestep, "timestep is wrong"
            temp_feature = features[j]
            sim = calculate_similarity_sklearn(cur_feature, temp_feature)
            # print(j, " ", sim)
            if sim >= threshold:
                d_timesteps.append(j)
    
    with open('edit_timesteps.json', 'w', encoding='utf-8') as f:
        json.dump({"timesteps": timesteps}, f, ensure_ascii = False)


def edit(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from): 
    
    guidance = DGEGuidance(OmegaConf.create({"min_step_percent": 0.02, "max_step_percent": 0.98}))
    prompt_utils = StableDiffusionPromptProcessor(
        {
            "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
            "prompt": dataset.prompt,
            "negative_prompt": "monochrome, black and white, grayscale, cartoonish, unnatural lighting, blurred, distorted, unrealistic"
        })()
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    if dataset.bind_to_mesh:
        gaussians = FlameGaussianModel(dataset.sh_degree, dataset.disable_flame_static_offset, dataset.not_finetune_flame_params)
        mesh_renderer = NVDiffRenderer()
    else:
        gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    # train_cameras = scene.getTrainCameras()
    # edit_cameras = scene.getEditCameras()
    # loader_camera_train = DataLoader(edit_cameras, batch_size=None, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=False)
    # iter_camera_train = iter(loader_camera_train)
    # viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    edit_round = 0
    # timestep = 0
    # start_timestep = 850
    debug = 1
    debug_path = "new_edit"
    debug_counter = 0
    perceptual_loss = PerceptualLoss().eval().to(get_device())
    # find_timesteps_for_editing(scene)
    # with open("edit_timesteps.json") as tf:
    #     timesteps = json.load(tf)["timesteps"]
    # iter_timestep = iter(timesteps)
    # if start_timestep != 0:
    #     while next(iter_timestep) != start_timestep:
    #         continue
    scene.setupEditCameras("edit_timesteps.json")
    nbatch = 18
    for iteration in range(first_iter, opt.iterations + 1):        
        
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        
        # if iteration % 250 == 1:
        #     with torch.no_grad():
        #         try:
        #             timestep = next(iter_timestep)
        #         except StopIteration:
        #             iter_timestep = iter(timesteps)
        #             timestep = next(iter_timestep)
        #         edit_cameras = scene.getEditCameras(timestep = timestep)
        #         edit_cameras = edit_dataset(edit_cameras, guidance, prompt_utils, gaussians, pipe, timestep, background, dataset.edit_path) 
        #         loader_camera_train = DataLoader(edit_cameras, batch_size=None, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=False)
        #         iter_camera_train = iter(loader_camera_train)

        if iteration % 60000 == 1:
            with torch.no_grad():
                
                cameras = []
                for i in range(nbatch): 
                    first_cams = scene.getEditCamerasByBatch(nbatch, i)
                    edit_cams = edit_dataset(first_cams, guidance, prompt_utils, gaussians, pipe, edit_round*10+i, background, dataset.edit_path)
                    cameras = cameras + edit_cams.cameras
                edit_cameras = CameraDataset(cameras)
                loader_camera_train = DataLoader(edit_cameras, batch_size=None, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
                iter_camera_train = iter(loader_camera_train)
                edit_round += 1

        try:
            viewpoint_cam = next(iter_camera_train)
        except StopIteration:
            iter_camera_train = iter(loader_camera_train)
            viewpoint_cam = next(iter_camera_train)

        if gaussians.binding != None:
            gaussians.select_mesh_by_timestep(viewpoint_cam.timestep)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        # gt_image = viewpoint_cam.original_image.cuda()
        gt_image = viewpoint_cam.edited_image.cuda()
        # edit_result = guidance(image.unsqueeze(0).permute(0, 2, 3, 1), gt_image.unsqueeze(0).permute(0, 2, 3, 1), prompt_utils)
        # edit_image = edit_result["edit_images"].detach().clone().squeeze(0).permute(2, 0, 1)
        # gt_image = edit_image.cuda()
        # gt_image = get_edited_image(viewpoint_cam, edit_round, dataset.edit_path).cuda()

        if debug == 1: 
            temp_debug_path = os.path.join(debug_path, str(debug_counter))
            os.makedirs(temp_debug_path, exist_ok = True)
            save_image(image, os.path.join(temp_debug_path, "render_image.png"))
            save_image(gt_image, os.path.join(temp_debug_path, "gt_image.png"))
            debug = 0
            debug_counter += 1
        
        
        losses = {}
        losses['l1'] = l1_loss(image, gt_image) * (1.0 - opt.lambda_dssim)
        # losses['ssim'] = (1.0 - ssim(image, gt_image)) * opt.lambda_dssim
        losses['perceptual'] = perceptual_loss(image.unsqueeze(0), gt_image.unsqueeze(0)).sum() * opt.lambda_dssim

        if gaussians.binding != None:
            if opt.metric_xyz:
                losses['xyz'] = F.relu((gaussians._xyz*gaussians.face_scaling[gaussians.binding])[visibility_filter] - opt.threshold_xyz).norm(dim=1).mean() * opt.lambda_xyz
            else:
                # losses['xyz'] = gaussians._xyz.norm(dim=1).mean() * opt.lambda_xyz
                losses['xyz'] = F.relu(gaussians._xyz[visibility_filter].norm(dim=1) - opt.threshold_xyz).mean() * opt.lambda_xyz

            if opt.lambda_scale != 0:
                if opt.metric_scale:
                    losses['scale'] = F.relu(gaussians.get_scaling[visibility_filter] - opt.threshold_scale).norm(dim=1).mean() * opt.lambda_scale
                else:
                    # losses['scale'] = F.relu(gaussians._scaling).norm(dim=1).mean() * opt.lambda_scale
                    losses['scale'] = F.relu(torch.exp(gaussians._scaling[visibility_filter]) - opt.threshold_scale).norm(dim=1).mean() * opt.lambda_scale

            if opt.lambda_dynamic_offset != 0:
                losses['dy_off'] = gaussians.compute_dynamic_offset_loss() * opt.lambda_dynamic_offset

            if opt.lambda_dynamic_offset_std != 0:
                ti = viewpoint_cam.timestep
                t_indices =[ti]
                if ti > 0:
                    t_indices.append(ti-1)
                if ti < gaussians.num_timesteps - 1:
                    t_indices.append(ti+1)
                losses['dynamic_offset_std'] = gaussians.flame_param['dynamic_offset'].std(dim=0).mean() * opt.lambda_dynamic_offset_std
        
            if opt.lambda_laplacian != 0:
                losses['lap'] = gaussians.compute_laplacian_loss() * opt.lambda_laplacian
        
        losses['total'] = sum([v for k, v in losses.items()])
        losses['total'].backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * losses['total'].item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                postfix = {"Loss": f"{ema_loss_for_log:.{7}f}"}
                if 'xyz' in losses:
                    postfix["xyz"] = f"{losses['xyz']:.{7}f}"
                if 'scale' in losses:
                    postfix["scale"] = f"{losses['scale']:.{7}f}"
                if 'dy_off' in losses:
                    postfix["dy_off"] = f"{losses['dy_off']:.{7}f}"
                if 'lap' in losses:
                    postfix["lap"] = f"{losses['lap']:.{7}f}"
                if 'dynamic_offset_std' in losses:
                    postfix["dynamic_offset_std"] = f"{losses['dynamic_offset_std']:.{7}f}"
                progress_bar.set_postfix(postfix)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, losses, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
                # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            if iteration % opt.densification_interval == 0:
                size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
            if iteration % opt.edit_opacity_reset_interval == 1:
                gaussians.reset_opacity()

            # Optimizer step
            
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            if (iteration % 10000 == 0):
                debug = 1


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    # tb_writer = None
    # if TENSORBOARD_FOUND:
    #     tb_writer = SummaryWriter(args.model_path)
    # else:
    #     print("Tensorboard not available: not logging progress")
    # return tb_writer

def training_report(tb_writer, iteration, losses, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', losses['l1'].item(), iteration)
        tb_writer.add_scalar('train_loss_patches/ssim_loss', losses['ssim'].item(), iteration)
        if 'xyz' in losses:
            tb_writer.add_scalar('train_loss_patches/xyz_loss', losses['xyz'].item(), iteration)
        if 'scale' in losses:
            tb_writer.add_scalar('train_loss_patches/scale_loss', losses['scale'].item(), iteration)
        if 'dynamic_offset' in losses:
            tb_writer.add_scalar('train_loss_patches/dynamic_offset', losses['dynamic_offset'].item(), iteration)
        if 'laplacian' in losses:
            tb_writer.add_scalar('train_loss_patches/laplacian', losses['laplacian'].item(), iteration)
        if 'dynamic_offset_std' in losses:
            tb_writer.add_scalar('train_loss_patches/dynamic_offset_std', losses['dynamic_offset_std'].item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', losses['total'].item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        print("[ITER {}] Evaluating".format(iteration))
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'val', 'cameras' : scene.getValCameras()},
            {'name': 'test', 'cameras' : scene.getTestCameras()},
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                num_vis_img = 10
                image_cache = []
                gt_image_cache = []
                vis_ct = 0
                for idx, viewpoint in tqdm(enumerate(DataLoader(config['cameras'], shuffle=False, batch_size=None, num_workers=8)), total=len(config['cameras'])):
                    if scene.gaussians.num_timesteps > 1:
                        scene.gaussians.select_mesh_by_timestep(viewpoint.timestep)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx % (len(config['cameras']) // num_vis_img) == 0):
                        tb_writer.add_images(config['name'] + "_{}/render".format(vis_ct), image[None], global_step=iteration)
                        error_image = error_map(image, gt_image)
                        tb_writer.add_images(config['name'] + "_{}/error".format(vis_ct), error_image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_{}/ground_truth".format(vis_ct), gt_image[None], global_step=iteration)
                        vis_ct += 1
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()

                    image_cache.append(image)
                    gt_image_cache.append(gt_image)

                    if idx == len(config['cameras']) - 1 or len(image_cache) == 16:
                        batch_img = torch.stack(image_cache, dim=0)
                        batch_gt_img = torch.stack(gt_image_cache, dim=0)
                        lpips_test += lpips(batch_img, batch_gt_img).sum().double()
                        image_cache = []
                        gt_image_cache = []

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                lpips_test /= len(config['cameras'])          
                ssim_test /= len(config['cameras'])          
                print("[ITER {}] Evaluating {}: L1 {:.4f} PSNR {:.4f} SSIM {:.4f} LPIPS {:.4f}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--interval", type=int, default=10000, help="A shared iteration interval for test and saving results and checkpoints.")
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    if args.interval > op.iterations:
        args.interval = op.iterations // 5
    if len(args.test_iterations) == 0:
        args.test_iterations.extend(list(range(args.interval, args.iterations+1, 60000)))
    if len(args.save_iterations) == 0:
        args.save_iterations.extend(list(range(args.interval, args.iterations+1, args.interval)))
    if len(args.checkpoint_iterations) == 0:
        args.checkpoint_iterations.extend(list(range(args.interval, args.iterations+1, args.interval)))
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    edit(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
