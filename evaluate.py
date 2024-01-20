import torch
import argparse
import yaml

import numpy as np

from models.create_fasterrcnn_model import create_model

from datasets import (
    create_valid_dataset, create_valid_loader
)

from torch_utils.engine import evaluate
from utils.general import set_dir


torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser()
    # Data config file
    parser.add_argument(
        '--data', 
        default='data_configs/test_image_config.yaml',
        help='(optional) path to the data config file'
    )
    # Base model path
    parser.add_argument(
        '-m', '--model', 
        default='fasterrcnn_resnet50_fpn',
        help='name of the model'
    )
    # Model weights path
    parser.add_argument(
        '-mw', '--weights', 
        required=True,
        help='path to trained checkpoint weights if providing custom YAML file'
    )
    # Image size
    parser.add_argument(
        '-ims', '--imgsz', 
        default=640, 
        type=int, 
        help='image size to feed to the network'
    )
    # Num workers
    parser.add_argument(
        '-w', '--workers', default=4, type=int,
        help='number of workers for data processing/transforms/augmentations'
    )
    # Batch size
    parser.add_argument(
        '-b', '--batch', 
        default=8, 
        type=int, 
        help='batch size to load the data'
    )
    # Override device
    parser.add_argument(
        '-d', '--device', 
        default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        help='computation/training device, default is GPU if GPU present'
    )
    # Square training
    parser.add_argument(
        '-st', '--square-training',
        dest='square_training',
        action='store_true',
        help='Resize images to square shape instead of aspect ratio resizing \
              for single image training. For mosaic training, this resizes \
              single images to square shape first then puts them on a \
              square canvas.'
    )
    # Project name
    parser.add_argument(
        '-n', '--name', 
        default=None, 
        type=str, 
        help='evaluation result dir name in outputs/training/, (default res_#)'
    )
    # Project directory
    parser.add_argument(
        '--project-dir',
        dest='project_dir',
        default=None,
        help='save resutls to custom dir instead of `outputs` directory, \
              --project-dir will be named if not already present',
        type=str
    )
    
    # Parse arguments
    args = vars(parser.parse_args())
    
    # Load the data configurations
    with open(args['data']) as file:
        data_configs = yaml.safe_load(file)
        
    # Validation settings and constants.
    try: # Use test images if present.
        VALID_DIR_IMAGES = data_configs['TEST_DIR_IMAGES']
        VALID_DIR_LABELS = data_configs['TEST_DIR_LABELS']
    except: # Else use the validation images.
        VALID_DIR_IMAGES = data_configs['VALID_DIR_IMAGES']
        VALID_DIR_LABELS = data_configs['VALID_DIR_LABELS']
    NUM_CLASSES = data_configs['NC']
    CLASSES = data_configs['CLASSES']
    SAVE_VALID_PREDICTIONS = data_configs['SAVE_VALID_PREDICTION_IMAGES']
    COLORS = np.random.uniform(0, 1, size=(len(CLASSES), 3))
    NUM_WORKERS = args['workers']
    DEVICE = args['device']
    BATCH_SIZE = args['batch']

    # Model configurations
    IMAGE_SIZE = args['imgsz']
    
    # Set the output directory
    OUT_DIR = set_dir(type="evaluation", dir_name=args['name'], project_dir=args['project_dir'])
    
    # Load the pretrained model
    create_model = create_model[args['model']]
    
    # Create & load the model
    model = create_model(num_classes=NUM_CLASSES, coco_model=False)
    checkpoint = torch.load(args['weights'], map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Send model to specified device
    model.to(DEVICE).eval()
    
    # Setup the validation dataset
    valid_dataset = create_valid_dataset(
        VALID_DIR_IMAGES, 
        VALID_DIR_LABELS, 
        IMAGE_SIZE, 
        CLASSES,
        square_training=args['square_training']
    )
    
    # Create the dataset loader for dataset with specified batch size & num workers
    valid_loader = create_valid_loader(valid_dataset, BATCH_SIZE, NUM_WORKERS)
    
    # Evaluate the model on the validation dataset
    stats, val_pred_image = evaluate(
        model, 
        valid_loader, 
        device=DEVICE,
        save_valid_preds=SAVE_VALID_PREDICTIONS,
        out_dir=OUT_DIR,
        classes=CLASSES,
        colors=COLORS
    )