import torch

import os
from detectron2.engine import DefaultTrainer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.structures import BoxMode
from detectron2.config import get_cfg as _get_cfg

from detectron2 import model_zoo

from detectron2.data.datasets import register_coco_instances


from loss import ValidationLoss

import cv2

def get_cfg(output_dir,learning_rate,batch_size,iterations, checkpoint_period, model, device, num_classes):
    """
    Create a Detectron2 configuration object and set its attributes.

    Args:
        output_dir (str): The Path to the output directory where the trained model and logs will be saved.

        learning_rate (float): The learning rate for the optimizer.

        batch_size (int): The batch size used during Training.

        iterations (int): The maximum number of training iterations.

        checkpoint_period (int): The number of iterations between consecutive checkpoints.

        model (str): The name of the model to use, which should be one of the models available in Detectron2's model zoo.

        device (str): The device used for training, which should be cpu or gpu or mps.

        num_classes (int): The number of classes in the dataset.
    
    Returns:
        The Detectron2 configuration object.

    """

    cfg = _get_cfg()

    # Merge the model's default configuration file with the default Detectron2 configuration file.
    cfg.merge_from_file(model_zoo.get_config_file(model))

    # Set the Training and Validation Datasets and exclude the Test Dataset.
    
    # If you look at register_datasets function, we have registered the datasets as train and val.
    cfg.DATASETS.TRAIN = ('train',)
    cfg.DATASETS.VAL = ('val',)
    cfg.DATASETS.TEST = ()

    # Enables cropping of images during Data Augmentation.
    cfg.INPUT.CROP.ENABLED = True


    # Set the Device to use for training.
    if device in ['cpu']:
        cfg.MODEL.DEVICE = device

    cfg.DATALOADER.NUM_WORKERS = 2

    # Set the model weights to the ones pretrained on the COCO Dataset.
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)

    # Set the batch size used by the solver.
    cfg.SOLVER.IMS_PER_BATCH = batch_size

    # Set the checkpoint period
    cfg.SOLVER.CHECKPOINT_PERIOD = checkpoint_period

    # Set the base learning rate.
    cfg.SOLVER.BASE_LR = learning_rate

    # Set the Maximum Training Iterations.
    cfg.SOLVER.MAX_ITER = iterations

    # Set the learning rate scheduler steps to an empty list, which means the learning rate will not be decayed.
    cfg.SOLVER.STEPS = []

    # Set the batch size used by the ROI heads during training.
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

    # Set the number of classes.
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

    cfg.MODEL.RETINANET.NUM_CLASSES = num_classes

    # Set the output directory.
    cfg.OUTPUT_DIR = output_dir

    return cfg


def register_datasets(root_dir, class_list_file):
    """
    Registers the train and validation datasets and returns the number of classes.

    Args:
        root_dir (str): Path to the root directory of the dataset.

        class_list_file (str): Path to the file containing the list of class names.

    Returns:
        int : The number of classes in the dataset.
    """

    # Read the list of class names from the class list file.
    with open(class_list_file,'r') as reader:
        classes_ = [l[:-1] for l in reader.readlines()]
    
    # Register the train and validation datasets.
    for d in ['train','val']:
        register_coco_instances(d, {}, os.path.join(root_dir,d,'anns',d+'_annos.json'), os.path.join(root_dir,d,'imgs'))

        # Set the metadata for the dataset.
        MetadataCatalog.get(d).set(thing_classes=classes_)

    return len(classes_)

def train(output_dir, data_dir, class_list_file, learning_rate, batch_size, iterations, checkpoint_period, device, model):
    """
    Train a Detectron2 model on a custom dataset.

    Args:
        output_dir (str): Path to the directory to save the trained model and output files.

        data_dir (str): Path to the directory containing the dataset.

        class_list_file (str): path to the file containing the list of class names in the dataset. (for our case num_classes=1, alpaca).

        learning_rate (float): learning rate of the optimizer.

        batch_size (int): Batch Size for training.

        iterations (int): Maximum number of training iterations.

        checkpoint_period (int): Number of iterations after which you want to save the checkpoint of the model.

        device (str): Device to use for training. (eg: 'cpu', 'mps', 'cuda')

        model (str): Name of the model configuration to use. Must be a key in the Detectron2 model zoo.

        Returns:
            None
    """

    # In Detectron2 we need to register the datasets before starting the training process

    # Register the dataset and get the num_classes
    num_classes = register_datasets(data_dir,class_list_file)

    # Get the Configuration for the model
    cfg = get_cfg(output_dir, learning_rate, batch_size, iterations, checkpoint_period, model, device, num_classes)

    # Create the Output Directory
    os.makedirs(cfg.OUTPUT_DIR,exist_ok=True)

    # Create the Trainer Object
    trainer = DefaultTrainer(cfg)

    # Create a Custom Validation Loss Object
    val_loss = ValidationLoss(cfg)


    trainer.register_hooks([val_loss])


    trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]

    trainer.resume_or_load(resume=False)


    trainer.train()

