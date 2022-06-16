"""Train
"""

from modules.utils import load_yaml, save_yaml, get_logger
from modules.earlystoppers import EarlyStopper
from modules.recorders import Recorder
from modules.datasets import *
from modules.trainer import Trainer
from modules.optimizers import get_optimizer
from modules.schedulers import PolyLR
from models.utils import get_model, EMA
import torch

from datetime import datetime, timezone, timedelta
import numpy as np
import random
import os
import copy

import wandb
import warnings
warnings.filterwarnings('ignore')

# Root directory
PROJECT_DIR = os.path.dirname(__file__)
GDRIVE_DIR = '/content/drive/MyDrive'

# Load config
config_path = os.path.join(PROJECT_DIR, 'config', 'train_config.yml')
config = load_yaml(config_path)

if 'is_trained' in config['TRAINER']:
    if config['TRAINER']['is_trained'] == True:
        is_trained = config['TRAINER']['is_trained']
        train_serial = config['TRAINER']['train_serial']
        config = load_yaml(os.path.join(PROJECT_DIR, 'results', 'train', train_serial, 'train_config.yml'))
else:
    # Train Serial
    is_trained = False
    kst = timezone(timedelta(hours=9))
    train_serial = datetime.now(tz=kst).strftime("%Y%m%d_%H%M%S")

# Recorder directory
if 'is_colab' in config['TRAINER']:
    if config['TRAINER']['is_colab'] == True:
        RECORDER_DIR = os.path.join(GDRIVE_DIR, 'results', 'train', train_serial)
else:
    RECORDER_DIR = os.path.join(PROJECT_DIR, 'results', 'train', train_serial)
os.makedirs(RECORDER_DIR, exist_ok=True)

# Data directory
DATA_DIR = os.path.join(PROJECT_DIR, 'data', config['DIRECTORY']['dataset'])

# Seed
torch.manual_seed(config['TRAINER']['seed'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(config['TRAINER']['seed'])
random.seed(config['TRAINER']['seed'])

# GPU
os.environ['CUDA_VISIBLE_DEVICES'] = str(config['TRAINER']['gpu'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__': 
    """
    00. Set Logger
    """
    logger = get_logger(name='train', dir_=RECORDER_DIR, stream=False)
    logger.info("Set Logger {}".format(RECORDER_DIR))

    """
    01. Load data
    """
    # Dataset
    data_loader = BuildDataLoader(num_labels=config['MODEL']['num_labels'], dataset_path=config['DIRECTORY']['dataset'],
                                  batch_size=config['DATALOADER']['batch_size'])
    train_l_loader, train_u_loader, valid_l_loader, _ = data_loader.build(supervised=False)
    logger.info("Load data, train (labeled):{} train (unlabeled):{} val:{}".format(len(train_l_loader), len(train_u_loader), len(valid_l_loader)))
        
    if is_trained == True:
        '''
        02.1 load model
        '''
        model = get_model(model_name=config['TRAINER']['model'], num_classes=config['MODEL']['num_labels'],
                        output_dim=config['MODEL']['output_dim']).to(device)
        checkpoint = torch.load(os.path.join(RECORDER_DIR, 'model.pt'))
        model.load_state_dict(checkpoint['model'])
    
    else:
        """
        02. Set model
        """
        # Load model
        model = get_model(model_name=config['TRAINER']['model'],num_classes=config['MODEL']['num_labels'],
                        output_dim=config['MODEL']['output_dim']).to(device)
    ema = EMA(model, 0.99)  # Mean teacher model

    """
    03. Set trainer
    """
    # Optimizer
    optimizer = get_optimizer(optimizer_name=config['TRAINER']['optimizer'])
    optimizer = optimizer(params=model.parameters(),lr=config['TRAINER']['learning_rate'])
    scheduler = PolyLR(optimizer, config['TRAINER']['n_epochs'], power=0.9)

    # Early stoppper
    early_stopper = EarlyStopper(patience=config['TRAINER']['early_stopping_patience'],
                                mode=config['TRAINER']['early_stopping_mode'],
                                logger=logger)

    # Trainer
    trainer = Trainer(model=model,
                      ema=ema,
                      data_loader=data_loader,
                      optimizer=optimizer,
                      device=device,
                      logger=logger,
                      config=config['TRAINER'],
                      interval=config['LOGGER']['logging_interval'])
    
    """
    Logger
    """
    # Recorder
    recorder = Recorder(record_dir=RECORDER_DIR,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        logger=logger)

    # !Wandb
    if config['LOGGER']['wandb'] == True: ## 사용시 본인 wandb 계정 입력
        wandb_project_serial = 'v3p_adamw_classmix'
        wandb_username =  'a22106'
        wandb.init(project=wandb_project_serial, dir=RECORDER_DIR, entity=wandb_username)
        wandb.run.name = train_serial
        wandb.config.update(config)
        wandb.watch(model)

    # Save train config
    save_yaml(os.path.join(RECORDER_DIR, 'train_config.yml'), config)

    """
    04. TRAIN
    """
    # Train
    n_epochs = config['TRAINER']['n_epochs']
    for epoch_index in range(n_epochs):

        # Set Recorder row
        row_dict = dict()
        row_dict['epoch_index'] = epoch_index
        row_dict['train_serial'] = train_serial
        
        """
        Train
        """
        print("Train {}/{}".format(epoch_index, n_epochs))
        logger.info("--Train {}/{}".format(epoch_index, n_epochs))
        trainer.train(train_l_loader=train_l_loader, train_u_loader=train_u_loader)
        
        row_dict['train_loss'] = trainer.loss_mean
        row_dict['train_elapsed_time'] = trainer.elapsed_time 
        
        for metric_str, score in trainer.score_dict.items():
            row_dict["train_{}".format(metric_str)] = score
        trainer.clear_history()
        
        """
        Validation
        """
        print("Val {}/{}".format(epoch_index, n_epochs))
        logger.info("--Val {}/{}".format(epoch_index, n_epochs))
        trainer.valid(valid_l_loader=valid_l_loader)
        
        row_dict['val_loss'] = trainer.loss_mean
        row_dict['val_elapsed_time'] = trainer.elapsed_time 
        
        for metric_str, score in trainer.score_dict.items():
            row_dict["val_{}".format(metric_str)] = score
        trainer.clear_history()

        """
        Record
        """
        recorder.add_row(row_dict)
        recorder.save_plot(config['LOGGER']['plot'])

        #!WANDB
        if config['LOGGER']['wandb'] == True:
            wandb.log(row_dict)

        """
        Early stopper
        """
        early_stopping_target = config['TRAINER']['early_stopping_target']
        early_stopper.check_early_stopping(loss=row_dict[early_stopping_target])

        if early_stopper.patience_counter == 0:
            recorder.save_weight(epoch=epoch_index)
            best_row_dict = copy.deepcopy(row_dict)
        
        if early_stopper.stop == True:
            logger.info("Eearly stopped, coutner {}/{}".format(early_stopper.patience_counter, config['TRAINER']['early_stopping_patience']))
            
            if config['LOGGER']['wandb'] == True:
                wandb.log(best_row_dict)
            break
        
        if epoch_index %5 == 0:
          os.system('zip result.zip /content/baseline/result/train/*')
          os.system('mv result.zip /content/drive/MyDrive/data')
