import argparse
import json
import os
import sys
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from sklearn.metrics import det_curve
from malafide import Malafide
from data_utils_AASIST_base_train import (Dataset_train,Dataset_dev,Dataset_eval, genSpoof_list)
from utils import seed_worker, set_seed
import torch.distributed as dist
import torch.multiprocessing as mp


def main_worker(gpu, ngpus_per_node, args: argparse.Namespace):
    """
    Main function.
    Trains, validates, and evaluates the ASVspoof detection model.
    """
    args.gpu = gpu
    cuda = torch.cuda.is_available()
    os.environ['MASTER_ADDR']='localhost'
    os.environ['MASTER_PORT']="1298"

    dist.init_process_group(backend='nccl', world_size=ngpus_per_node, rank=args.gpu)
    torch.cuda.set_device(args.gpu)
        
    
    
    # load experiment configurations
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]
    track = config["track"]
    config['attack'] = args.attack  # putting the attack type in the config ball so that i can pass it to the
    # dataloader function without additional clutter
    assert track in ["LA", "PA", "DF"], "Invalid track given"
    if "eval_all_best" not in config:
        config["eval_all_best"] = "True"
    if "freq_aug" not in config:
        config["freq_aug"] = "False"

    # make experiment reproducible
    set_seed(args.seed, config)

    # define database related paths
    output_dir = Path(args.output_dir)

    # define model related paths
    model_tag = "{}_{}_ep{}_bs{}".format(
        track,
        os.path.splitext(os.path.basename(args.config))[0],
        config["num_epochs"], config["batch_size"])
    if args.comment:
        model_tag = model_tag + "_{}".format(args.comment)
    model_tag = output_dir / model_tag
    model_save_path = model_tag / "weights"
    eval_score_path = model_tag / config["eval_output"]
    writer = SummaryWriter(model_tag)
    os.makedirs(model_save_path, exist_ok=True)
    copy(args.config, model_tag / "config.conf")

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))
    if device == "cpu":
        raise ValueError("GPU not detected!")

    # define model architecture
    model = get_model(model_config, device)
    model.cuda(args.gpu)

    if args.fine_tuned:
        model.load_state_dict(torch.load(config["model_path"], map_location=device))
        print("Model loaded : {}".format(config["model_path"]))

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    # define dataloaders
    trn_loader, dev_loader,eval_loader= get_loader(args.seed, config)

    # get optimizer and scheduler
    malafilter = Malafide(args.adv_filter_size,device)
    malafilter = malafilter.cuda(args.gpu)
    print(f'Instantiated Malafide filter (malafilter) of size {malafilter.get_filter_size()}')

    # set Adam optimizer
    optimizer = torch.optim.Adam(
        malafilter.parameters(),
        lr=optim_config['base_lr'],
        weight_decay=optim_config['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs']*len(trn_loader) - 10,
        eta_min=optim_config['base_lr']/10
    )
   
    if args.eval:
        # load best filter model
        Fastpitch_state = torch.load('/medias/speech/projects/tak/Malafide_experiments_CM/AASIST_SSL/AASIST_SSL_12_5_Fastpitch_malafide_v2_trained_filters_513_thr_0.9_lr_0.1_dubug/LA_AASIST_mls_malafide_v2_th_0.9_lr_0.1_ep1_bs32/weights/epoch_0.pth', map_location=device)
        malafilter.load_state_dict(Fastpitch_state)
        
        
        model.load_state_dict(torch.load(config["model_path"], map_location=device))
        print("Model loaded : {}".format(config["model_path"]))
        print("Start evaluation...")
        produce_evaluation_file(eval_loader, model,malafilter, device,eval_score_path)
        sys.exit(0)

    # start with some baseline readings...
    baseline_eer, baseline_attack_rate = evaluate_eer(dev_loader, model, None, device)
    writer.add_scalar('val_eer', baseline_eer, global_step=0)
    writer.add_scalar('val_attack_rate', baseline_attack_rate, global_step=0)
    print(f'BASELINE READINGS')
    print(f'\tVal EER: {baseline_eer:.3}\tVal attack rate: {baseline_attack_rate:.3}')
    
    best_train_attack_rate = 0

    # Training
    num_epochs = config["num_epochs"]
    for epoch in range(num_epochs):
        print(f'STARTING EPOCH [{epoch+1}/{num_epochs}]')
        running_loss, train_attack_rate = train_epoch(trn_loader, model, malafilter, optimizer, scheduler, device)
        writer.add_scalar('train_loss', running_loss, global_step=epoch+1)
        writer.add_scalar('train_attack_rate', train_attack_rate, global_step=epoch+1)

        # Validation
        eer, val_attack_rate = evaluate_eer(dev_loader, model, malafilter, device)
        writer.add_scalar('val_eer', eer, global_step=epoch+1)
        writer.add_scalar('val_attack_rate', val_attack_rate, global_step=epoch+1)
        print(f'--- RESULTS FOR {args.attack} {args.adv_filter_size} (epoch {epoch+1})')
        print(f'\tTrain loss: {running_loss:.3}\tTrain attack rate: {train_attack_rate:.3}')
        print(f'\tVal EER: {eer:.3}\tVal attack rate: {val_attack_rate:.3}')

        torch.save(malafilter.state_dict(), os.path.join(
            model_save_path, 'epoch_{}.pth'.format(epoch)))

        if train_attack_rate > best_train_attack_rate:
            best_train_attack_rate = train_attack_rate
            print(f'Found new best train attack rate {best_train_attack_rate} at epoch {epoch+1}')
            
            torch.save(
                malafilter,
                os.path.join(model_save_path, 'best_filter.pth')
                )
    

   


def get_model(model_config: Dict, device: torch.device):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    # model = _model(model_config).to(device)
    model = _model("This parameter isn't even used", device)
    model = (model).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))
    return model


def get_loader(
        seed: int,
        config: dict) -> Tuple[torch.utils.data.DataLoader]:
    """Make PyTorch DataLoaders for train / developement / evaluation"""

    database_root = config['dataset_root']

    trn_list_path = config['train_protocol'].format(config['attack'])
    dev_trial_path = config['dev_protocol'].format(config['attack'])
    eval_trial_path = config['eval_protocol'].format(config['attack'])

    d_label_trn, file_train = genSpoof_list(dir_meta=trn_list_path,
                                            is_train=True,
                                            is_eval=False)
    print("no. training files:", len(file_train))

    train_set = Dataset_train(list_IDs=file_train,
                                           labels=d_label_trn,
                                           base_dir=database_root)
    gen = torch.Generator()
    gen.manual_seed(seed)
    trn_loader = DataLoader(train_set,
                            batch_size=config["batch_size"],
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True,
                            worker_init_fn=seed_worker,
                            generator=gen)

    d_label_dev, file_dev = genSpoof_list(
        dir_meta=dev_trial_path,
        is_train=False,
        is_eval=False
    )
    print("no. validation files:", len(file_dev))

    dev_set = Dataset_dev(list_IDs=file_dev,labels=d_label_dev,
                                            base_dir=database_root)
    dev_loader = DataLoader(dev_set,
                            batch_size=config["batch_size"],
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True)

    
    d_label_eval,file_eval,eval_utt,eval_label  = genSpoof_list(dir_meta=eval_trial_path,
                              is_train=False,
                              is_eval=True)   
                                  
    print("no. eval files:", len(file_eval))    
                  
    eval_set = Dataset_eval(list_IDs=file_eval,labels=d_label_eval,UTT_ID=eval_utt,base_dir=database_root,att_list=eval_label)
    
    eval_loader = DataLoader(eval_set,
                             batch_size=config["batch_size"],
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True) 
    return trn_loader, dev_loader, eval_loader   


def get_eer(y_true, y_score):
    """
    A simple function to compute eer.
    More sophisticated metrics are available in this framework.
    This is a quick and easy implementation to use during validation.
    """
    fpr, fnr, thresholds = det_curve(y_true, y_score)
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
    return eer



def produce_evaluation_file(
    data_loader: DataLoader,
    model,naughty_filter,
    device: torch.device,
    save_path: str) -> None:
    """
    Perform evaluation and save the score to a file
    """
    warnings.warn('panariel: I didn\'t write this, and I certainly didn\'t test this. Use this function at your own '
                  'risk. Who knows, you might even set the server room on fire.')
    model.eval()
    key_list = []
    fname_list = []
    score_list = []
    labels = []
    for batch_x,batch_y, utt_id,att_id in data_loader:
        batch_x = batch_x.to(device)
        
        with torch.no_grad():
            attack_id = att_id[0]

            if attack_id == 'bonafide':
                new_x = batch_x
            else:
                new_x = naughty_filter(batch_x,is_test=True)
                
            _, batch_out = model(new_x)
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())
        key_list.extend(
          ['bonafide' if key == 1 else 'spoof' for key in list(batch_y)])
        labels.extend(batch_y.tolist())
    eer = get_eer(labels, score_list)
    print(eer)
    with open(save_path, "w") as fh:
        for fn, key,sco in zip(fname_list,key_list, score_list):
            
            fh.write("{} {} {}\n".format(fn, key, sco))
    print("Scores saved to {}".format(save_path))


def evaluate_eer(
    dev_loader: DataLoader,
    model,  naughty_filter, 
    device: torch.device, threshold=0.9, desc='Running evaluation'):
    
    num_total = 0
    successful_attacks = 0

    scores = []
    labels = []

    sf = nn.Softmax(dim=1)

    model.eval()
    if naughty_filter is not None:
        naughty_filter.eval()
    
    num_batches = len(dev_loader)
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(dev_loader):
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)

            batch_x_bonafide = batch_x[batch_y == 1]
            batch_x_spoof = batch_x[batch_y == 0]

            # First evaluate the samples in buona fede
            if batch_x_bonafide.nelement() > 0:
                bonafide_scores = sf(model(batch_x_bonafide)[1])
                bonafide_scores = bonafide_scores[:, 1].tolist()
                scores.extend(bonafide_scores)
                print(f'[{i+1}/{num_batches}] BF scores of BF: {bonafide_scores}')
            
            # now the spooky scary spoofed files
            if batch_x_spoof.nelement() > 0:
                # if a filter is given, attack
                if naughty_filter is not None:
                    batch_x_spoof = naughty_filter(batch_x_spoof.unsqueeze(1)).squeeze(1)  # there used to be is_test=True here
                spoof_scores = sf(model(batch_x_spoof)[1])
                spoof_scores = spoof_scores[:, 1].tolist()
                scores.extend(spoof_scores)

                # count successful attacks
                successful_attacks += (np.array(spoof_scores) >= threshold).sum()
                print(f'[{i+1}/{num_batches}] BF scores of spoof: {spoof_scores}')

            # keep the labels
            labels.extend(batch_y.tolist())

    # we now have scores and labels. Guess what we do next
    eer = get_eer(labels, scores)
    # also compute attack success rate, for science
    attack_success_rate = successful_attacks/num_total

    return eer, attack_success_rate

def train_epoch(
    trn_loader: DataLoader,
    model,
    naughty_filter,
    optim: Union[torch.optim.SGD, torch.optim.Adam],
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
    device: torch.device,threshold=0.9):
    """Train the model for one epoch"""
    running_loss = 0
    num_total = 0.0
    ii = 0
    model.eval()
    naughty_filter.train()

    criterion = nn.CrossEntropyLoss()
    sf = nn.Softmax(dim=1)

    successfully_attacked = 0
    num_batches = len(trn_loader)
    
    for i, (batch_x, batch_y) in enumerate(trn_loader):
        optim.zero_grad()
        current_lr = optim.param_groups[0]['lr']
        batch_size = batch_x.size(0)
        batch_x = batch_x.unsqueeze(1)
        num_total += batch_size
        ii += 1
        batch_x = batch_x.to(device)
        fake_batch_y = torch.ones_like(batch_y, device=device)  # make the label all ones, we optimize for bonafide
        
        batch_x = naughty_filter(batch_x)  # is_test = False used to be here
        batch_x = batch_x.squeeze()  # back to (B, L)

        _, batch_out = model(batch_x)
        batch_loss = criterion(batch_out, fake_batch_y)
        running_loss += batch_loss.item() * batch_size

        batch_loss.backward()
        optim.step()
        scheduler.step()

        naughty_filter.project() # reset central spike in filter

        # update the attack success rate
        bf_scores = sf(batch_out)[:, 1]
        successfully_attacked += (bf_scores >= threshold).sum().item()
        
        print(f'[{i+1}/{num_batches}] (lr: {current_lr:.5}) BF scores: {bf_scores.data}')

    running_loss /= num_total
    attack_success_rate = successfully_attacked/num_total
    return running_loss, attack_success_rate


if __name__ == "__main__":
    warnings.warn(
        "This script uses the full-power amazing SSL-AASIST CM trained with RawBoost. It doesn't really work that well on Macron apparently, but still.")
    parser = argparse.ArgumentParser(description="ASVspoof detection system")
    parser.add_argument("--config",
                        dest="config",
                        type=str,
                        help="configuration file",
                        required=True)
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        help="output directory for results",
        default="./results",
    )
    parser.add_argument('--attack', type=str, default='AX',
                        help='Which kind of attack to optimize onto. Defaults to AX, which is a protocol that contains all attacks.')

   
    parser.add_argument('--adv_filter_size', type=int, default=513) # should be odd
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="random seed (default: 1234)")
    parser.add_argument('--model_path', type=str,
                        default=None, help='Model checkpoint')
    parser.add_argument(
        "--eval",
        action="store_true",
        help="when this flag is given, evaluates given model and exit")
    parser.add_argument(
        "--fine_tuned",
        action="store_true",
        help="when this flag is given,pre_trained CM weights used to intialize the model")
    parser.add_argument("--comment",
                        type=str,
                        default=None,
                        help="comment to describe the saved model")
    parser.add_argument("--eval_model_weights",
                        type=str,
                        default=None,
                        help="directory to the model weight file (can be also given in the config file)")

    n_gpus = torch.cuda.device_count()
    print(f'Python Version: {sys.version}')
    print(f'PyTorch Version: {torch.__version__}')
    print(f'Number of GPUs: {n_gpus}')
    
    mp.spawn(main_worker, nprocs=n_gpus, args=(n_gpus, parser.parse_args()))
    
