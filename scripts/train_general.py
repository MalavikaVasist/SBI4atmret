#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import wandb
import pandas as pd 
import importlib

from dawgz import job, schedule
from itertools import islice
from pathlib import Path
from torch import Tensor
from tqdm import tqdm

from petitRADTRANS.retrieval.util import *
import petitRADTRANS as prt
from petitRADTRANS import nat_cst as nc, Radtrans
from petitRADTRANS.retrieval.data import Data


from lampe.data import H5Dataset
from lampe.inference import NPE, NPELoss
from lampe.nn import ResMLP
from lampe.nn.flows import NAF
from lampe.utils import GDStep
from zuko.distributions import BoxUniform
from lampe.plots import nice_rc, corner, mark_point

scratch = os.environ.get('SCRATCH') 
home = os.environ.get('HOME')

# from sbi.added_scripts.AverageEstimator import avgestimator
from added_scripts.corner_modified import *
from added_scripts.pt_plotting import *
from code2explore.plotting_HST_consistency import HST_consistency
from code2explore.plotting_Gemini_consistency import Gemini_consistency
from code2explore.plotting_MIRI_consistency import MIRI_consistency


import json

from utils.Loss import BNPELoss
from utils.Noise_models import noisybfactor
from utils.Plots import plots, ratio, computing_gravity, computing_mass
from utils.f import plotting_contribution
from utils.ees_general import Simulator
from observations.load_obs import load_observations

import sys
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

# Helper to load a class from a module
def load_callable(module_path, name):
    module = importlib.import_module(module_path)
    return getattr(module, name) 


with open("script_general.json") as f:
    config_script = json.load(f)

scratch = os.environ.get(config_script['scratch_env'])
home = os.environ.get(config_script['home_env'])

##Loading observations 
observation = load_observations(config_script['source'], \
                                 config_script['instruments'], \
                                    config_script['tag'], \
                                        config_script['D'], \
                                            config_script['simulator']['D_pl'], \
                                               config_script['simulator']['scale'])

# obs_wlen_inst = np.append(observation, obs_wlen_gemini)
# index_argsort = np.argsort(obs_wlen_inst)

# obs_inst_app = np.append(obs_hst, obs_gemini)
# obs_inst = obs_inst_app[index_argsort]
# x_star = np.append(obs_inst, obs_miri)


##Loading simulator
simulator_cfg = config_script["simulator"]

simulator_cfg_callable = simulator_cfg.pop("callable")
emission_model = load_callable(
                                simulator_cfg_callable["emission_model"]["module"],
                                simulator_cfg_callable["emission_model"]["function"])

pt_profile = load_callable(
                            simulator_cfg_callable["pt_profile"]["module"],
                            simulator_cfg_callable["pt_profile"]["function"])
wav = config_script['wav']
simulator = {}
sim_type = simulator_cfg.pop('type')
for atm_type in sim_type: 
    simulator[atm_type] = {}
    for i, instrument in enumerate(config_script['instruments']):
        simulator[type][instrument] = Simulator(
                                            emission_model_diseq=emission_model,
                                            PTprofile=pt_profile,
                                            a=wav[i][0],
                                            b=wav[i][1],
                                            **simulator_cfg
                                        )

savepath = Path(scratch) / config_script['sim_paths']['savepath']
savepath.mkdir(parents=True, exist_ok=True)


@job(array=1, cpus=2, gpus=1, ram='16GB', time='10-00:00:00')
def train(i: int):

    i = 9

    name = str(model_configs["name"][i])
    model_configs = config_script["model_configs"]
    batch_size = model_configs["batch_size"][i]
    hidden_features = model_configs["hidden_features"][i]
    hidden_features_no = model_configs["hidden_features_no"][i]
    emb_gem_output = model_configs["embedding"]["gemini_output"][i]
    init_lr = model_configs["init_lr"][i]
    min_lr = model_configs["min_lr"][i]
    no_of_params = model_configs["no_of_params"][i]
    patience = model_configs["patience"][i]
    transforms = model_configs["transforms"][i]
    weight_decay = model_configs["weight_decay"][i]
    embmiri = model_configs["embedding"]["miri"][i]
    embgem = model_configs["embedding"]["gemini"][i]
    signal = model_configs["signal"][i]
    loss_name = model_configs["loss"][i]
    epoch_fin = model_configs["epoch_fin"][i]
    ep = model_configs["epochs"][i]
    model_params = model_configs['model_params']
    model = model_configs["model"][i]
    D = config_script["simulator"]["D"]


    training_configs = config_script["training"]
    gradient_steps_train = training_configs["gradient_steps_train"]
    gradient_steps_valid = training_configs["gradient_steps_valid"]
    scheduler_type = training_configs["scheduler"]["type"]
    lr_factor = training_configs["scheduler"]["factor"]
    threshold = training_configs["scheduler"]["threshold"]
    stop_criterion = training_configs["stop_criterion"]

    spectral_info_configs = config_script["spectral_info"]
    spectral_range = spectral_info_configs["spectral_range"]
    wavelength_range = spectral_info_configs["wavelength_range"]

    LABELS, LOWER, UPPER = zip(*config_script["PARAMETERS"])

    datasets = {}
    for type in config_script['simulator']["type"]:  # "cloudfree", "cloudy"
        datasets[type] = {}
        for instrument in config_script['instruments']:  # "miri", "gemini", "hst"
            try:
                path = config_script["sim_paths"][type][instrument]
            except KeyError:
                raise ValueError(f"Missing path for {instrument} and type {atm_type}")

            datasets[type][instrument] = {
                'train': H5Dataset(Path(scratch) / path / 'train.h5', batch_size=batch_size),
                'valid': H5Dataset(Path(scratch) / path / 'valid.h5', batch_size=batch_size),
                'test':  H5Dataset(Path(scratch) / path / 'test.h5', batch_size=16),
            }
   
    config_dict = {
            'embedding_miri' : '1298 to 64 ' + str(embmiri) , 
            'embedding_hst_gem' : '434 to ' + str(emb_gem_output) + ' ' + str(embgem) , 
            'flow': 'NAF',
            'transforms': transforms, 
            'signal' : signal, 
            'hidden_features': hidden_features, # hidden layers of the autoregression network
            'hidden_features_no' : hidden_features_no,
            'activation': 'ELU',
            'optimizer': 'AdamW',
            'init_lr': init_lr,
            'weight_decay': weight_decay,
            'scheduler': scheduler_type, 
            'min_lr': min_lr,
            'lr_factor': lr_factor,
            'patience': patience,
            'epochs': ep,
            'stop_criterion': stop_criterion, 
            'batch_size': batch_size,
            'gradient_steps_train': gradient_steps_train, 
            'gradient_steps_valid': gradient_steps_valid, 
            'noisy' : 'with_Cushing_b_factor_rightway',
            'obs_spectrum': 'MIRI+HST+Gemini', 
            'wavelength_range' : wavelength_range,       # '' ,  4.5-19 um , 4.5-19 um
            'length_spectrum' : str(emb_gem_output) + ' + 64', #'8+128',  305+1298         # 1296 , 1298 , 1296
            'spectral_range' : spectral_range,   #3:1299 , 87:1385 , 89:1385
            'model_params' : model_params, #_filterM50', 
            'no_of_params' : no_of_params, 
            'source_distance' : D + ' pc',
            'model' : model,
            'loss' : loss_name,
            } 

    wandb_info = config_script["wandb"]
    project = wandb_info["project"]

    # Run
    run = wandb.init(project= project, config = config_dict, name = model+ '_'+ name)   
    
    # Training
    estimator_class = load_callable(config_script["estimator"]["module"], config_script["estimator"]["class"])
    estimator = estimator_class(hf_miri= embmiri, hf_inst= embgem, emb_miri_output = 64, emb_inst_output = emb_gem_output ,
                                        instrument = config_script["estimator"]["instrument"],
                                        hidden_features = hidden_features,
                                        no_of_params = no_of_params,
                                        transforms = transforms, 
                                        signal = signal, \
                                        LOWER = LOWER, UPPER = UPPER).cuda()


    #retraining
    # states = torch.load(savepath / model_arch / ('states_' + str(epoch_fin) + '.pth'), map_location='cpu')
    # estimator.load_state_dict(states['estimator'])
    # estimator.cuda()
    # runpath = savepath / model_arch


    prior = BoxUniform(torch.tensor(LOWER).cuda(), torch.tensor(UPPER).cuda())
    if loss_name == 'NPELoss':
        loss = NPELoss(estimator) 
    elif loss_name == 'BNPELoss':
        loss = BNPELoss(estimator, prior)
    
    optimizer = optim.AdamW(estimator.parameters(), lr= init_lr, weight_decay= weight_decay)
    step = GDStep(optimizer, clip=1.0)

    scheduler = sched.ReduceLROnPlateau(
        optimizer,
        factor=lr_factor,
        min_lr= min_lr,
        patience= patience,
        threshold=threshold,
        threshold_mode='abs',
    )

    pipe = load_callable(config_script["pipe"]["module"], config_script["pipe"]["function"], loss)

    for epoch in tqdm(range(epoch_fin, ep), unit='epoch'):
        estimator.train()
        start = time.time()

        trainsets =[]
        for type in config_script['simulator']["type"]:  # "cloudfree", "cloudy"
            for instrument in config_script['instruments']: 
                trainsets.append(datasets[type][instrument]['train'])

        loss_train_list = []
        
        for data_tuple in islice(zip(*trainsets), 1700):
            sim_models = {}
            idx = 0
            for type in config_script['simulator']["type"]:
                sim_models[type] = {}
                for instrument in config_script['instruments']:
                    sim_models[type][instrument] = data_tuple[idx]
                    idx += 1
            output = pipe(sim_models, simulator[type][instrument])
            loss_train = step(output)
            loss_train_list.append(loss_train)
        losses_train = torch.stack(loss_train_list).cpu().numpy()

        end = time.time()
        estimator.eval()

        validsets = []
        for type in config_script['simulator']["type"]:  # "cloudfree", "cloudy"
            for instrument in config_script['instruments']: 
                validsets.append(datasets[type][instrument]['valid'])

        loss_valid_list = []
        with torch.no_grad():
            for data_tuple in islice(zip(*validsets), 170):
                sim_models = {}
                idx = 0
                for type in config_script['simulator']["type"]:
                    sim_models[type] = {}
                    for instrument in config_script['instruments']:
                        sim_models[type][instrument] = data_tuple[idx]
                        idx += 1                
                output = pipe(sim_models, simulator[type][instrument])
                loss_valid = step(output)
                loss_valid_list.append(loss_valid)
        losses_val = torch.stack(loss_valid_list).cpu().numpy()

        run.log({
            'lr': optimizer.param_groups[0]['lr'],
            'loss': np.nanmean(losses_train),
            'loss_val': np.nanmean(losses_val),
            'nans': np.isnan(losses_train).mean(),  #percentage of NaNs
            'nans_val': np.isnan(losses_val).mean(),
            'speed': len(losses_train) / (end - start),
            'trainigset_len' :  len(losses_train),
            'validationset_len' : len(losses_val),
        })

        scheduler.step(np.nanmean(losses_val))

        runpath = savepath / run.name
        runpath.mkdir(parents=True, exist_ok=True)
        
        # if epoch > 0:
        if epoch > 100:
            if epoch % 50 == 0 : 
                torch.save({
                'estimator': estimator.state_dict(),
                'optimizer': optimizer.state_dict(),
            },  runpath / f'states_{epoch}.pth')

        if stopping == 'early': 
            if optimizer.param_groups[0]['lr'] <= scheduler.min_lrs[0]:
                break

            if epoch % 100 == 0 : 
                plot = plots(runpath, int(epoch/50) * 50)
                cov_fig = plot.coverage()
                corner_fig = plot.cornerplot()
                fig_pt = plot.ptprofile()
                res_fig_miri = plot.consistencyplot_MIRI()
                res_fig_gemini = plot.consistencyplot_Gemini() 
                res_fig_hst = plot.consistencyplot_HST()
                cornerWratio_fig = plot.cornerWratio()

    # observation = load_class(config_script["pipe"]["module"], config_script["pipe"]["observation"])
    # index_argsort, x_star = observation()

    plot = plots(runpath, int(epoch/50) * 50, estimator, observation)

    testsets =[]
    for type in config_script['simulator']["type"]:  # "cloudfree", "cloudy"
        for instrument in config_script['instruments']: 
            testsets.append(datasets[type][instrument]['test'])
    cov_fig = plot.coverage(testsets, pipe, simulator)

    # corner_fig = plot.cornerplot(LABELS, LOWER, UPPER)

    appending_params_dict = {r'$^{14}N/^{15}N$' : {"limits": [0, 1000], "method" : ratio}, 
                                    r'$log g$' : {"limits": [2, 6], "method" : computing_gravity},
                                    r'$Mass$' : {"limits": [1, 50], "method" : computing_mass}}
    cornerWratio_notfull(LOWER, UPPER, LABELS, theta=None, columns = [0, 1, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25], appending_params_dict = appending_params_dict, \
                        legends = ['NPE'], colors = ['steelblue'], savepath= None, labelsize = 18, titlesize = 20, fontsize= 16, legend_fontsize = 20, xtick_labelsize = 28 , ytick_labelsize = 28,  \
                        theta_star= None, loc= 'center', bbox_to_anchor= (0.4,0.9), labl= True, alpha = [0, 0.9])
   

    fig_pt = plot.ptprofile()
    res_fig_miri = plot.consistencyplot_MIRI()
    res_fig_gemini = plot.consistencyplot_Gemini() 
    res_fig_hst = plot.consistencyplot_HST()
    cornerWratio_fig = plot.cornerWratio()

    run.log({
        'coverage': wandb.Image(cov_fig),
        'corner': wandb.Image(corner_fig),
        'pt_profile': wandb.Image(fig_pt),
        'res_fig_miri': wandb.Image(res_fig_miri),
        'res_fig_gemini': wandb.Image(res_fig_gemini),
        'res_fig_hst': wandb.Image(res_fig_hst),
        'cornerWratio_fig' : wandb.Image(cornerWratio_fig)
    })
    run.finish()
    
if __name__ == '__main__':

    schedule(
        train, 
        name='Training',
        backend='slurm',
        env=[
            'source ~/.bashrc',
            'conda activate WISEJ1828',
            'export WANDB_SILENT=true',
        ]
    )



