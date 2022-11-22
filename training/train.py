# ------python import------------------------------------
import os
import urllib
import warnings

import libauc.datasets
import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.utils.data.sampler import SequentialSampler
import logging
import wandb
from CheXpert2.Experiment import Experiment
# ----------- parse arguments----------------------------------
from CheXpert2.Parser import init_parser
from CheXpert2.custom_utils import set_parameter_requires_grad
from CheXpert2.dataloaders.CXRLoader import CXRLoader
# -----local imports---------------------------------------
from CheXpert2.models.CNN import CNN
from CheXpert2.training.training import training
from CheXpert2 import names

from libauc.losses import AUCM_MultiLabel
from libauc.optimizers import PESG
# -----------cuda optimization tricks-------------------------
# DANGER ZONE !!!!!
# torch.autograd.set_detect_anomaly(True)
# torch.autograd.profiler.profile(False)
# torch.autograd.profiler.emit_nvtx(False)
# torch.backends.cudnn.benchmark = True

try:
    os.environ["img_dir"] = os.environ["img_dir"]
except:
    os.environ["img_dir"] = ""
def initialize_config(args):
    # -------- proxy config ---------------------------
    #
    # proxy = urllib.request.ProxyHandler(
    #     {
    #         "https": "http://ccsmtl.proxy.mtl.rtss.qc.ca:8080",
    #         "http": "http://ccsmtl.proxy.mtl.rtss.qc.ca:8080",
    #     }
    # )
    # os.environ["HTTPS_PROXY"] = "http://ccsmtl.proxy.mtl.rtss.qc.ca:8080"
    # os.environ["HTTP_PROXY"] = "http://ccsmtl.proxy.mtl.rtss.qc.ca:8080"
    # # construct a new opener using your proxy settings
    # opener = urllib.request.build_opener(proxy)
    # # install the openen on the module-level
    # urllib.request.install_opener(opener)

    # ------------ parsing & Debug -------------------------------------

    # 1) set up debug env variable
    os.environ["DEBUG"] = str(args.debug)
    if args.debug:
        os.environ["WANDB_MODE"] = "offline"
    # 2) load from env.variable the data repository location
    try:
        img_dir = os.environ["img_dir"]
    except:
        img_dir = "data"

    # ---------- Device Selection ----------------------------------------
    if torch.cuda.is_available():
        if dist.is_initialized():
            rank = dist.get_rank()
            device = rank % torch.cuda.device_count()
        else:
            device = args.device

    else:
        device = "cpu"
        warnings.warn("No gpu is available for the computation")

    # ----------- hyperparameters-------------------------------------<

    config = vars(args)
    experiment = Experiment(
        f"{args.model}", names=names, tag=None, config=config, epoch_max=args.pretraining, patience=5)
    torch.set_num_threads(max(config["num_worker"],1))

    # ----------- load classes ----------------------------------------






    # --------- instantiate experiment tracker ------------------------


    if dist.is_initialized():
        dist.barrier()
        torch.cuda.device(device)
    else:
        config = wandb.config


    return config, img_dir,experiment, device

def main() :
    logging.basicConfig(filename='RADIA.log', level=logging.DEBUG)
    parser = init_parser()
    args = parser.parse_args()
    config, img_dir,experiment, device = initialize_config(args)
    num_classes = len(names)

    # -----------model initialisation------------------------------

    model = CNN(config["model"], num_classes, img_size=config["img_size"], freeze_backbone=config["freeze"],
                pretrained=config["pretrained"], channels=config["channels"], drop_rate=config["drop_rate"],
                global_pool=config["global_pool"])

    # send model to gpu


    print("The model has now been successfully loaded into memory")

    #------------pre-training--------------------------------------
    if config["pretraining"] != 0:

        experiment.compile(
            model,
            optimizer="Adam",
            criterion="BCEWithLogitsLoss",
            train_datasets=["ChexPert","PadChest"],
            val_datasets=["ChexPert"],
            config=config,
            device=device
        )
        results = experiment.train()
        model.backbone.reset_classifier(num_classes=num_classes, global_pool=config["global_pool"])
        model = model.to(device)

    # setting up for the training
    experiment = Experiment(
        f"{config['model']}", names=names, tag=config["tag"], config=config, epoch_max=config["epoch"], patience=5
    )


    #------------training--------------------------------------
    experiment.compile(
        model=model,
        optimizer = "AdamW",
        criterion="BCEWithLogitsLoss",
        train_datasets=["CIUSSS"],
        val_datasets = ["CIUSSS"],
        config=config,
        device=device
    )

    #Libauc : not working for now
    # config.update({"lr": 0.1}, allow_val_change=True)
    # loss = AUCM_MultiLabel(device=device, num_classes=num_classes,
    #                        imratio=(np.array(experiment.training_loader.dataset.count) / len(
    #                            experiment.training_loader.dataset)).tolist())
    # criterion = lambda outputs, preds: loss(torch.sigmoid(outputs), preds)
    #experiment.train(optimizer=PESG(model, loss_fn=loss, device=device,lr=config["lr"],margin=1,epoch_decay=(2e-3),weight_decay=(1e-5)) , criterion=criterion)
    results = experiment.train()
    experiment.end(results)




if __name__ == "__main__":
  main()
