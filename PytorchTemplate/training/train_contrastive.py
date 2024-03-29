# ------python import------------------------------------
import os

import warnings

import timm
import torch
import torch.distributed as dist
import logging
import wandb
from torch.optim.swa_utils import AveragedModel

# -----local imports---------------------------------------
from PytorchTemplate.Parser import init_parser
from PytorchTemplate.Experiment import Experiment as Experiment
from PytorchTemplate import names
from PytorchTemplate.datasets.cifar10 import cifar10


# -----------cuda optimization tricks-------------------------
# DANGER ZONE !!!!!
# torch.autograd.set_detect_anomaly(True)
# torch.autograd.profiler.profile(False)
# torch.autograd.profiler.emit_nvtx(False)
torch.backends.cudnn.benchmark = True


def main() :



    parser = init_parser()
    args = parser.parse_args()
    config = vars(args)
    # -------- proxy config ----------------------------------------
    #
    # proxy = urllib.request.ProxyHandler(
    #     {
    #         "https": "",
    #         "http": "",
    #     }
    # )
    # os.environ["HTTPS_PROXY"] = ""
    # os.environ["HTTP_PROXY"] = ""
    # # construct a new opener using your proxy settings
    # opener = urllib.request.build_opener(proxy)
    # # install the openen on the module-level
    # urllib.request.install_opener(opener)






    #----------- load the datasets--------------------------------
    batch_size = config["batch_size"]
    valid_size = 0.1
    shuffle = True
    num_workers = config["num_worker"]
    pin_memory = True
    mean = (0.485, 0.456, 0.406) # (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
    std = (0.229, 0.224, 0.225)
    root = "data"
    train_loader,val_loader = cifar10(batch_size, valid_size, shuffle, num_workers, pin_memory, mean, std, root,config["debug"],prob=config["augment_prob"],label_smoothing=config["label_smoothing"])



    # import torch.nn.utils.prune as prune
    #
    # parameters_to_prune = [
    #     (module, "weight") for module in
    #     filter(lambda m: type(m) in [torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.Linear], model.modules())
    # ]
    # prune.global_unstructured(
    #     parameters_to_prune,
    #     prune.L1Unstructured,
    #     amount=0.6,
    # )
    # for module,name in parameters_to_prune :
    #     torch.nn.utils.prune.remove(module, name)







    #------------ Training --------------------------------------



    # setting up for the  experiment

    config["swa_start"] = 1

    experiment = Experiment(names, config)

    experiment.compile(
        model_name=config["model"],
        optimizer = "AdamW",
        criterion="CrossEntropyLoss",
        train_loader=train_loader,
        val_loader=val_loader,
        final_activation="softmax",

    )
    setattr(experiment.model, "name", experiment.model.name + "_contrastive")
    from pytorch_metric_learning import losses
    class SupervisedContrastiveLoss(torch.nn.Module):
        def __init__(self, temperature=0.1):
            super(SupervisedContrastiveLoss, self).__init__()
            self.temperature = temperature

        def forward(self, feature_vectors, labels):
            labels = torch.argmax(labels, dim=1)

            # Normalize feature vectors
            feature_vectors_normalized = torch.nn.functional.normalize(feature_vectors, p=2, dim=1)
            # Compute logits
            logits = torch.div(
                torch.matmul(
                    feature_vectors_normalized, torch.transpose(feature_vectors_normalized, 0, 1)
                ),
                self.temperature,
            )
            return losses.NTXentLoss(temperature=0.07)(logits, torch.squeeze(labels))


    criterion = SupervisedContrastiveLoss().to(experiment.device)
    results = experiment.train(criterion=criterion,use_features=True)

    experiment.end()




if __name__ == "__main__":
  main()
