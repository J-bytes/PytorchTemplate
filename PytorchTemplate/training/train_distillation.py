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
from PytorchTemplate.variation.DistillationExperiment import DistillationExperiment as Experiment
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



    # setting up for the Distillation experiment

    config["swa_start"] = 1

    experiment = Experiment(names, config)
    experiment.compile(
        model_name=config["model"],
        optimizer = "AdamW",
        criterion="MSELoss",
        train_loader=train_loader,
        val_loader=val_loader,
        final_activation="softmax",

    )
    setattr(experiment.model, "name", experiment.model.name + "_distillation")


    teacher_model = timm.create_model("convnext_nano",pretrained=False,num_classes=10)
    name = teacher_model.default_cfg['architecture']
    teacher_model = AveragedModel(teacher_model)
    setattr(teacher_model,"name",name)
    teacher_model.load_state_dict(torch.load(f"models_weights/{name}.pt"))
    teacher_model = teacher_model.to(experiment.device,dtype=torch.float16)


    # import torch.nn.utils.prune as prune
    #
    # parameters_to_prune = [
    #     (module, "weight") for module in
    #     filter(lambda m: type(m) in [torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.Linear], teacher_model.modules())
    # ]
    # prune.global_unstructured(
    #     parameters_to_prune,
    #     prune.L1Unstructured,
    #     amount=0.3,
    # )




    # for module,name in parameters_to_prune :
    #     torch.nn.utils.prune.remove(module, name)

    import copy
    experiment2 = copy.copy(experiment)
    experiment2.model = teacher_model
    val_loss,results = experiment2.validation_loop()
    accuracy = (torch.argmax(results[1],dim=1).cpu().numpy() == torch.argmax(torch.softmax(results[0],dim=1),dim=1).cpu().numpy()).sum()/len(results[0])
    print(val_loss,accuracy)
    results = experiment.train(teacher = teacher_model)

    experiment.end()




if __name__ == "__main__":
  main()
