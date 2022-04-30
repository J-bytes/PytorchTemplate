# ------python import------------------------------------
import argparse
import time
import warnings
import numpy as np
import torch
import tqdm
from custom_utils import  Experiment,preprocessing,metrics
from sklearn.metrics import confusion_matrix
from  sklearn.metrics import top_k_accuracy_score
from training.training import validation_loop
from training.dataloaders.cct_dataloader import CustomImageDataset
def init_argparse() :
    parser = argparse.ArgumentParser(description='Launch testing for a specific model')

    parser.add_argument("-t",'--testset',
                        default="unseen",

                        type=str,

                        choices=["seen", "unseen"],
                        required=True,
                        help='Choice of the test set 1-seen locations 2-unseen locations')

    parser.add_argument("-m",'--model',
                        default='alexnet',

                        type=str,

                        choices=["alexnet", "resnext50_32x4d", "vgg19"],
                        required=True,
                        help='Choice of the model')

    parser.add_argument("-d",'--dataset',
                        default=2,

                        type=int,

                        choices=[1, 2, 3, 4],
                        required=True,
                        help='Version of the training dataset used')


    return parser

def main() :



    criterion = torch.nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        warnings.warn("No gpu is available for the computation")


    #----- parsing arguments --------------------------------------
    parser=init_argparse()
    args=parser.parse_args()
    num_classes=14


    #------loading test set --------------------------------------
    prepro = preprocessing(img_size=320)
    preprocess = prepro.preprocessing()
    if args.testset =="seen":
        test_dataset = CustomImageDataset(f"data/data_split{args.dataset}/test",transform=preprocess)
    if args.testset=="unseen" :
        test_dataset = CustomImageDataset(f"data/test_set3/test",transform=preprocess)


    #----------------loading model -------------------------------
    model = torch.hub.load('pytorch/vision:v0.10.0', args.model, pretrained=True)
    if args.model in ["vgg19", "alexnet"]:
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes, bias=True)
    else:  # for resnext
        model.fc = torch.nn.Linear(2048, num_classes)

    batch_size = 16

    model.load_state_dict(
        torch.load(f"models/models_weights/{model._get_name()}/v{args.dataset}/{model._get_name()}.pt"))
    model = model.to(device)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8,
                                              pin_memory=True)  # keep









    a = time.time()
    running_loss, results = validation_loop(model=model, loader=tqdm.tqdm(test_loader), criterion=criterion, device=device)
    print("time :" ,(time.time() - a) / len(test_dataset))
    experiment = Experiment(f"{model._get_name()}/test")
    from custom_utils import metrics #had to reimport due to bug
    if experiment:
        for key in metrics:
            experiment.log_metric(key, metrics[key](results[1].numpy(), results[0].numpy()), epoch=0)
            # wandb.log({key: metrics[key](results[1].numpy(), results[0].numpy())})




    def answer(v):
        v = v.numpy()
        return np.where(np.max(v, axis=1) > 0.6, np.argmax(v, axis=1), 14)


    y_true, y_pred = answer(results[0]), answer(results[1])
    print(len(results),y_true.shape,y_pred.shape)
    # -----------------------------------------------------------------------------------
    metric = metrics(num_classes=14)
    metrics = metric.metrics()

    for metric in metrics.keys():
        print(metric + " : ", metrics[metric](y_true, y_pred))

    print("top-1",np.mean(np.where(y_true==y_pred,1,0)))
    m = confusion_matrix(y_true, y_pred, normalize="pred").round(2)
    #np.savetxt(f"{model._get_name()}_confusion_matrix.txt",m)
    print("avg class : ", np.mean(np.diag(m)))
    x = ['bobcat', 'opossum', 'car', 'coyote', 'raccoon', 'bird', 'dog', 'cat', 'squirrel', 'rabbit', 'skunk', 'fox',
         'rodent', 'deer', "empty"]
    z_text = [[str(y) for y in x] for x in m]

    import plotly.figure_factory as ff

    fig = ff.create_annotated_heatmap(m, x=x, y=x, annotation_text=z_text,colorscale="Blues")

    fig.update_layout(
        margin=dict(t=50, l=200),

        # title="ResNext50 3.0",
        xaxis_title="True labels",
        yaxis_title="Predictions",

    )

    fig['data'][0]['showscale'] = True
    import plotly.io as pio
    pio.write_image(fig, f"{model._get_name()}_conf_mat.png", width=1920, height=1080)
    fig.show()

if __name__ == "__main__":
    main()