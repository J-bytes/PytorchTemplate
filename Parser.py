import argparse
import os


def init_parser():
    parser = argparse.ArgumentParser(description="Launch training for a specific model")

    parser.add_argument(
        "--model",
        default="densenet201",
        const="all",
        type=str,
        nargs="?",
        required=False,
        help="Choice of the model",
    )

    parser.add_argument(
        "--img_size",
        default=500,
        const="all",
        type=int,
        nargs="?",
        required=False,
        help="width and length to resize the images to. Choose a value between 320 and 608.",
    )

    parser.add_argument(
        "--device",
        default=0,
        type=int,
        nargs="?",
        required=False,
        help="GPU on which to execute your code. Parallel to use all available gpus",
    )

    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="do you wish (and did you setup) wandb? You will need to add the project name in the initialization of wandb in train.py",
    )

    parser.add_argument(
        "--epoch",
        default=50,
        const="all",
        type=int,
        nargs="?",
        required=False,
        help="Number of epochs to train ; a patience of 5 is implemented by default",
    )
    parser.add_argument(
        "--augment_prob",
        default=[0,0,0,0,0,0],
        type=float,
        nargs="+",
        required=False,
        help="the probability of an augmentation. Between 0 and 1",
    )
    parser.add_argument(
        "--augment_intensity",
        default=0.1,
        const="all",
        type=float,
        nargs="?",
        required=False,
        help="The intensity of the data augmentation.Between 0 and 1. Default is 0.1",
    )
    parser.add_argument(
        "--label_smoothing",
        default=0,
        const="all",
        type=float,
        nargs="?",
        required=False,
        help="Label smoothing. Should be small. Try 0.05",
    )
    parser.add_argument(
        "--clip_norm",
        default=100,
        const="all",
        type=int,
        nargs="?",
        required=False,
        help="Norm for gradient clipping",
    )
    # parser.add_argument(
    #     "--pos_weight",
    #     default=1,
    #     const="all",
    #     type=int,
    #     nargs="?",
    #     required=False,
    #     help="A weight for the positive class.",
    # )
    parser.add_argument(
        "--lr",
        default=0.0001,
        const="all",
        type=float,
        nargs="?",
        required=False,
        help="learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        default=0.01,
        const="all",
        type=float,
        nargs="?",
        required=False,
        help="weight decay",
    )
    parser.add_argument(
        "--beta1",
        default=0.9,
        const="all",
        type=float,
        nargs="?",
        required=False,
        help="beta1 parameter adamw",
    )
    parser.add_argument(
        "--beta2",
        default=0.999,
        const="all",
        type=float,
        nargs="?",
        required=False,
        help="beta2 parameter of adamw",
    )
    parser.add_argument(
        "--batch_size",
        default=100,
        const="all",
        type=int,
        nargs="?",
        required=False,
        help="The batch size to use. If > max_batch_size,gradient accumulation will be used",
    )
    parser.add_argument(
        "--accumulate",
        default=1,
        const="all",
        type=int,
        nargs="?",
        required=False,
        help="The number of epoch to accumulate gradient. Choose anything >0",
    )
    parser.add_argument(
        "--num_worker",
        default=int(os.cpu_count() / 4),
        const="all",
        type=int,
        nargs="?",
        required=False,
        help="The number of process to use to retrieve the data. Please do not exceed 16",
    )

    parser.add_argument(
        "--channels",
        default=1,
        nargs="?",
        const="all",
        type=int,
        choices=[1, 3],
        required=False,
        help="The number of channels for the inputs",
    )

    parser.add_argument(
        "--tag",
        default=None,
        nargs="?",
        required=False,
        type=str,
        help="tag to add to the logs",
    )

    parser.add_argument(
        "--freeze",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="do you wish  to freeze the backbone? This option has been disabled for now",
    )
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="do you wish  to use pretrained weights?",
    )

    parser.add_argument(
        "--use_frontal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If this argument is given , the frontal images only will be used",
    )


    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="do you wish to run in debug mode ? Only 100 images will be loaded",
    )
    parser.add_argument(
        "--autocast",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="do you wish to disable autocast",
    )
    parser.add_argument(
        "--pretraining",
        default=0,
        nargs="?",
        const="all",
        type=int,

        required=False,
        help="Number of step to pretrain the model (in case you want to use diff. training config)",
    )
    parser.add_argument(
        "--drop_rate",
        default=0,
        nargs="?",
        const="all",
        type=float,

        required=False,
        help="The dropout rate. Must be between 0 and 1",
    )
    parser.add_argument(
        "--global_pool",
        default="avg",
        nargs="?",
        const="all",
        type=str,

        required=False,
        help="the type of pooling to effectuate before the classifier",
    )

    return parser
