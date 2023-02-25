# Welcome to PytorchTemplate!

This is a template for PyTorch projects. It is designed to be a starting point for new projects, and includes training and evaluation code for common tasks such as image classification and segmentation. It also includes a number of common components such as logging, visualization, and checkpointing.

## Installation

To install the dependencies, run:

```pip install -r requirements.txt ```

## Usage

The main class Experiment wraps Timm to load model, Weight & Biases to log metrics, 
and pytorch to train and evaluate models. The main function is defined in train.py, and can be run with the following command:
 ```
python train.py 
 ```
for help on the arguments, run:
 ```
python train.py --help
 ```

### Customization 

These templates file would be useless without the ability to rapidly customize them to your needs. This is where PytorchTemplate shines.
The following sections describe how to customize the template for your own project.


#### Config

Most parameters are already defined in Parser.py . Ideally every hyperparameters should be defined there as they will then be given 
to Weight&Biases as the config of the run. If those parameters are defined, and are used by the selected optimizer, PytorchTemplate will
automatically use them without any additional code.


#### Training/validation loop

If you wish to perform a different task than image classification, you will need to modify the training and validation loops.
You can find examples in PytorchTemplate/variation of children class of Experiment for adversarial training and Distillation.

Simply copy the body of thetraining loop, and modify it to your needs. Just make sure you import the right experiment in your training file!

#### Train.py

When the experiment is initialized, you then need to compile te different components of the training loop.


1. experiment = Experiment(args)
2. experiment.compile(args)
3. experiment.train()


Compile mostly take string arguments to define the model, the optimizer, and the loss function. While this may seem like a limitation,

ANY KEYWORD ARGUMENTS PASSED TO EXPERIMENT.TRAIN() WILL OVERRIDE THE DEFAULTS DEFINED IN COMPILE!

You can therefore, for example, write a custom loss function and simply call experiment.train(criterion=my_custom_loss) to use it!