import torch.nn as nn
import torch.optim as optim
import torchvision.models.resnet as resnet

from avalanche.logging import InteractiveLogger, TextLogger
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, loss_metrics, timing_metrics, StreamConfusionMatrix
from avalanche.training.supervised import EWC, GEM, SynapticIntelligence, Cumulative, LwF, Naive, PNNStrategy
from avalanche.training.plugins import EvaluationPlugin
from avalanche.models import MultiHeadClassifier, MultiTaskModule


from utils.mlflow_logger import MLFlowLogger
from methods.custom_replay import *
from methods.custom_cumulative import *
from methods.custom_agem import *
from methods.debug_plugin import DebugPlugin
# from methods.mir import *
# from methods.hat import *


def get_cl_algorithm(args, device, classes_per_task, use_mlflow=True):
    loggers = list()
    if args.interactive_logger:
        loggers.append(InteractiveLogger())
    else:
        loggers.append(TextLogger())

    mlf_logger = None
    if use_mlflow:
        mlf_logger = MLFlowLogger(experiment_name=args.experiment, nested=args.nested_run, run_name=args.run_name)
        mlf_logger.log_parameters(args.__dict__)
        loggers.append(mlf_logger)

    evaluation_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=False, epoch=True, experience=True, stream=True),
        # forgetting_metrics(experience=True, stream=True),
        loss_metrics(minibatch=False, epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True),
        loggers=loggers,
        suppress_warnings=True)

    plugins = list()
    if args.debug:
        plugins.append(DebugPlugin())

    if args.method == 'cumulative':
        model = resnet18_multihead(num_classes=classes_per_task, pretrained=args.pretrained)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        strategy = CumulativeModified(model, optimizer, criterion,
                                      train_mb_size=args.batch_size, eval_mb_size=args.batch_size,
                                      device=device, train_epochs=args.n_epochs, plugins=plugins,
                                      evaluator=evaluation_plugin, eval_every=-1
                                      )
    elif args.method == 'naive':
        model = resnet18_multihead(num_classes=classes_per_task, pretrained=args.pretrained)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        strategy = Naive(model, optimizer, criterion,
                         train_mb_size=args.batch_size, eval_mb_size=args.batch_size,
                         device=device, train_epochs=args.n_epochs, plugins=plugins,
                         evaluator=evaluation_plugin, eval_every=-1
                         )
    elif args.method == 'ewc':
        model = resnet18_multihead(num_classes=classes_per_task, pretrained=args.pretrained)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0)
        criterion = nn.CrossEntropyLoss()
        # plugins.append(ConvertedLabelsPlugin())
        ewc_lambda = 1000
        strategy = EWC(model, optimizer, criterion,
                       ewc_lambda=ewc_lambda, train_mb_size=args.batch_size, eval_mb_size=args.batch_size,
                       device=device, train_epochs=args.n_epochs, plugins=plugins, evaluator=evaluation_plugin)
    elif args.method == 'si':
        model = resnet18_multihead(num_classes=classes_per_task, pretrained=args.pretrained)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0)
        criterion = nn.CrossEntropyLoss()
        # plugins.append(ConvertedLabelsPlugin())
        strategy = SynapticIntelligence(model, optimizer, criterion,
                                        si_lambda=1000, train_mb_size=args.batch_size, eval_mb_size=args.batch_size,
                                        device=device, train_epochs=args.n_epochs, plugins=plugins, evaluator=evaluation_plugin)
    elif args.method == 'gem':
        model = resnet18_multihead(num_classes=classes_per_task, pretrained=args.pretrained)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        strategy = GEM(model, optimizer, criterion, patterns_per_exp=250,
                       train_mb_size=args.batch_size, eval_mb_size=args.batch_size, device=device,
                       train_epochs=args.n_epochs, plugins=plugins, evaluator=evaluation_plugin, eval_every=-1)
    elif args.method == 'agem':
        model = resnet18_multihead(num_classes=classes_per_task, pretrained=args.pretrained)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        strategy = AGEMModified(model, optimizer, criterion, patterns_per_exp=500, sample_size=256,
                                train_mb_size=args.batch_size, eval_mb_size=args.batch_size, device=device,
                                train_epochs=args.n_epochs, plugins=plugins, evaluator=evaluation_plugin, eval_every=-1)
    elif args.method == 'pnn':
        # model = model((2, 2, 2, 2), in_features=3, hidden_features_per_column=64, classifier_in_size=256)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=False)
        criterion = nn.CrossEntropyLoss()
        strategy = PNNStrategy(model, optimizer, criterion, args.lr, args.weight_decay,
                               train_mb_size=args.batch_size, eval_mb_size=args.batch_size,
                               train_epochs=args.n_epochs, plugins=plugins, device=device, evaluator=evaluation_plugin, eval_every=-1)
    elif args.method == 'replay':
        model = resnet18_multihead(num_classes=classes_per_task, pretrained=args.pretrained)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        strategy = ReplayModified(model, optimizer, criterion, mem_size=250*args.n_experiences,
                                  train_mb_size=args.batch_size, eval_mb_size=args.batch_size, device=device,
                                  train_epochs=args.n_epochs, plugins=plugins, evaluator=evaluation_plugin, eval_every=-1)
    elif args.method == 'lwf':
        model = resnet18_multihead(num_classes=classes_per_task, pretrained=args.pretrained)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        strategy = LwF(model, optimizer, criterion, alpha=1.0, temperature=1.0,
                       train_mb_size=args.batch_size, eval_mb_size=args.batch_size, device=device,
                       train_epochs=args.n_epochs, plugins=plugins, evaluator=evaluation_plugin, eval_every=-1)
    # elif args.method == 'mir':
    #     model = resnet18_multihead(num_classes=classes_per_task, pretrained=args.pretrained)
    #     optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    #     criterion = nn.CrossEntropyLoss()
    #     strategy = Mir(model, optimizer, criterion, patterns_per_exp=250, sample_size=50,
    #                    train_mb_size=args.batch_size, eval_mb_size=args.batch_size, device=device,
    #                    train_epochs=args.n_epochs, plugins=plugins, evaluator=evaluation_plugin, eval_every=-1)
    # elif args.method == 'hat':
    #     model = HATModel(classes_per_task, args.image_size)  # , wide=10)
    #     optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0, weight_decay=0)
    #     lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
    #     plugins.append(LRSchedulerPlugin(lr_scheduler))
    #     strategy = HATStrategy(model, optimizer,
    #                            train_mb_size=args.batch_size, eval_mb_size=args.batch_size, device=device,
    #                            train_epochs=args.n_epochs, plugins=plugins, evaluator=evaluation_plugin, eval_every=-1)
    # elif args.method == 'cat':
    #     model = CATModel(classes_per_task, n_head=5, size=args.image_size)
    #     optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0, weight_decay=0)  # TODO check and change weight decay and momentum
    #     lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
    #     plugins.append(LRSchedulerPlugin(lr_scheduler))
    #     strategy = CATStrategy(model, optimizer, len(classes_per_task),
    #                            train_mb_size=args.batch_size, eval_mb_size=args.batch_size, device=device,
    #                            train_epochs=args.n_epochs, plugins=plugins, evaluator=evaluation_plugin, eval_every=-1)

    return strategy, model, mlf_logger


class MultiHeadReducedResNet18(MultiTaskModule):
    """
    As from GEM paper, a smaller version of ResNet18, with three times less feature maps across all layers.
    It employs multi-head output layer.
    """

    def __init__(self, base_model, output_size=160):
        super().__init__()
        self.resnet = base_model
        self.classifier = MultiHeadClassifier(output_size, masking=False)

    def forward(self, x, task_labels):
        out = self.resnet(x)
        return self.classifier(out, task_labels)


def resnet18_multihead(**kwargs):
    base_model = resnet.resnet18(**kwargs)
    output_size = base_model.fc.in_features
    base_model.fc = nn.Identity()
    model = MultiHeadReducedResNet18(base_model, output_size)
    return model
