import torch.nn as nn
import torch.optim as optim
import torchvision.models.resnet as resnet

from avalanche.logging import InteractiveLogger, TextLogger
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, loss_metrics, timing_metrics, StreamConfusionMatrix
from avalanche.training.supervised import EWC, GEM, LwF, Naive, ICaRL
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins.lr_scheduling import LRSchedulerPlugin
from avalanche.models import IcarlNet, make_icarl_net, initialize_icarl_net

from utils.mlflow_logger import MLFlowLogger
from methods.custom_replay import *
from methods.custom_cumulative import *
from methods.custom_agem import *
from methods.debug_plugin import DebugPlugin
from methods.mir import *


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
        model = resnet.resnet18(num_classes=classes_per_task, pretrained=args.pretrained)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        strategy = CumulativeModified(model, optimizer, criterion,
                                      train_mb_size=args.batch_size, eval_mb_size=args.batch_size,
                                      device=device, train_epochs=args.n_epochs, plugins=plugins,
                                      evaluator=evaluation_plugin, eval_every=-1
                                      )
    elif args.method == 'naive':
        model = resnet.resnet18(num_classes=classes_per_task, pretrained=args.pretrained)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        strategy = Naive(model, optimizer, criterion,
                         train_mb_size=args.batch_size, eval_mb_size=args.batch_size,
                         device=device, train_epochs=args.n_epochs, plugins=plugins,
                         evaluator=evaluation_plugin, eval_every=-1
                         )
    elif args.method == 'ewc':
        model = resnet.resnet18(num_classes=classes_per_task, pretrained=args.pretrained)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0)
        criterion = nn.CrossEntropyLoss()
        ewc_lambda = 1000
        strategy = EWC(model, optimizer, criterion,
                       ewc_lambda=ewc_lambda, train_mb_size=args.batch_size, eval_mb_size=args.batch_size,
                       device=device, train_epochs=args.n_epochs, plugins=plugins, evaluator=evaluation_plugin)
    elif args.method == 'gem':
        model = resnet.resnet18(num_classes=classes_per_task, pretrained=args.pretrained)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        strategy = GEM(model, optimizer, criterion, patterns_per_exp=250,
                       train_mb_size=args.batch_size, eval_mb_size=args.batch_size, device=device,
                       train_epochs=args.n_epochs, plugins=plugins, evaluator=evaluation_plugin, eval_every=-1)
    elif args.method == 'agem':
        model = resnet.resnet18(num_classes=classes_per_task, pretrained=args.pretrained)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        strategy = AGEMModified(model, optimizer, criterion, patterns_per_exp=500, sample_size=256,
                                train_mb_size=args.batch_size, eval_mb_size=args.batch_size, device=device,
                                train_epochs=args.n_epochs, plugins=plugins, evaluator=evaluation_plugin, eval_every=-1)
    elif args.method == 'replay':
        model = resnet.resnet18(num_classes=classes_per_task, pretrained=args.pretrained)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        strategy = ReplayModified(model, optimizer, criterion, mem_size=250*args.n_experiences,
                                  train_mb_size=args.batch_size, eval_mb_size=args.batch_size, device=device,
                                  train_epochs=args.n_epochs, plugins=plugins, evaluator=evaluation_plugin, eval_every=-1)
    elif args.method == 'lwf':
        model = resnet.resnet18(num_classes=classes_per_task, pretrained=args.pretrained)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        strategy = LwF(model, optimizer, criterion, alpha=1.0, temperature=1.0,
                       train_mb_size=args.batch_size, eval_mb_size=args.batch_size, device=device,
                       train_epochs=args.n_epochs, plugins=plugins, evaluator=evaluation_plugin, eval_every=-1)
    elif args.method == 'mir':
        model = resnet.resnet18(num_classes=classes_per_task, pretrained=args.pretrained)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        strategy = Mir(model, optimizer, criterion, patterns_per_exp=250, sample_size=50,
                       train_mb_size=args.batch_size, eval_mb_size=args.batch_size, device=device,
                       train_epochs=args.n_epochs, plugins=plugins, evaluator=evaluation_plugin, eval_every=-1)
    elif args.method == 'icarl':
        model: IcarlNet = make_icarl_net(num_classes=classes_per_task)
        model.apply(initialize_icarl_net)

        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
        plugins.append(LRSchedulerPlugin(optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40, 45], gamma=0.2)))
        strategy = ICaRL(
            model.feature_extractor, model.classifier, optimizer,
            250*args.n_experiences,
            buffer_transform=None,
            fixed_memory=True, train_mb_size=args.batch_size,
            train_epochs=args.n_epochs, eval_mb_size=args.batch_size,
            plugins=plugins, device=device, evaluator=evaluation_plugin, eval_every=-1
        )

    return strategy, model, mlf_logger
