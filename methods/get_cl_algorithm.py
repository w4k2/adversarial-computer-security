import torch.nn as nn
import torch.optim as optim
import torchvision.models.resnet as resnet

from avalanche.logging import InteractiveLogger, TextLogger
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, loss_metrics, timing_metrics, StreamConfusionMatrix
from avalanche.training.supervised import EWC, GEM, LwF, Naive, ICaRL, GDumb, SynapticIntelligence, BiC
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins.lr_scheduling import LRSchedulerPlugin
from avalanche.models import IcarlNet, make_icarl_net, initialize_icarl_net

from utils.mlflow_logger import MLFlowLogger
from methods.custom_replay import *
from methods.custom_cumulative import *
from methods.custom_agem import *
from methods.debug_plugin import DebugPlugin
from methods.mir import *


def get_cl_algorithm(args, device, num_classes, single_channel=False, use_mlflow=True):
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
    )

    plugins = list()
    if args.debug:
        plugins.append(DebugPlugin())

    if args.method == 'cumulative':
        model = get_resnet(num_classes, single_channel)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        strategy = CumulativeModified(model, optimizer, criterion,
                                      train_mb_size=args.batch_size, eval_mb_size=args.batch_size,
                                      device=device, train_epochs=args.n_epochs, plugins=plugins,
                                      evaluator=evaluation_plugin, eval_every=-1
                                      )
    elif args.method == 'naive':
        model = get_resnet(num_classes, single_channel)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        strategy = Naive(model, optimizer, criterion,
                         train_mb_size=args.batch_size, eval_mb_size=args.batch_size,
                         device=device, train_epochs=args.n_epochs, plugins=plugins,
                         evaluator=evaluation_plugin, eval_every=-1
                         )
    elif args.method == 'ewc':
        model = get_resnet(num_classes, single_channel)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0)
        criterion = nn.CrossEntropyLoss()
        ewc_lambda = 1000
        strategy = EWC(model, optimizer, criterion,
                       ewc_lambda=ewc_lambda, train_mb_size=args.batch_size, eval_mb_size=args.batch_size,
                       device=device, train_epochs=args.n_epochs, plugins=plugins, evaluator=evaluation_plugin)
    elif args.method == 'gem':
        model = get_resnet(num_classes, single_channel)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        strategy = GEM(model, optimizer, criterion, patterns_per_exp=100,
                       train_mb_size=args.batch_size, eval_mb_size=args.batch_size, device=device,
                       train_epochs=args.n_epochs, plugins=plugins, evaluator=evaluation_plugin, eval_every=-1)
    elif args.method == 'agem':
        model = get_resnet(num_classes, single_channel)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        strategy = AGEMModified(model, optimizer, criterion, patterns_per_exp=100, sample_size=256,
                                train_mb_size=args.batch_size, eval_mb_size=args.batch_size, device=device,
                                train_epochs=args.n_epochs, plugins=plugins, evaluator=evaluation_plugin, eval_every=-1)
    elif args.method == 'replay':
        model = get_resnet(num_classes, single_channel)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        strategy = ReplayModified(model, optimizer, criterion, mem_size=100*args.n_experiences,
                                  train_mb_size=args.batch_size, eval_mb_size=args.batch_size, device=device,
                                  train_epochs=args.n_epochs, plugins=plugins, evaluator=evaluation_plugin, eval_every=-1)
    elif args.method == 'lwf':
        model = get_resnet(num_classes, single_channel)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        strategy = LwF(model, optimizer, criterion, alpha=1.0, temperature=1.0,
                       train_mb_size=args.batch_size, eval_mb_size=args.batch_size, device=device,
                       train_epochs=args.n_epochs, plugins=plugins, evaluator=evaluation_plugin, eval_every=-1)
    elif args.method == 'mir':
        model = get_resnet(num_classes, single_channel)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        strategy = Mir(model, optimizer, criterion, patterns_per_exp=100, sample_size=50,
                       train_mb_size=args.batch_size, eval_mb_size=args.batch_size, device=device,
                       train_epochs=args.n_epochs, plugins=plugins, evaluator=evaluation_plugin, eval_every=-1)
    elif args.method == 'icarl':
        model: IcarlNet = make_icarl_net(num_classes=num_classes, c=1 if single_channel else 3)
        model.apply(initialize_icarl_net)

        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
        plugins.append(LRSchedulerPlugin(optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40, 45], gamma=0.2)))
        strategy = ICaRL(
            model.feature_extractor, model.classifier, optimizer,
            100*args.n_experiences,
            buffer_transform=None,
            fixed_memory=True, train_mb_size=args.batch_size,
            train_epochs=args.n_epochs, eval_mb_size=args.batch_size,
            plugins=plugins, device=device, evaluator=evaluation_plugin, eval_every=-1
        )
    elif args.method == 'gdumb':
        model = get_resnet(num_classes, single_channel)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        strategy = GDumb(model, optimizer, criterion, mem_size=100*args.n_experiences,
                         train_mb_size=args.batch_size, eval_mb_size=args.batch_size, device=device,
                         train_epochs=args.n_epochs, plugins=plugins, evaluator=evaluation_plugin, eval_every=-1)
    elif args.method == 'si':
        model = get_resnet(num_classes, single_channel)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        strategy = SynapticIntelligence(model, optimizer, criterion, si_lambda=0.01,
                                        train_mb_size=args.batch_size, eval_mb_size=args.batch_size, device=device,
                                        train_epochs=args.n_epochs, plugins=plugins, evaluator=evaluation_plugin, eval_every=-1)
    elif args.method == 'bic':
        model = get_resnet(num_classes, single_channel)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        strategy = BiC(model, optimizer, criterion, mem_size=100*args.n_experiences, val_percentage=0.1, T=2,
                       train_mb_size=args.batch_size, eval_mb_size=args.batch_size, device=device,
                       train_epochs=args.n_epochs, plugins=plugins, evaluator=evaluation_plugin, eval_every=-1)

    return strategy, model, mlf_logger


def get_resnet(num_classes, single_input_channel=False):
    model = resnet.resnet18(num_classes=num_classes)
    if single_input_channel:
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model
