import argparse
import os
import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import models
from torch.autograd import Variable
from data import get_dataset
from preprocess import get_transform
from utils import *
from datetime import datetime
from ast import literal_eval
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from models import small_network
from collections import OrderedDict


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ConvNet Training')

parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results',
                    help='results dir')
parser.add_argument('--save', metavar='SAVE', default='',
                    help='saved folder')
parser.add_argument('--dataset', metavar='DATASET', default='imagenet',
                    help='dataset name or folder')
parser.add_argument('--model', '-a', metavar='MODEL', default='alexnet',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: alexnet)')
parser.add_argument('--input_size', type=int, default=None,
                    help='image input size')
parser.add_argument('--model_config', default='',
                    help='additional architecture configuration')
parser.add_argument('--type', default='torch.cuda.FloatTensor',
                    help='type of tensor - e.g torch.cuda.HalfTensor')
parser.add_argument('--gpus', default='0',
                    help='gpus used for training - e.g 0,1,3')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=2500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--optimizer', default='SGD', type=str, metavar='OPT',
                    help='optimizer function used')
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', type=str, metavar='FILE',
                    help='evaluate model FILE on validation set')



def main():
    nb_hidden_neurons_1 = 512*4*4
    nb_hidden_neurons_2 = 128
    for iteration in range(4):
        
        path = f'Image_Augmentation_BNN_CIFAR_August17_Step_{iteration}.pt'
        
        global args, best_prec1
        best_acc=0
        args = parser.parse_args()
    
        if args.evaluate:
            args.results_dir = '/tmp'
        if args.save is '':
            args.save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        save_path = os.path.join(args.results_dir, args.save)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    
        setup_logging(os.path.join(save_path, 'log.txt'))
        results_file = os.path.join(save_path, 'results.%s')
        results = ResultsLog(results_file % 'csv', results_file % 'html')
    
        logging.info("saving to %s", save_path)
        logging.debug("run arguments: %s", args)
    
        if 'cuda' in args.type:
            args.gpus = [int(i) for i in args.gpus.split(',')]
            torch.cuda.set_device(args.gpus[0])
            cudnn.benchmark = True
        else:
            args.gpus = None
          

        if iteration==0:
            
            # create model for first iteration
            logging.info("creating model %s", args.model)
            model = models.__dict__[args.model]
            model_config = {'input_size': args.input_size, 'dataset': args.dataset}
        
            if args.model_config is not '':
                model_config = dict(model_config, **literal_eval(args.model_config))
        
            model = model(**model_config)
            logging.info("created model with configuration: %s", model_config)
            

            # Data loading code
            default_transform = {
                'train': get_transform(args.dataset,
                                       input_size=args.input_size, augment=True),
                'eval': get_transform(args.dataset,
                                      input_size=args.input_size, augment=False)
            }
            transform = getattr(model, 'input_transform', default_transform)
            regime = getattr(model, 'regime', {0: {'optimizer': args.optimizer,
                                                   'lr': args.lr,
                                                   'momentum': args.momentum,
                                                   'weight_decay': args.weight_decay}})
            # define loss function (criterion) and optimizer
            criterion = getattr(model, 'criterion', nn.CrossEntropyLoss)()
            criterion.type(args.type)
            model.type(args.type)
        
            test_data, empty_data = get_dataset(args.dataset, 'test', transform['eval'])
            test_loader = torch.utils.data.DataLoader(
                test_data,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)
        
            if args.evaluate:
                validate(val_loader, model, criterion, 0)
                return
        
            train_data, val_data = get_dataset(args.dataset, 'train', transform['train'])
            train_loader = torch.utils.data.DataLoader(
                train_data,
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True)
        
            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)
        
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
            logging.info('training regime: %s', regime)
        
            train_losses = [0] * args.epochs
            train_accuracies = [0] * args.epochs
            val_losses = [0] * args.epochs
            val_accuracies = [0] * args.epochs
            best_prec1 = 0
            
            for epoch in range(args.start_epoch, args.epochs):
                optimizer = adjust_optimizer(optimizer, epoch, regime)

                # train for one epoch
                train_loss, train_prec1, train_prec5 = train(
                    train_loader, model, criterion, epoch, optimizer)
        
                # evaluate on validation set
                val_loss, val_prec1, val_prec5 = validate(
                    val_loader, model, criterion, epoch)
        
                # remember best prec@1 and save checkpoint
                is_best = val_prec1 > best_prec1
                best_prec1 = max(val_prec1, best_prec1)
                if is_best:
                    print('Best validation accuracy yet, model is being saved...')
                    save_model(model,best_prec1,epoch,val_loss,path, optimizer)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model': args.model,
                    'config': args.model_config,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    'regime': regime
                }, is_best, path=save_path)
                logging.info('\n Epoch: {0}\t'
                             'Training Loss {train_loss:.4f} \t'
                             'Training Prec@1 {train_prec1:.3f} \t'
                             'Training Prec@5 {train_prec5:.3f} \t'
                             'Validation Loss {val_loss:.4f} \t'
                             'Validation Prec@1 {val_prec1:.3f} \t'
                             'Validation Prec@5 {val_prec5:.3f} \n'
                             .format(epoch + 1, train_loss=train_loss, val_loss=val_loss,
                                     train_prec1=train_prec1, val_prec1=val_prec1,
                                     train_prec5=train_prec5, val_prec5=val_prec5))
                train_losses[epoch] = train_loss
                val_losses[epoch] = val_loss
                train_accuracies[epoch] = train_prec1
                val_accuracies[epoch] = val_prec1
        
                results.add(epoch=epoch + 1, train_loss=train_loss, val_loss=val_loss,
                            train_error1=100 - train_prec1, val_error1=100 - val_prec1,
                            train_error5=100 - train_prec5, val_error5=100 - val_prec5)

                results.save()
            
            # Load the best weights to get the virtual points
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch=100

            # Print the graphs for the 0th iteration
            plt.figure(1)
            plt.title("Loss for iteration {}".format(iteration+1))
            plt.plot([loss for loss in train_losses[:-1]], label="Training Loss") 
            plt.plot([val_loss for val_loss in val_losses[:-1]], label="Validation Loss")
            plt.legend()
            plt.show()
        
            plt.figure(2)
            plt.title("Accuracy for iteration {}".format(iteration+1))
            plt.plot([acc_train for acc_train in train_accuracies[:-1]], label="Training Accuracy")
            plt.plot([acc for acc in val_accuracies[:-1]], label="Validation Accuracy")
            plt.legend()
            plt.show()

            print(' ')
            print('Highest training accuracy: ', max(train_accuracies))
            print('Highest validation accuracy: ', max(val_accuracies))
            print(' ')

            # evaluate on testing set
            test_loss, test_prec1, test_prec5 = test(test_loader, model, criterion, epoch)
            print(' ')
            print('Testing loss: ', test_loss)
            print('Testing Top1 accuracy: ', test_prec1)
            print('Testting Top5 accuracy: ', test_prec5)
            print(' ')
            
        else:

            model = models.Small_Network(iteration)
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
            criterion = nn.CrossEntropyLoss()

            ##############################################
            ####### INSERT TRAINING HERE #################

            # Merge the training and validation virtual points and then shuffle the whole thing

            #all_inputs_and_targets = torch.cat((all_inputs_and_targets_train,all_inputs_and_targets_val),dim=0)
            #all_inputs_and_targets=all_inputs_and_targets[torch.randperm(all_inputs_and_targets.size()[0])]
            #nb_inputs_train = all_inputs_and_targets_train.shape[0]
            #all_inputs_and_targets_train = all_inputs_and_targets[:nb_inputs_train,:]
            #all_inputs_and_targets_val = all_inputs_and_targets[nb_inputs_train:,:]

            training_accuracies = [0] * args.epochs
            training_losses = [0] * args.epochs
            validation_accuracies = [0] * args.epochs
            validation_losses = [0] * args.epochs

            for epoch in range(args.epochs):
                model.train()
                train_loss_add = 0.0
                correct_train = 0
                for i in range(len(train_loader)):
                    if i % 10 == 0:
                      print(i, end=" ")

                    optimizer.zero_grad()

                    inputs = all_inputs_and_targets_train[(i*args.batch_size):((i+1)*args.batch_size),:-1].cuda()
                    target = all_inputs_and_targets_train[(i*args.batch_size):((i+1)*args.batch_size),-1].cuda()
                    inputs = inputs.cuda()

                    target = target.long()
                    output, hook = model(inputs)

                    train_loss = criterion(output, target)
                    train_loss_add += train_loss
                    pred_train = output.data.max(1,keepdim=True)[1]
                    correct_train += pred_train.eq(target.data.view_as(pred_train)).cpu().sum()
                    train_loss.backward()
                    for p in list(model.parameters()):
                        if hasattr(p,'org'):
                            p.data.copy_(p.org)
                    optimizer.step()
                    for p in list(model.parameters()):
                        if hasattr(p,'org'):
                            p.org.copy_(p.data.clamp_(-1,1))
                    
                    if i % 100 == 0:
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch, i * len(inputs), len(train_loader.dataset),
                            100. * i / len(train_loader), train_loss.data.item()))
                  
                train_loss_mean = train_loss_add/len(train_loader)
                acc_train = 100.0 * correct_train/len(train_loader.dataset)
                print('Training accuracy for this epoch: ', acc_train, ' %')

                training_losses[epoch] = train_loss_mean
                training_accuracies[epoch] = acc_train

            ##########################################
            ######### EVALUATION IS HERE #############

                model.eval()
                val_loss = 0
                correct = 0
                i=0
                start = len(train_loader.dataset)

                with torch.no_grad():

                    for i in range(len(val_loader)):
                        input = all_inputs_and_targets_val[(i*args.batch_size):((i+1)*args.batch_size),:-1].cuda()
                        target = all_inputs_and_targets_val[(i*args.batch_size):((i+1)*args.batch_size),-1].cuda()

                        target = target.long()
                        output, hook = model(input)
                        val_loss += criterion(output, target)
                        pred = output.data.max(1,keepdim=True)[1]
                        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

                        i+=1

                    acc = 100. * correct/len(val_loader.dataset)

                    val_loss /= len(val_loader)
                    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
                        val_loss, correct, len(val_loader.dataset),
                        acc))
                
                    if acc > best_acc:
                      best_acc = acc
                      save_model(model, best_acc, epoch, val_loss, path, optimizer)
                
                validation_losses[epoch] = val_loss
                validation_accuracies[epoch] = acc


        # Print the graphs, starting at the second iteration
        if iteration>0:
            plt.figure(1)
            plt.title("Loss for iteration {}".format(iteration+1))
            plt.plot([loss for loss in training_losses[:-1]], label="Training Loss") 
            plt.plot([val_loss for val_loss in validation_losses[:-1]], label="Validation Loss")
            plt.legend()
            plt.show()
        
            plt.figure(2)
            plt.title("Accuracy for iteration {}".format(iteration+1))
            plt.plot([acc_train for acc_train in training_accuracies[:-1]], label="Training Accuracy")
            plt.plot([acc for acc in validation_accuracies[:-1]], label="Validation Accuracy")
            plt.legend()
            plt.show()

            print(' ')
            print('Highest training accuracy: ', max(training_accuracies))
            print('Highest validation accuracy: ', max(validation_accuracies))
            print(' ')

        # Load the best weights to get the virtual points
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Testing
        if iteration>0:
            model.eval()
            test_loss = 0
            correct = 0
            i=0

            with torch.no_grad():

                for i in range(len(test_loader)):
                    input = all_inputs_and_targets_test[(i*args.batch_size):((i+1)*args.batch_size),:-1].cuda()
                    target = all_inputs_and_targets_test[(i*args.batch_size):((i+1)*args.batch_size),-1].cuda()

                    target = target.long()
                    output, hook = model(input)
                    test_loss += criterion(output, target)
                    pred = output.data.max(1,keepdim=True)[1]
                    correct += pred.eq(target.data.view_as(pred)).cpu().sum()

                    i+=1

                test_acc = 100. * correct/len(test_loader.dataset)

                test_loss /= len(test_loader)

                print(' ')
                print('Testing: ')
                print('Accuracy: ', test_acc, ' %')
                print('Loss: ', test_loss)
                print(' ')

        # Create the virtual points for the trainset, validation set and testset
        if iteration==0:
          activ_1_train, all_inputs_and_targets_train = virtual_pts_generator(train_loader, model, criterion, epoch, nb_hidden_neurons_1, iteration, [])
          activ_1_val, all_inputs_and_targets_val = virtual_pts_generator(val_loader, model, criterion, epoch, nb_hidden_neurons_1, iteration, [])
          activ_1_test, all_inputs_and_targets_test = virtual_pts_generator(test_loader, model, criterion, epoch, nb_hidden_neurons_1, iteration, [])
        else:
          previous_inputs_and_targets_train = all_inputs_and_targets_train.clone()
          previous_inputs_and_targets_val = all_inputs_and_targets_val.clone()
          previous_inputs_and_targets_test = all_inputs_and_targets_test.clone()
          activ_1_train, all_inputs_and_targets_train = virtual_pts_generator(train_loader, model, criterion, epoch, nb_hidden_neurons_2, iteration, previous_inputs_and_targets_train)
          activ_1_val, all_inputs_and_targets_val = virtual_pts_generator(val_loader, model, criterion, epoch, nb_hidden_neurons_2, iteration, previous_inputs_and_targets_val)
          activ_1_test, all_inputs_and_targets_test = virtual_pts_generator(test_loader, model, criterion, epoch, nb_hidden_neurons_2, iteration, previous_inputs_and_targets_test)
            
        print('all inputs and targets train: ', all_inputs_and_targets_train)
        print(all_inputs_and_targets_train.shape) 
        print('all inputs and targets val: ', all_inputs_and_targets_val)
        print(all_inputs_and_targets_val.shape) 
        print('all inputs and targets test: ', all_inputs_and_targets_test)  
        print(all_inputs_and_targets_test.shape)  
        
      
        

def forward(data_loader, model, criterion, epoch=0, training=True, optimizer=None):
    if args.gpus and len(args.gpus) > 1:
        model = torch.nn.DataParallel(model, args.gpus)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for i, (inputs, target) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpus is not None:
            target = target.cuda()

        if not training:
            with torch.no_grad():
                input_var = Variable(inputs.type(args.type), volatile=not training)
                target_var = Variable(target)
                # compute output
                output, hook = model(input_var)

        else:
            input_var = Variable(inputs.type(args.type), volatile=not training)
            target_var = Variable(target)
            # compute output
            output, hook = model(input_var)

        loss = criterion(output, target_var)
        if type(output) is list:
            output = output[0]

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        if training:
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            for p in list(model.parameters()):
                if hasattr(p,'org'):
                    p.data.copy_(p.org)
            optimizer.step()
            for p in list(model.parameters()):
                if hasattr(p,'org'):
                    p.org.copy_(p.data.clamp_(-1,1))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logging.info('{phase} - Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                             epoch, i, len(data_loader),
                             phase='TRAINING' if training else 'EVALUATING',
                             batch_time=batch_time,
                             data_time=data_time, loss=losses, top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


def train(data_loader, model, criterion, epoch, optimizer):
    # switch to train mode
    model.train()
    return forward(data_loader, model, criterion, epoch,
                   training=True, optimizer=optimizer)


def validate(data_loader, model, criterion, epoch):
    # switch to evaluate mode
    model.eval()
    return forward(data_loader, model, criterion, epoch,
                   training=False, optimizer=None)

def test(data_loader, model, criterion, epoch):
    # switch to evaluate mode
    model.eval()
    return forward(data_loader, model, criterion, epoch,
                   training=False, optimizer=None)

def save_model(model, acc, epoch, loss, path, optimizer):
    print('==>>>Saving model ...')
    state = {
        'acc':acc,        
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss 
    }
    torch.save(state,path)
    print('*** DONE! ***')
    print(' ')



def virtual_pts_generator(data_loader, model, criterion, epoch, nb_hidden_neurons, iteration, previous_inputs_and_targets):
    training = False

    activ_1 = torch.zeros((len(data_loader.dataset),nb_hidden_neurons)).cuda()

    all_inputs_and_targets = torch.zeros((len(data_loader.dataset), nb_hidden_neurons+1)).cuda()

    # switch to evaluate mode
    model.eval()
    
    if args.gpus and len(args.gpus) > 1:
        model = torch.nn.DataParallel(model, args.gpus)

    if iteration==0:
        for i, (inputs, target) in enumerate(data_loader):
            if args.gpus is not None:
                target = target.cuda()
            
            with torch.no_grad():
                input_var = Variable(inputs.type(args.type), volatile=not training)
                target_var = Variable(target)
                # compute output
                output, hook = model(input_var)

            if i!=len(data_loader)-1:
              activ_1[(i*args.batch_size):((i+1)*args.batch_size),:] = hook
            else:
              activ_1[(i*args.batch_size):,:] = hook


            all_inputs_and_targets[(i*args.batch_size):((i+1)*args.batch_size),0:activ_1.shape[1]] = activ_1[i*args.batch_size:(i+1)*args.batch_size,:]
            all_inputs_and_targets[(i*args.batch_size):((i+1)*args.batch_size),-1] = target
        
    else:
        for i in range(len(data_loader)):
            with torch.no_grad():
                inputs = previous_inputs_and_targets[(i*args.batch_size):((i+1)*args.batch_size),:-1]
                inputs = inputs.cuda()
                output, hook = model(inputs)

            if i!=len(data_loader)-1:
              activ_1[(i*args.batch_size):((i+1)*args.batch_size),:] = hook
            else:
              activ_1[(i*args.batch_size):,:] = hook

            all_inputs_and_targets[(i*args.batch_size):((i+1)*args.batch_size),0:activ_1.shape[1]] = activ_1[i*args.batch_size:(i+1)*args.batch_size,:]
        
        previous_targets = previous_inputs_and_targets[:,-1]
        previous_activ = previous_inputs_and_targets[:,:-1]
        all_inputs_and_targets[:,-1] = previous_targets
        #print(previous_activ)
        #print(previous_activ.shape)
        #print(all_inputs_and_targets)
        #print(all_inputs_and_targets.shape)
        all_inputs_and_targets = torch.cat((previous_activ,all_inputs_and_targets), 1)

    print('activ_1: ', activ_1)
    print(activ_1.shape)
    print(' ')
        
    return activ_1, all_inputs_and_targets


if __name__ == '__main__':
    main()
