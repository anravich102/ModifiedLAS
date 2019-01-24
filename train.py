
import argparse
import json
import os
import sys
import time
import torch.distributed as distributed
import torch.utils.data.distributed  # Distributed sampler
import torch.nn as nn
from tqdm import tqdm
from helper_functions import AverageMeter
from helper_functions import cross_entropy_with_masking_smoothing, to_np
from decoder import LasDecoder
from data.data_loader import AudioDataLoader, SpectrogramDataset, BucketingSampler, DistributedBucketingSampler
from model import Las, supported_rnns, gradient_hook
import torch
torch.set_printoptions(threshold=10000)
parser = argparse.ArgumentParser(description='LAS training')
parser.add_argument('--train_manifest', metavar='DIR',
                    help='path to train manifest csv', default='/home/ubuntu/anirudh/pytorch/deepspeech.pytorch/data/libri_train_manifest.csv')
parser.add_argument('--val_manifest', metavar='DIR',
                    help='path to validation manifest csv', default='/home/ubuntu/anirudh/pytorch/deepspeech.pytorch/data/libri_val_manifest.csv')
parser.add_argument('--batch_size', default=40, type=int, help='Batch size for training')
parser.add_argument('--batch_size_gpu', default=10, type=int, help='Size to split minibatch into (and then accumulate gradients) if the original batch size is too big for the GPU')
parser.add_argument('--num_workers', default=2, type=int, help='Number of workers used in data-loading')
parser.add_argument('--labels_path', default='labels.json', help='Contains all characters for transcription')
##
parser.add_argument('--sample_rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--window_size', default=.04, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--window_stride', default=.02, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
parser.add_argument('--fmin', default=20, type=int, help='Min freq in spectrogram.Required only if feature type is log_mel_spect')
parser.add_argument('--fmax', default=20000, type=int, help='Max freq in spectrogram.Required only if feature type is log_mel_spect')
parser.add_argument('--nmels', default=16, type=int, help='Number of mel bands. Required only if feature type is log_mel_spect')
parser.add_argument('--feature_type', default='mfcc', type=str, help='Audio feature type: options are log_mel_spect and log_spect')


##
parser.add_argument('--listener_hidden_size', default=256, type=int, help='Hidden size of RNNs')
parser.add_argument('--listener_hidden_layers', default=3, type=int, help='Number of encoder layers')
parser.add_argument('--listener_pyramidal', default=True, type=bool, help='Listener is pyramidal')
parser.add_argument('--listener_bidirectional', default=True, type=bool, help='Listener is bidirectional')
parser.add_argument('--listener_stacking_mode', default='avg', type=str, help='Listener stacking mode')
parser.add_argument('--listener_stacking_degree', default=2, type=int, help='Listener stacking degree')

parser.add_argument('--speller_hidden_size', default=512, type=int, help='Hidden size of RNNs')
parser.add_argument('--speller_hidden_layers', default=3, type=int, help='Number of decoder layers')
parser.add_argument('--speller_teacher_force_rate', default=1.0, type=float, help='teacher force rate during decoding')
parser.add_argument('--speller_decode_mode', default=0, type=int, help='Decoding mode (0 or  1)')
parser.add_argument('--speller_max_label_len', default=250, type=int, help='max label length to decode to')

parser.add_argument('--rnn_type', default='lstm', help='Type of the RNN. rnn|gru|lstm are supported')
parser.add_argument('--batch_norm', default=False, type=bool, help='Batch norm')


##
parser.add_argument('--epochs', default=300, type=int, help='Number of training epochs')
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=5e-2, type=float, help='initial learning rate (per_gpu)')
parser.add_argument('--momentum', default=0.95, type=float, help='momentum')
parser.add_argument('--max_norm', default=0, type=int, help='Norm cutoff to prevent explosion of gradients')
parser.add_argument('--learning_anneal', default=1.1, type=float, help='Annealing applied to learning rate every epoch')
##

parser.add_argument('--silent', dest='silent', action='store_true', help='Turn off progress tracking per iteration')
parser.add_argument('--checkpoint', dest='checkpoint', action='store_true', help='Enables checkpoint saving of model')
parser.add_argument('--checkpoint_per_batch', default=0, type=int, help='Save checkpoint per batch. 0 means never save')
parser.add_argument('--save_folder', default='models/', help='Location to save epoch models')
parser.add_argument('--model_path', default='models/las_final.pth',
                    help='Location to save best validation model')
parser.add_argument('--continue_from', default='', help='Continue from checkpoint model')
parser.add_argument('--finetune', dest='finetune', action='store_true',
                    help='Finetune the model from checkpoint "continue_from"')

#
parser.add_argument('--tensorboard', dest='tensorboard', action='store_true', help='Turn on tensorboard graphing')
parser.add_argument('--log_dir', default='visualize/LAS_final', help='Location of tensorboard log')
parser.add_argument('--log_params', dest='log_params', action='store_true', help='Log parameter values and gradients')
parser.add_argument('--id', default='LAS training', help='Identifier for visdom/tensorboard run')
#

parser.add_argument('--augment', dest='augment', action='store_true', help='Use random tempo and gain perturbations.')
parser.add_argument('--noise_dir', default=None,
                    help='Directory to inject noise into audio. If default, noise Inject not added')
parser.add_argument('--noise_prob', default=0.4, help='Probability of noise being added per sample')
parser.add_argument('--noise_min', default=0.0,
                    help='Minimum noise level to sample from. (1.0 means all noise, not original signal)', type=float)
parser.add_argument('--noise_max', default=0.5,
                    help='Maximum noise levels to sample from. Maximum 1.0', type=float)
parser.add_argument('--no_shuffle', dest='no_shuffle', action='store_true',
                    help='Turn off shuffling and sample from dataset based on sequence length (smallest to largest)')
parser.add_argument('--no_sortaGrad', dest='no_sorta_grad', action='store_true',
                    help='Turn off ordering of dataset on sequence length for the first epoch.')


###
parser.add_argument('--backend', type=str, default='tcp', help='Name of the backend to use.')
parser.add_argument('--init_method', '-i', type=str, default='tcp://172.31.7.132:23456',
                    help='URL specifying how to initialize the package.')
parser.add_argument('--rank', '-r', type=int, help='Rank of the current process.')
parser.add_argument('--world_size', '-s', type=int, help='Number of processes participating in the job.')


torch.manual_seed(123456)
torch.cuda.manual_seed_all(123456)


def init_process(args):
    distributed.init_process_group(
        backend=args.backend,
        init_method=args.init_method,
        rank=args.rank,
        world_size=args.world_size)


def init_xavier(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.zero_()


def average_gradients(model, device):
    world_size = distributed.get_world_size()
    # print("iprinting model param gradients")
    for p in model.parameters():
        # print(p)

        group = distributed.new_group(ranks=list(range(world_size)))
        tensor = p.grad.data.cpu()
        # print(tensor)
        distributed.all_reduce(tensor, op=distributed.reduce_op.SUM, group=group)
        tensor /= float(world_size)
        p.grad.data = tensor.to(device)


if __name__ == '__main__':

    args = parser.parse_args()
    if args.rank == 0:
        print(args)

    init_process(args)

    main_proc = args.rank == 0  # Only the first proc should save models
    device = torch.device('cuda' if args.cuda else 'cpu')
    save_folder = args.save_folder

    loss_results, cer_results, wer_results = torch.Tensor(args.epochs), torch.Tensor(args.epochs), torch.Tensor(args.epochs)
    best_wer = None

    if args.tensorboard and main_proc:
        os.makedirs(args.log_dir, exist_ok=True)
        from tensorboardX import SummaryWriter
        tensorboard_writer = SummaryWriter(args.log_dir)

    os.makedirs(save_folder, exist_ok=True)
    avg_loss, start_epoch, start_iter = 0, 0, 0
    if args.continue_from:  # Starting from previous model
        print("Loading checkpoint model %s" % args.continue_from)
        package = torch.load(args.continue_from, map_location=lambda storage, loc: storage)

        model = Las.load_model_package(package)
        labels = Las.get_labels(model)
        audio_conf = Las.get_audio_conf(model)
        parameters = model.parameters()
        optimizer = torch.optim.SGD(parameters, lr=(args.lr),
                                    momentum=args.momentum, nesterov=True)

        if not args.finetune:  # not finetuning( restarting training) --> so  load optimizer state:
            if args.cuda:
                model = model.to(device)
            optimizer.load_state_dict(package['optim_dict'])
            start_epoch = int(package.get('epoch', 1)) - 1  # Index start at 0 for training
            start_iter = package.get('iteration', None)
            if start_iter is None:
                start_epoch += 1  # We saved model after epoch finished, start at the next epoch.
                start_iter = 0
            else:
                start_iter += 1

            avg_loss = int(package.get('avg_loss', 0))
            loss_results, cer_results, wer_results = package['loss_results'], package[
                'cer_results'], package['wer_results']

            # populate tensorboard with history before continue_from:
            if main_proc and args.tensorboard and \
                package[
                    'loss_results'] is not None and start_epoch > 0:  # Previous scores to tensorboard logs
                for i in range(start_epoch):
                    values = {
                        'Avg Train Loss': loss_results[i],
                        'Avg WER': wer_results[i],
                        'Avg CER': cer_results[i]
                    }
                    tensorboard_writer.add_scalars(args.id, values, i + 1)

    else:

        # initialize new model

        with open(args.labels_path) as label_file:
            labels = str(''.join(json.load(label_file)))
            print("labels: %s" % labels)

        audio_conf = dict(sample_rate=args.sample_rate,
                          window_size=args.window_size,
                          window_stride=args.window_stride,
                          window=args.window,
                          fmin=args.fmin,
                          fmax=args.fmax,
                          nmels=args.nmels,
                          feature_type=args.feature_type,
                          noise_dir=args.noise_dir,
                          noise_prob=args.noise_prob,
                          noise_levels=(args.noise_min, args.noise_max))

        rnn_type = args.rnn_type.lower()
        assert rnn_type in supported_rnns, "rnn_type should be either lstm, rnn or gru"
        model = Las(audio_conf, rnn_type=supported_rnns[rnn_type],
                    labels=labels,
                    listener_hidden_size=args.listener_hidden_size,
                    nb_listener_layers=args.listener_hidden_layers,
                    listener_stacking_mode=args.listener_stacking_mode,
                    listener_stacking_degree=args.listener_stacking_degree,
                    listener_is_pyramidal=args.listener_pyramidal,
                    listener_is_bidirectional=args.listener_bidirectional,
                    speller_hidden_size=args.speller_hidden_size,
                    nb_speller_layers=args.speller_hidden_layers,
                    batch_norm=args.batch_norm)
        parameters = model.parameters()
        # optimizer = torch.optim.SGD(parameters, lr=(args.lr * args.world_size),
        #                             momentum=args.momentum, nesterov=False)
        optimizer = torch.optim.ASGD(parameters, lr=(args.lr), weight_decay=0.0005)
        optimizer.zero_grad()

    #criterion = cross_entropy_with_masking_smoothing
    criterion = nn.CrossEntropyLoss(reduce=False)
    decoder = LasDecoder(labels)

    train_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.train_manifest, labels=labels,
                                       normalize=False, augment=args.augment)
    val_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.val_manifest, labels=labels,
                                     normalize=False, augment=False)

    train_sampler = DistributedBucketingSampler(train_dataset,
                                                batch_size=args.batch_size_gpu,
                                                num_replicas=args.world_size, rank=args.rank)

    train_loader = AudioDataLoader(train_dataset,
                                   num_workers=args.num_workers, batch_sampler=train_sampler)
    val_loader = AudioDataLoader(val_dataset, batch_size=1,
                                 num_workers=args.num_workers)

    if (not args.no_shuffle and start_epoch != 0) or args.no_sorta_grad:
        print("Shuffling batches for the following epochs")
        train_sampler.shuffle(start_epoch)

    model.apply(init_xavier)
    model = model.to(device)

    if main_proc:
        print(model)
        for name, param in model.named_parameters():
            if param.requires_grad:
                print (name, param.size())
        print("Number of parameters: %d" % Las.get_param_size(model))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        optimizer.zero_grad()
        start_epoch_time = time.time()

        accumulate_steps = args.batch_size / float(args.batch_size_gpu)
        batch_loss = torch.Tensor([0.0]).to(device)
        batch_data_time = 0.0
        batch_overall_time = 0.0

        for i, (data) in enumerate(train_loader, start=start_iter):
            end = time.time()
            accumulate_steps -= 1
            # if i == 15:
            #     break

            if i == len(train_sampler):
                break

            inputs, labels, seq_lens, target_sizes, original_text, targets = data
            # input_sizes = input_percentages.mul_(int(inputs.size(0))).int()
            # print("labels fetched from loader", labels.size())
            # print("inputs.size", inputs.size())
            # # Labels = LxNxV
            # # measure data loading time
            # # print("Labels fetched from loader:")
            # print("original_text:", original_text[0])
            # print("labels:", labels[:2, 0, :])
            # print(target_sizes[0])

            batch_data_time += time.time() - end
            if accumulate_steps == 0:
                data_time.update(batch_data_time, n=args.batch_size)
                batch_data_time = 0.0

            if args.cuda:
                inputs = inputs.to(device)  # (TxNxH)
                labels = labels.to(device)  # (L x N x V)
                target_sizes = target_sizes.to(device)  # (N,)
                seq_lens = seq_lens.to(device)
                targets = targets.to(device)
                # print("labels being fed into model:", labels.size())

            out = model(inputs, seq_lens, labels, device=device, teacher_force_rate=args.speller_teacher_force_rate, max_label_len=args.speller_max_label_len,
                        decode_mode=args.speller_decode_mode)  # out is a list od tensors of shape (N,1,V)

            out = torch.cat(out, dim=1)  # (N,L-1,V)
            label_len = out.size(1)
            out = out.view(-1, out.size(2))

            # out.register_hook(lambda gradient: gradient_hook(gradient, "gradient of out: "))

            # out.backward(torch.ones_like(out))
            # sys.exit(0)

            # output is on gpu - (N x  L-1 x V)
            # print("out[0,:5,:][: ", out[0, :5, :])
            # batch_transcript = decoder.batch_decode(out)
            # if i % 100 == 0:
            #     for reference, transcript in zip(original_text, batch_transcript):

            #         print(reference, ' vs ', transcript)

            labels = labels.transpose(0, 1)  # (N x L x V)
            labels = labels[:, 1:, :]  # remove <sos>  #(N x L-1 x V)
            # print("after model:")
            # print("out.size:", out.size())
            # print("labels before sending to loss", labels[0])
            # print("labels before sending to loss", labels[1])

            #########
            # loss = criterion(logits=out, gold=labels, masking=False, smoothing=False, seq_lens=target_sizes, cuda=args.cuda, average_across_batch=False)  # loss is on gpu or cpu depending on args.cuda
            loss = criterion(out, targets)
            loss = loss.sum()  # sum losses across batch
            loss = loss / label_len
            loss = loss / args.batch_size  # divided by (original) batch size
            batch_loss += loss
            loss.backward()
            # sys.exit(0)

            if accumulate_steps != 0:
                batch_overall_time += time.time() - end
                continue
            else:
                total_norm = 0.0
                for p in model.parameters():
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                print("total_norm of gradient before: ", total_norm)
                # clip gradients
                if args.max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)

                # average the gradients
                average_gradients(model, device)

                # take one step
                optimizer.step()

                batch_loss_value = batch_loss.item()  # contains average loss for the minibatch

                avg_loss += batch_loss_value
                losses.update(batch_loss_value, args.batch_size)

                # measure elapsed time
                batch_overall_time += time.time() - end
                batch_time.update(batch_overall_time, n=args.batch_size)
                batch_overall_time = 0.0

                if not args.silent:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t '
                          'Proc# {3}\t '.format(
                              (epoch + 1), (i + 1), len(train_sampler), (args.rank), batch_time=batch_time, data_time=data_time, loss=losses))

                if args.tensorboard and main_proc:
                    step = epoch * len(train_sampler) + i
                    tensorboard_writer.add_scalar('Average Minibatch Loss', batch_loss_value, step)

                    if args.log_params and i % 100 == 0:
                        for tag, value in model.named_parameters():
                            tag = tag.replace('.', '/')
                            tensorboard_writer.add_histogram(tag, to_np(value), i + 1)
                            tensorboard_writer.add_histogram(tag + '/grad', to_np(value.grad), i + 1)

                if args.checkpoint_per_batch > 0 and i > 0 and (i + 1) % args.checkpoint_per_batch == 0 and main_proc:
                    file_path = '%s/las_checkpoint_epoch_%d_iter_%d.pth' % (save_folder, epoch + 1, i + 1)

                    print("Saving checkpoint model to %s" % file_path)
                    torch.save(Las.serialize(model, optimizer=optimizer, epoch=epoch, iteration=i, loss_results=loss_results, wer_results=wer_results,
                                             cer_results=cer_results, avg_loss=avg_loss), file_path)
                    # sys.exit(0)

                # reset
                optimizer.zero_grad()
                # model.zero_grad()
                accumulate_steps = args.batch_size / float(args.batch_size_gpu)
                batch_loss = 0.0

                del loss
                del out

        avg_loss /= len(train_sampler)  # contains average loss for the epoch (which is the average loss over batches average further across batches --> len(train_sampler) givers the number of minibatches in an epoch  for that process)

        epoch_time = time.time() - start_epoch_time

        if main_proc:
            print('Main Proc Training Summary Epoch: [{0}]\t'
                  'Time taken (s): {epoch_time:.0f}\t'
                  'Average Loss {loss:.3f}\t'.format(epoch + 1, epoch_time=epoch_time, loss=avg_loss))

        start_iter = 0  # Reset start iteration for next epoch

        # validation
        if main_proc:
            total_cer, total_wer = 0, 0
            model.eval()

            with torch.no_grad():  # TODO: need Kenlm integration for scorer

                for i, (data) in tqdm(enumerate(val_loader), total=len(val_loader)):
                    inputs, labels, seq_lens, target_sizes, original_text, targets = data

                    if i == 10:
                        break

                    if args.cuda:
                        inputs = inputs.to(device)  # dont send labels for val
                        target_sizes = target_sizes.to(device)
                        seq_lens = seq_lens.to(device)
                        targets = targets.to(device)

                    out = model(inputs, seq_lens, device=device, teacher_force_rate=args.speller_teacher_force_rate, max_label_len=args.speller_max_label_len,
                                decode_mode=1)

                    # out is (N x L-1 X V)
                    out = torch.cat(out, dim=1)

                    out_cpu = out.cpu()
                    batch_transcript = decoder.batch_decode(out_cpu)

                    wer, cer = 0, 0

                    for reference, transcript in zip(original_text, batch_transcript):
                        if i < 5:
                            print(reference, ' vs ', transcript)
                        wer += decoder.wer(transcript, reference) / float(len(reference.split()))
                        cer += decoder.cer(transcript, reference) / float(len(reference))

                    total_cer += cer
                    total_wer += wer
                    del out

                # total_wer = torch.Tensor([total_wer], device=torch.device('cpu'))
                # total_cer = torch.Tensor([total_cer], device=torch.device('cpu'))

                # group = distributed.new_group(ranks=list(range(args.world_size)))

                # # let each gpu finish  validating on its portion of dataset -> then combine wer,cer results:
                # torch.distributed.reduce(total_wer, dst=0, op=distributed.reduce_op.SUM, group=group)
                # torch.distributed.reduce(total_cer, dst=0, op=distributed.reduce_op.SUM, group=group)

                avg_wer = total_wer / len(val_loader.dataset)
                avg_cer = total_cer / len(val_loader.dataset)
                avg_wer *= 100
                avg_cer *= 100

                loss_results[epoch] = avg_loss
                wer_results[epoch] = avg_wer
                cer_results[epoch] = avg_cer

                print('Validation Summary Epoch: [{0}]\t'
                      'Average WER {wer:.3f}\t'
                      'Average CER {cer:.3f}\t'.format(epoch + 1, wer=avg_wer, cer=avg_cer))

            if args.tensorboard and main_proc:
                values = {
                    'Avg Train Loss': avg_loss,
                    'Avg WER': avg_wer,
                    'Avg CER': avg_cer
                }
                tensorboard_writer.add_scalars(args.id, values, epoch + 1)
                # if args.log_params:
                #     for tag, value in model.named_parameters():
                #         tag=tag.replace('.', '/')
                #         tensorboard_writer.add_histogram(tag, to_np(value), epoch + 1)
                #         tensorboard_writer.add_histogram(tag + '/grad', to_np(value.grad), epoch + 1)

            if args.checkpoint and main_proc:
                file_path = '%s/LAS_%d.pth' % (save_folder, epoch + 1)
                torch.save(Las.serialize(model, optimizer=optimizer, epoch=epoch, loss_results=loss_results,
                                         wer_results=wer_results, cer_results=cer_results),
                           file_path)
                # anneal lr
                optim_state = optimizer.state_dict()
                optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0]['lr'] / args.learning_anneal
                optimizer.load_state_dict(optim_state)
                print('Learning rate annealed to: {lr:.6f}'.format(lr=optim_state['param_groups'][0]['lr']))

            if (best_wer is None or best_wer > avg_wer) and main_proc:
                print("Found better validated model, saving to %s" % args.model_path)
                torch.save(Las.serialize(model, optimizer=optimizer, epoch=epoch, loss_results=loss_results,
                                         wer_results=wer_results, cer_results=cer_results), args.model_path)
                best_wer = avg_wer
                avg_loss = 0

            if not args.no_shuffle:
                print("Shuffling batches...")
                train_sampler.shuffle(epoch)

                # anneal lr
                optim_state = optimizer.state_dict()
                optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0]['lr'] / args.learning_anneal
                optimizer.load_state_dict(optim_state)
                print('Learning rate annealed to: {lr:.6f}'.format(lr=optim_state['param_groups'][0]['lr']))
