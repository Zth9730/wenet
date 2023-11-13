import logging
from contextlib import nullcontext
import torch.distributed as dist

# if your python version < 3.7 use the below one
# from contextlib import suppress as nullcontext
import torch
from torch.nn.utils import clip_grad_norm_
import datetime
import yaml
from wenet.utils.checkpoint import save_checkpoint
import os

class Executor:
    def __init__(self):
        self.step = 0
        self.back_loss = 100000
    def train(self, model, optimizer, scheduler, data_loader, cv_data_loader, device, writer,
              args, scaler, logger, model_dir):
        ''' Train one epoch
        '''
        model.train()
        clip = args.get('grad_clip', 50.0)
        log_interval = args.get('log_interval', 10)
        save_interval = args.get('save_interval', 5000)
        finish_step = args.get('step_nums', 400000)
        rank = args.get('rank', 0)
        epoch = args.get('epoch', 0)
        accum_grad = args.get('accum_grad', 1)
        is_distributed = args.get('is_distributed', True)
        is_deepspeed = args.get('is_deepspeed', False)

        use_amp = args.get('use_amp', False)
        ds_dtype = args.get('ds_dtype', "fp32")
        if ds_dtype == "fp16":
            ds_dtype = torch.float16
        elif ds_dtype == "bf16":
            ds_dtype = torch.bfloat16
        else:
            ds_dtype = None
        logging.info('using accumulate grad, new batch size is {} times'
                     ' larger than before'.format(accum_grad))
        if use_amp:
            assert scaler is not None
        # A context manager to be used in conjunction with an instance of
        # torch.nn.parallel.DistributedDataParallel to be able to train
        # with uneven inputs across participating processes.
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_context = model.join
        else:
            model_context = nullcontext
        num_seen_utts = 0
        target_ids_all = []

        with model_context():
            for batch_idx, batch in enumerate(data_loader):
                key, feats, labels, feats_lengths, label_lengths = batch
                feats = feats.to(device)
                feats_lengths = feats_lengths.to(device)
                num_utts = feats.size(0)
                if num_utts == 0:
                    continue
                context = None
                # Disable gradient synchronizations across DDP processes.
                # Within this context, gradients will be accumulated on module
                # variables, which will later be synchronized.
                if is_distributed and batch_idx % accum_grad != 0:
                    context = model.no_sync
                # Used for single gpu training and DDP gradient synchronization
                # processes.
                else:
                    context = nullcontext
                with context():
                    if is_deepspeed:  # deepspeed
                        with torch.cuda.amp.autocast(
                            enabled=ds_dtype is not None,
                            dtype=ds_dtype, cache_enabled=False
                        ):
                            loss_dict = model(feats, feats_lengths, self.step)
                        loss = loss_dict['loss']
                        # NOTE(xcsong): Zeroing the gradients is handled automatically by DeepSpeed after the weights # noqa
                        #   have been updated using a mini-batch. DeepSpeed also performs gradient averaging automatically # noqa
                        #   at the gradient accumulation boundaries and addresses clip_grad_norm internally. In other words # noqa
                        #   `model.backward(loss)` is equivalent to `loss.backward() + clip_grad_norm_() + optimizer.zero_grad() + accum_grad` # noqa
                        #   ref: https://www.deepspeed.ai/tutorials/megatron/#using-the-training-api  # noqa
                        model.backward(loss)
                    else:             
                        # pytorch native ddp
                        # autocast context
                        # The more details about amp can be found in
                        # https://pytorch.org/docs/stable/notes/amp_examples.html
                        with torch.cuda.amp.autocast(scaler is not None):
                            loss_dict, target_ids = model(feats, feats_lengths, self.step)
                            target_ids_all.append(target_ids.reshape(-1))
                            loss = loss_dict['loss'] / accum_grad
                        if use_amp:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()
                

                num_seen_utts += num_utts
                if is_deepspeed:
                    if rank == 0 and writer is not None \
                            and model.is_gradient_accumulation_boundary():
                        writer.add_scalar('train_loss', loss.item(), self.step)
                    # NOTE(xcsong): The step() function in DeepSpeed engine updates the model parameters as well as the learning rate. There is # noqa
                    #   no need to manually perform scheduler.step(). In other words: `ds_model.step() = optimizer.step() + scheduler.step()` # noqa
                    #   ref: https://www.deepspeed.ai/tutorials/megatron/#using-the-training-api  # noqa
                    model.step()
                    self.step += 1
                elif not is_deepspeed and batch_idx % accum_grad == 0:
                    if rank == 0 and writer is not None:
                        # import pdb
                        # pdb.set_trace()
                        # gradients = torch.autograd.grad(loss, model.parameters())
                        # writer.add_scalar('grad_norm', gradients, self.step)
                        for keys, values in loss_dict.items():
                            if values:
                                writer.add_scalar(keys, values, self.step)
                        # writer.add_scalar('train_codes_acc', loss_dict['codes_acc'], self.step)
                    # Use mixed precision training
                    if use_amp:
                        scaler.unscale_(optimizer)
                        grad_norm = clip_grad_norm_(model.parameters(), clip)
                        # Must invoke scaler.update() if unscale_() is used in
                        # the iteration to avoid the following error:
                        #   RuntimeError: unscale_() has already been called
                        #   on this optimizer since the last update().
                        # We don't check grad here since that if the gradient
                        # has inf/nan values, scaler.step will skip
                        # optimizer.step().
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        grad_norm = clip_grad_norm_(model.parameters(), clip)
                        if torch.isfinite(grad_norm):
                            optimizer.step()
                            
                    if torch.isfinite(grad_norm) and rank == 0 and writer is not None:
                        writer.add_scalar('grad_norm', grad_norm, self.step)
                        
                    optimizer.zero_grad()
                    scheduler.step()
                    self.step += 1
                if batch_idx % log_interval == 0:
                    lr = optimizer.param_groups[0]['lr']
                    if writer is not None:
                        writer.add_scalar('lr', lr, self.step)
                    log_str = 'TRAIN Batch {}/{} loss {:.6f} '.format(
                        epoch, batch_idx,
                        loss.item() * accum_grad)
                    for name, value in loss_dict.items():
                        if name != 'loss' and value is not None:
                            log_str += '{} {:.6f} '.format(
                                name,
                                value.item() if isinstance(
                                    value, torch.Tensor) else value)
                    log_str += 'lr {:.8f} '.format(lr)
                    log_str += 'step {} rank {}'.format(self.step, rank)
                    # logging.info(log_str)
                    logger.info(log_str)
                # if loss > 2* self.back_loss:    # loss崩塌时打印key
                #     logger.info('rank {} error keys are: {}'.format(rank, key))
                # else:
                #     self.back_loss = loss
                
                if self.step !=0 and self.step % save_interval == 0:
                    total_loss_dict, num_seen_utts = self.cv(model, cv_data_loader, device, args, logger)
                    cv_loss_dict = {}
                    for name, value in total_loss_dict.items():
                        if value is not None:
                            cv_loss_dict['cv_' + name] = value / num_seen_utts
                            logger.info('name {} value {}'.format(name, value))
                    cv_loss = total_loss_dict['total_loss'] / num_seen_utts
                    if rank == 0:
                    
                    # writer.add_histogram(tag='last_layer_conv_norm', values=model.module.encoder.encoders[-1].norm_conv.weight, global_step=self.step)
                    # writer.add_histogram(tag='last_layer_ffn_norm', values=model.module.encoder.encoders[-1].norm_ff.weight, global_step=self.step)
                    # writer.add_histogram(tag='last_layer_self-attn_norm', values=model.module.encoder.encoders[-1].norm_mha.weight, global_step=self.step)
            
                        logger.info('Epoch {} CV info cv_loss {}'.format(epoch, cv_loss))
                        save_model_path = os.path.join(model_dir, '{}.pt'.format(self.step))
                        save_infos = {
                                'epoch': epoch,
                                'lr': lr,
                                'cv_loss': cv_loss,
                                'step': self.step
                            }
                        # save_infos.update(cv_loss_dict)
                        save_checkpoint(
                            model, save_model_path, save_infos)
                        writer.add_scalar('epoch/cv_loss', cv_loss, epoch)
                        writer.add_scalar('epoch/lr', lr, epoch)
                    # infos = {
                    # 'epoch': epoch, 'lr': lr, 'step': self.step,
                    # 'save_time': datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
                    # with open("{}/{}.yaml".format(model_dir, self.step), 'w') as fout:
                    #     data = yaml.dump(infos)
                    #     fout.write(data)
                    # save_model_path = os.path.join(model_dir, '{}.pt'.format(self.step))
                    # save_checkpoint(model, save_model_path, infos)
                    target_ids_all = torch.cat(target_ids_all, dim=0).reshape(-1)
                    logger.info('epoch {} codebook embedding usage number: {}'.format(epoch, target_ids_all.unique().numel()))
                    target_ids_all = []
            
                if rank == 0 and self.step == finish_step:
                    infos = {
                        'epoch': epoch, 'lr': lr, 'step': self.step,
                        'save_time': datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
                    with open("{}/last.yaml".format(model_dir), 'w') as fout:
                        data = yaml.dump(infos)
                        fout.write(data)
                    save_model_path = os.path.join(model_dir, 'last.pt')
                    save_checkpoint(model, save_model_path, infos)
                    break
            
        target_ids_all = torch.cat(target_ids_all, dim=0).reshape(-1)
        logger.info('epoch {} codebook embedding usage number: {}'.format(epoch, target_ids_all.unique().numel()))
        if writer is not None and epoch==0:
            writer.add_histogram(tag='codebook embedding usage', values=target_ids_all, global_step=epoch)
        t_step = torch.tensor(self.step).to(model.device)
        dist.all_reduce(t_step, op=dist.ReduceOp.MAX)
        self.step = t_step.item()
        scheduler.set_step(self.step)

    def cv(self, model, data_loader, device, args, logger):
        ''' Cross validation on
        '''
        model.eval()
        rank = args.get('rank', 0)
        epoch = args.get('epoch', 0)
        log_interval = args.get('log_interval', 10)
        is_deepspeed = args.get('is_deepspeed', False)

        # in order to avoid division by 0
        num_seen_utts = 1
        total_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                key, feats, labels, feats_lengths, label_lengths = batch
                feats = feats.to(device)
                feats_lengths = feats_lengths.to(device)
                num_utts = feats.size(0)
                if num_utts == 0:
                    continue
                if is_deepspeed:
                    with torch.cuda.amp.autocast(
                        enabled=ds_dtype is not None,
                        dtype=ds_dtype, cache_enabled=False
                    ):
                        loss_dict, _ = model(feats, feats_lengths,
                                          self.step)
                else:
                    loss_dict, _ = model(feats, feats_lengths, self.step)
                loss = loss_dict['loss']
                if torch.isfinite(loss):
                    num_seen_utts += num_utts
                    total_loss += loss.item() * num_utts
                if batch_idx % log_interval == 0:
                    log_str = 'CV Batch {}/{} loss {:.6f} '.format(
                        epoch, batch_idx, loss.item())
                    for name, value in loss_dict.items():
                        if name != 'loss' and value is not None:
                            log_str += '{} {:.6f} '.format(
                                name,
                                value.item() if isinstance(
                                    value, torch.Tensor) else value)
                    log_str += 'history loss {:.6f}'.format(total_loss /
                                                            num_seen_utts)
                    log_str += ' rank {}'.format(rank)
                    logging.info(log_str)
                    logger.info(log_str)
                loss_dict['total_loss'] = total_loss
            model.train()
        return loss_dict, num_seen_utts
