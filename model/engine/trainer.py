import time
import os
from tqdm import tqdm
import datetime
import wandb

import torch


def do_train(args, cfg, model, optimizer, scheduler, scaler, data_loader):
    max_iter = len(data_loader['train']) + args.resume_iter
    trained_time = 0
    tic = time.time()
    end = time.time()

    logging_losses = {}

    det_params = 
    dis_params = 

    print('Training Starts!!!')
    model.train()
    for iteration, (images, targets) in enumerate(zip(data_loader['train']), args.resume_iter+1):
        det_train_flag = not (iteration > cfg.SOLVER.DETECTOR.INIT_TRAIN_ITER and iteration <= cfg.SOLVER.DETECTOR.INIT_TRAIN_ITER + cfg.SOLVER.DISCRIMINATOR.INIT_TRAIN_ITER)
        dis_train_flag = iteration > cfg.SOLVER.DETECTOR.INIT_TRAIN_ITER

        for factor, (image, target) in enumerate(zip(images, targets)):
            optimizer.zero_grad()
            det_loss, dis_loss, loss_dict = model(image, target, factor)
            det_loss, dis_loss = det_loss.mean(), dis_loss.mean()

            optimizer.zero_grad()
            if det_train_flag:
                scaler.scale(det_loss).backward(retain_graph=dis_train_flag, inputs=det_params)
                if cfg.SOLVER.DETECTOR.GRADIENT_CLIP > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(det_params, cfg.SOLVER.DETECTOR.GRADIENT_CLIP)

            if dis_train_flag:
                scaler.scale(dis_loss).backward(inputs=dis_params)
                if cfg.SOLVER.DISCRIMINATOR.GRADIENT_CLIP > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(dis_params, cfg.SOLVER.DISCRIMINATOR.GRADIENT_CLIP)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            for k in loss_dict.keys():
                if k in logging_losses:
                    logging_losses[k] += loss_dict[k].item()
                else:
                    logging_losses[k] = loss_dict[k].item()

            trained_time += time.time() - end
            end = time.time()

            if iteration % args.log_step == 0:
                eta_seconds = int((trained_time / iteration) * (max_iter - iteration))
                logging_loss /= args.log_step
                print('===> Iter: {:07d}, LR: {:.06f}, Cost: {:2f}s, Eta: {}, Loss: {:.6f}'.format(iteration, optimizer.param_groups[0]['lr'], time.time() - tic, str(datetime.timedelta(seconds=eta_seconds)), logging_loss))

                if not args.debug:
                    log_dict = {'train/loss': logging_loss, 'train/lr': optimizer.param_groups[0]['lr']}
                    wandb.log(log_dict, step=iteration)

                logging_loss = 0

                tic = time.time()

            if iteration % args.save_step == 0 and not args.debug:
                model_path = os.path.join(cfg.OUTPUT_DIR, 'model', 'iteration_{}.pth'.format(iteration))
                optimizer_path = os.path.join(cfg.OUTPUT_DIR, 'optimizer', 'iteration_{}.pth'.format(iteration))
                scheduler_path = os.path.join(cfg.OUTPUT_DIR, 'scheduler', 'iteration_{}.pth'.format(iteration))
                scaler_path = os.path.join(cfg.OUTPUT_DIR, 'scaler', 'iteration_{}.pth'.format(iteration))

                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                os.makedirs(os.path.dirname(optimizer_path), exist_ok=True)
                os.makedirs(os.path.dirname(scaler_path), exist_ok=True)

                if args.num_gpus > 1:
                    torch.save(model.module.state_dict(), model_path)
                else:
                    torch.save(model.state_dict(), model_path)

                torch.save(optimizer.state_dict(), optimizer_path)
                torch.save(scheduler.state_dict(), scheduler_path)
                torch.save(scaler.state_dict(), scaler_path)

                print('=====> Save Checkpoint to {}'.format(model_path))

            if 'val' in data_loader.keys() and iteration % args.eval_step == 0:
                print('Validating...')
                model.eval()
                val_loss = 0
                for _ in tqdm(data_loader['val']):
                    with torch.inference_mode():
                        loss = model()
                        val_loss += loss.item()

                val_loss /= len(data_loader['val'])

                validation_time = time.time() - end
                trained_time += validation_time
                end = time.time()
                tic = time.time()
                print('======> Cost: {:2f}s, Loss: {:.06f}'.format(validation_time, val_loss))

                if not args.debug:
                    log_dict = {'val/loss': val_loss}
                    wandb.log(log_dict, step=iteration)

                model.train()
