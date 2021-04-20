import time
import datetime
import os

import torch
import torch.nn as nn


def do_train(args, cfg, model, optimizer, d_optimizer, train_loader, device, summary_writer):
    max_iter = len(train_loader) - args.resume_iter
    trained_time = 0
    tic = time.time()
    end = time.time()

    adv_weight = [cfg.SOLVER.ADV_LOSS_INIT_WEIGHT] * len(cfg.MODEL.DOWN_RATIOS)

    pid_adv = [0] * len(cfg.MODEL.DOWN_RATIOS)
    pid_d = [0] * len(cfg.MODEL.DOWN_RATIOS)

    logging_det_loss = [0] * len(cfg.MODEL.DOWN_RATIOS)
    logging_adv_loss = [0] * len(cfg.MODEL.DOWN_RATIOS)
    logging_d_loss = [0] * len(cfg.MODEL.DOWN_RATIOS)

    print('Training Starts')
    model.train()
    for iteration, (images, targets) in enumerate(train_loader, args.resume_iter+1):
        ### Discriminator Training
        for _ in range(cfg.SOLVER.DIS_TRAIN_RATIO):
            if iteration <= cfg.SOLVER.INIT_GEN_TRAIN:
                break
            d_optimizer.zero_grad()
            # for i in cfg.MODEL.VALID_SCALE:
            #     d_loss = model(images[i], i)
            #     d_loss = d_loss.mean()
            #     logging_d_loss[i] += d_loss.item() / cfg.SOLVER.DIS_TRAIN_RATIO

            #     pid_d[i] = d_loss.item()

            #     d_loss.backward()

            d_loss = model(images, 0)
            d_loss = d_loss.mean()
            logging_d_loss[0] = d_loss.item()

            d_loss.backward()

            d_optimizer.step()

        ### Detector Training
        for _ in range(cfg.SOLVER.GEN_TRAIN_RATIO):
            if cfg.SOLVER.INIT_GEN_TRAIN < iteration <= cfg.SOLVER.INIT_DIS_TRAIN + cfg.SOLVER.INIT_GEN_TRAIN:
                break
            optimizer.zero_grad()
            for i in cfg.MODEL.VALID_SCALE:
                det_loss, adv_loss = model(images[i], i, target=targets[i])
                det_loss, adv_loss = det_loss.mean(), adv_loss.mean()

                logging_det_loss[i] += det_loss.item() / cfg.SOLVER.GEN_TRAIN_RATIO
                logging_adv_loss[i] += adv_loss.item() / cfg.SOLVER.GEN_TRAIN_RATIO

                pid_adv[i] = adv_loss.item()

                if iteration <= cfg.SOLVER.INIT_DIS_TRAIN + cfg.SOLVER.INIT_GEN_TRAIN:
                    loss = det_loss
                else:
                    loss = det_loss + adv_weight[i] * adv_loss
                loss.backward()

            optimizer.step()

        ### update adv_weight
        for i in cfg.MODEL.VALID_SCALE:
            if iteration <= cfg.SOLVER.INIT_DIS_TRAIN:
                break
            adv_weight[i] += cfg.SOLVER.ADV_LOSS_GAIN * (pid_adv[i] - pid_d[i])
            if adv_weight[i] < 0:
                adv_weight[i] = 0

        trained_time += time.time() - end
        end = time.time()

        if iteration % args.log_step == 0:
            for i in range(len(logging_det_loss)):
                logging_det_loss[i] /= args.log_step
                logging_adv_loss[i] /= args.log_step
                logging_d_loss[i] /= args.log_step
            eta_seconds = int((trained_time / iteration) * (max_iter - iteration))
            print('===> Iter: {:07d}, LR: {:.5f}, Cost: {:.2f}s, Eta: {}, Loss: {:.6f}, D_Loss: {:.6f}, adv_weight: {:.6f}'.format(iteration, optimizer.param_groups[0]['lr'], time.time() - tic, str(datetime.timedelta(seconds=eta_seconds)), sum(logging_det_loss)/len(cfg.MODEL.VALID_SCALE), sum(logging_d_loss)/len(cfg.MODEL.VALID_SCALE), sum(adv_weight)/len(cfg.MODEL.VALID_SCALE)))

            if summary_writer:
                for i in range(len(cfg.MODEL.DOWN_RATIOS)):
                    summary_writer.add_scalar('train/detector/loss_{}'.format(i), logging_det_loss[i], global_step=iteration)
                    summary_writer.add_scalar('train/detector_adv_loss_{}'.format(i), logging_adv_loss[i], global_step=iteration)
                    summary_writer.add_scalar('train/discriminator/d_loss_{}'.format(i), logging_d_loss[i], global_step=iteration)
                    summary_writer.add_scalar('train/adv_weight_{}'.format(i), adv_weight[i], global_step=iteration)
                summary_writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step=iteration)

            logging_det_loss = [0] * len(cfg.MODEL.DOWN_RATIOS)
            logging_adv_loss = [0] * len(cfg.MODEL.DOWN_RATIOS)
            logging_d_loss = [0] * len(cfg.MODEL.DOWN_RATIOS)
            tic = time.time()

        if iteration % args.save_step == 0 and not args.debug:
            extractor_path = os.path.join(cfg.OUTPUT_DIR, 'extractor', 'iteration_{}.pth'.format(iteration))
            detector_path = os.path.join(cfg.OUTPUT_DIR, 'detector', 'iteration_{}.pth'.format(iteration))
            optimizer_path = os.path.join(cfg.OUTPUT_DIR, 'optimizer', 'iteration_{}.pth'.format(iteration))
            d_model_path = os.path.join(cfg.OUTPUT_DIR, 'discriminator', 'iteration_{}.pth'.format(iteration))
            d_optimizer_path = os.path.join(cfg.OUTPUT_DIR, 'd_optimizer', 'iteration_{}.pth'.format(iteration))

            os.makedirs(os.path.dirname(extractor_path), exist_ok=True)
            os.makedirs(os.path.dirname(detector_path), exist_ok=True)
            os.makedirs(os.path.dirname(optimizer_path), exist_ok=True)
            if cfg.SOLVER.DIS_TRAIN_RATIO > 0:
                os.makedirs(os.path.dirname(d_model_path), exist_ok=True)
                os.makedirs(os.path.dirname(d_optimizer_path), exist_ok=True)

            if args.num_gpus > 1:
                torch.save(model.module.extractors.state_dict(), extractor_path)
                torch.save(model.module.detector.state_dict(), detector_path)
                if cfg.SOLVER.DIS_TRAIN_RATIO > 0:
                    torch.save(model.module.discriminator.state_dict(), d_model_path)
            else:
                torch.save(model.extractor.state_dict(), extractor_path)
                torch.save(model.detector.state_dict(), detector_path)
                if cfg.SOLVER.DIS_TRAIN_RATIO > 0:
                    torch.save(model.discriminator.state_dict(), d_model_path)

            torch.save(optimizer.state_dict(), optimizer_path)
            if cfg.SOLVER.DIS_TRAIN_RATIO > 0:
                torch.save(d_optimizer.state_dict(), d_optimizer_path)

            print('===== Save Checkpoint to {}'.format(detector_path))


        if iteration % cfg.SOLVER.LR_DACAY == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10.0
            for param_group in d_optimizer.param_groups:
                param_group['lr'] /= 10.0