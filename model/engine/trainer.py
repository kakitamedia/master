import time
import datetime
import os

import torch
import torch.nn as nn

import tracemalloc


def do_train(args, cfg, model, optimizer, d_optimizer, train_loader, device, summary_writer):
    max_iter = len(train_loader) - args.resume_iter
    trained_time = 0
    tic = time.time()
    end = time.time()

    logging_det_loss = [0] * len(cfg.MODEL.DOWN_RATIOS)
    logging_adv_loss = [0] * len(cfg.MODEL.DOWN_RATIOS)
    logging_contents_loss = [0] * len(cfg.MODEL.DOWN_RATIOS)
    logging_style_loss = [0] * len(cfg.MODEL.DOWN_RATIOS)
    logging_d_loss = [0] * len(cfg.MODEL.DOWN_RATIOS)
    logging_gp = 0

    print('Training Starts')
    model.train()

    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()

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

            d_loss = model(images)

            loss = 0
            if len(d_loss) == 2:
                d_loss, gp = d_loss

                gp = gp.mean()
                logging_gp += gp.item()
                loss += gp
                        
            for i in range(len(d_loss)):
                d_loss[i] = d_loss[i].mean()
                logging_d_loss[i] += d_loss[i].item()

                loss += d_loss[i] * cfg.SOLVER.DIS_ADV_LOSS_WEIGHT
            
            loss.backward()
            d_optimizer.step()

        ### Detector Training
        for _ in range(cfg.SOLVER.GEN_TRAIN_RATIO):
            if cfg.SOLVER.INIT_GEN_TRAIN < iteration <= cfg.SOLVER.INIT_DIS_TRAIN + cfg.SOLVER.INIT_GEN_TRAIN:
                break
            optimizer.zero_grad()

            det_loss, adv_loss, contents_loss, style_loss = model(images, target=targets)

            loss = 0
            for i in range(len(det_loss)):
                det_loss[i] = det_loss[i].mean()
                adv_loss[i] = adv_loss[i].mean()
                contents_loss[i] = contents_loss[i].mean()
                style_loss[i] = style_loss[i].mean()
               
                logging_det_loss[i] += det_loss[i].item() / cfg.SOLVER.GEN_TRAIN_RATIO
                logging_adv_loss[i] += adv_loss[i].item() / cfg.SOLVER.GEN_TRAIN_RATIO
                logging_contents_loss[i] += contents_loss[i].item() / cfg.SOLVER.GEN_TRAIN_RATIO
                logging_style_loss[i] += style_loss[i].item() / cfg.SOLVER.GEN_TRAIN_RATIO

                loss += det_loss[i] + adv_loss[i] * cfg.SOLVER.DET_ADV_LOSS_WEIGHT + contents_loss[i] * cfg.SOLVER.CONTENTS_LOSS_WEIGHT + style_loss[i] * cfg.SOLVER.STYLE_LOSS_WEIGHT

            loss.backward()
            optimizer.step()

        trained_time += time.time() - end
        end = time.time()

        if iteration % args.log_step == 0:
            for i in range(len(logging_det_loss)):
                logging_det_loss[i] /= args.log_step
                logging_adv_loss[i] /= args.log_step
                logging_contents_loss[i] / args.log_step
                logging_style_loss[i] / args.log_step
                logging_d_loss[i] /= args.log_step
            eta_seconds = int((trained_time / iteration) * (max_iter - iteration))
            print('===> Iter: {:07d}, LR: {:.5f}, Cost: {:.2f}s, Eta: {}, Loss: {:.6f}, D_Loss: {:.6f}, Contents_Loss: {:.6f}, Style_loss: {:.6f}'.format(iteration, optimizer.param_groups[0]['lr'], time.time() - tic, str(datetime.timedelta(seconds=eta_seconds)), sum(logging_det_loss)/len(cfg.MODEL.VALID_SCALE), sum(logging_d_loss)/len(cfg.MODEL.VALID_SCALE), sum(logging_contents_loss)/len(cfg.MODEL.VALID_SCALE), sum(logging_style_loss)/len(cfg.MODEL.VALID_SCALE)))

            if summary_writer:
                for i in range(len(cfg.MODEL.DOWN_RATIOS)):
                    summary_writer.add_scalar('train/detector/loss_{}'.format(i), logging_det_loss[i], global_step=iteration)
                    summary_writer.add_scalar('train/detector_adv_loss_{}'.format(i), logging_adv_loss[i], global_step=iteration)
                    summary_writer.add_scalar('train/contents_loss_{}'.format(i), logging_contents_loss[i], global_step=iteration)
                    summary_writer.add_scalar('train/style_loss_{}'.format(i), logging_style_loss[i], global_step=iteration)
                    summary_writer.add_scalar('train/discriminator/d_loss_{}'.format(i), logging_d_loss[i], global_step=iteration)
                summary_writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step=iteration)

            logging_det_loss = [0] * len(cfg.MODEL.DOWN_RATIOS)
            logging_adv_loss = [0] * len(cfg.MODEL.DOWN_RATIOS)
            logging_contents_loss = [0] * len(cfg.MODEL.DOWN_RATIOS)
            logging_style_loss = [0] * len(cfg.MODEL.DOWN_RATIOS)
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


        if iteration % 5000 == 0:
            snapshot2 = tracemalloc.take_snapshot()
            top_stats = snapshot2.compare_to(snapshot1, 'lineno')

            print('[Top 10 differences]')
            for stat in top_stats[:10]:
                print(stat)

            snapshot1 = snapshot2