import os
import sys
import random
import argparse
import time
from shutil import copyfile
from datetime import datetime

import numpy as np
import torch
from tensorboardX import SummaryWriter

from apsn import utils
from apsn.model import DocReaderModel

parser = argparse.ArgumentParser(
    description='Train a Document Reader model.'
)
parser = utils.add_arguments(parser)
args = parser.parse_args()
if not args.drop_nn:
    args.dropout_rate = 0.
# set model dir
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)
model_dir = os.path.abspath(model_dir)
timestamp = time.strftime("%mm%dd_%H%M%S")
print("timestamp {}".format(timestamp))
current_dir = os.path.join(args.model_dir, timestamp)
os.makedirs(current_dir)
torch.set_printoptions(precision=10)
# save model configuration
s = "\nParameters:\n"
for k in sorted(args.__dict__):
    s += "{} = {} \n".format(k.lower(), args.__dict__[k])
with open(os.path.join(args.model_dir, timestamp, "about.txt"),"w") as txtf:
    txtf.write(s); print(s)
if args.summary:
    writer = SummaryWriter(log_dir=current_dir)

# set random seed
seed = args.seed if args.seed >= 0 else int(random.random()*1000)
print ('seed:', seed)
random.seed(seed)
torch.manual_seed(seed)
if args.cuda:
    torch.cuda.manual_seed(seed)

log = utils.setup_logger(__name__, os.path.join(current_dir,args.log_file))
# batch validation size
bs_valid = 100
if args.n_actions > 8:
    bs_valid = 50

def main():
    log.info('[program starts.]')
    train, dev, dev_y, train_y, embedding, opt, q_labels, ql_mask = utils.load_data(vars(args), args)
    log.info('[Data loaded.ql_mask]')

    if args.resume:
        log.info('[loading previous model...]')
        checkpoint = torch.load(os.path.join(model_dir, args.restore_dir, args.resume))
        if args.resume_options:
            opt = checkpoint['config']
        state_dict = checkpoint['state_dict']
        model = DocReaderModel(opt, embedding, state_dict)
        epoch_0 = checkpoint['epoch'] + 1
        indices = list(range(len(train)))
        for i in range(checkpoint['epoch']):
            random.shuffle(indices) # synchronize random seed
        train = [train[i] for i in indices]
        train_y = [train_y[i] for i in indices]
        q_labels = [q_labels[i] for i in indices]
        ql_mask = [ql_mask[i] for i in indices]
        if args.reduce_lr:
            utils.lr_decay(model.optimizer, args.reduce_lr, log)
    else:
        model = DocReaderModel(opt, embedding)
        epoch_0 = 1

    train_y = np.array(train_y) # text answers for training set
    q_labels = np.array(q_labels)
    ql_mask = np.array(ql_mask)
    print("timestamp {}".format(timestamp))    

    if args.cuda:
        model.cuda()
    # evaluate pre-trained model
    if args.resume and not args.debug:
        batches = utils.BatchGen(train[:20000], batch_size=bs_valid, evaluation=True, gpu=args.cuda)
        predictions = []
        for batch in batches:
            predictions.extend(model.predict(batch)[0])
        em_t, f1_t = utils.score(predictions, train_y[:20000])
        log.info("[train EM: {0:.3f} F1: {1:3f}]".format(em_t, f1_t))

        batches = utils.BatchGen(dev, batch_size=bs_valid, evaluation=True, gpu=args.cuda)
        predictions = []
        for i, batch in enumerate(batches):
            predictions.extend(model.predict(batch)[0])
        em_v, f1_v = utils.score(predictions, dev_y)
        log.info("[dev EM: {} F1: {}]".format(em_v, f1_v))
        best_val_score = f1_v
        if args.summary:
                writer.add_scalars('accuracies', {'em_t':em_t, 'f1_t':f1_t, 'em_v':em_v, 'f1_v':f1_v}, epoch_0-1)
    else: 
        best_val_score = 0.0

    if 'const' in args.beta:
        beta = float(args.beta.split('_')[1])*0.1
    if 'const' in args.alpha:
        alpha = float(args.alpha.split('_')[1])*0.1
    scope = 'pi_q'

    dummy_r = np.zeros(args.batch_size); latent_a = None # induced interpretation
    # training
    for epoch in range(epoch_0, epoch_0 + args.epochs):
        log.warn('Epoch {} timestamp {}'.format(epoch, timestamp))
        batches = utils.BatchGen(train, batch_size=args.batch_size, gpu=args.cuda)
        start = datetime.now()
        if args.vae:
            scope = utils.select_scope_update(args, epoch-epoch_0)
        print("scope = {} beta = {} alpha = {} ".format(scope, beta, alpha))
        for i, batch in enumerate(batches):
            inds = batches.indices[i] 
            # synchronize available interpretations with the current batch
            labels = np.take(q_labels, inds, 0)
            l_mask = np.take(ql_mask, inds, 0)
            if args.vae: # VAE framework
                if args.rl_tuning == 'pgm': 
                    # policy gradient with EM scores for rewards
                    scope = 'rl'
                    truth = np.take(train_y, inds, 0)
                    pred_m, latent_a, indices = model.predict(batch)
                    _, f1_m = utils.score_em(None, pred_m, truth)
                    rewards = f1_m
                    # normalize rewards over batch
                    rewards -= rewards.mean(); rewards /= (rewards.std()+1e-08)
                elif args.rl_tuning == 'pg':  
                    # policy gradient with F1 scores for rewards
                    scope = 'rl'
                    truth = np.take(train_y, inds, 0)
                    pred_m, latent_a, indices = model.predict(batch)
                    _, f1_m = utils.score_sc(None, pred_m, truth)
                    rewards = f1_m
                    # normalize rewards over batch
                    rewards -= rewards.mean(); rewards /= (rewards.std()+1e-08)
                elif args.rl_tuning == 'sc': 
                    # reward computed by self-critic 
                    scope = 'rl'
                    truth = np.take(train_y, inds, 0)
                    pred_s, pred_m, latent_a, indices = model.predict_self_critic(batch)
                    rs, rm = utils.score_sc(pred_s, pred_m, truth)
                    rewards = rs - rm
                else:
                    rewards = dummy_r 
                model.update(batch, q_l=[labels, l_mask], r=rewards, scope=scope, beta=beta, alpha=alpha, latent_a=latent_a, span=indices)

            elif args.self_critic:
                # self-critic framework where rewards are computed as difference between the F1 score produced 
                # by the current model during greedy inference and by sampling
                truth = np.take(train_y, inds, 0)
                if args.critic_loss:
                    pred_m, latent_a, indices = model.predict(batch)
                    _, f1_m = utils.score_sc(None, pred_m, truth)
                    rewards = f1_m
                else:
                    pred_s, pred_m, latent_a, indices = model.predict_self_critic(batch)
                    rs, rm = utils.score_sc(pred_s, pred_m, truth)
                    rewards = rs - rm
                model.update(batch, r=rewards, q_l=[labels, l_mask], latent_a=latent_a)
            else:
                model.update(batch, q_l=[labels, l_mask])

            if i % args.log_per_updates == 0:
                # printing
                if args.vae:
                    log.info('updates[{0:6}] l_p[{1:.3f}] l_q[{2:.3f}] l_rl[{3:.3f}] l_ce[{4:.3f}] remaining[{5}]'.format(
                    model.updates, model.train_loss['p'].avg, model.train_loss['q'].avg, model.train_loss['rl'].avg, model.train_loss['ce'].avg,
                    str((datetime.now() - start) / (i + 1) * (len(batches) - i - 1)).split('.')[0]))
                    if args.summary:
                        writer.add_scalars('losses', {'p':model.train_loss['p'].avg, 'q':model.train_loss['q'].avg, 'ce':model.train_loss['ce'].avg, \
                                                            'rl':model.train_loss['rl'].avg}, (epoch-1)*len(batches)+i)
                else:
                    log.info('updates[{0:6}] train loss[{1:.5f}] remaining[{2}]'.format(
                    model.updates, model.train_loss.avg,
                    str((datetime.now() - start) / (i + 1) * (len(batches) - i - 1)).split('.')[0]))
                    if args.summary:
                        writer.add_scalar('loss', model.train_loss.avg, (epoch-1)*len(batches)+i)


                if scope == 'rl' and (i % 4*args.log_per_updates == 0):
                    vbatches = utils.BatchGen(dev, batch_size=bs_valid, evaluation=True, gpu=args.cuda)
                    predictions = []
                    for batch in vbatches:
                        predictions.extend(model.predict(batch)[0])
                    em_v, f1_v = utils.score(predictions, dev_y)
                    log.warn("dev EM: {0:.3f} F1: {1:3f}".format(em_v, f1_v))

        # eval
        if epoch % args.eval_per_epoch == 0:

            batches = utils.BatchGen(dev, batch_size=bs_valid, evaluation=True, gpu=args.cuda)
            predictions = []
            for batch in batches:
                predictions.extend(model.predict(batch)[0])
            em_v, f1_v = utils.score(predictions, dev_y)
            log.warn("dev EM: {0:.3f} F1: {1:3f}".format(em_v, f1_v))

            batches = utils.BatchGen(train[:20000], batch_size=bs_valid, evaluation=True, gpu=args.cuda)
            predictions = []
            for batch in batches:
                predictions.extend(model.predict(batch)[0])
            em_t, f1_t = utils.score(predictions, train_y[:20000])
            log.warn("train EM: {0:.3f} F1: {1:3f}".format(em_t, f1_t))
            
            print("current_dir {}".format(current_dir))
            
            if args.summary:
                writer.add_scalars('accuracies', {'em_t':em_t, 'f1_t':f1_t, 'em_v':em_v, 'f1_v':f1_v}, epoch)

        # save
        if not args.save_last_only or epoch == epoch_0 + args.epochs - 1:
            try:
                os.remove(os.path.join(current_dir, 'checkpoint_epoch_{}.pt'.format(epoch-1)))
            except OSError:
                pass
            model_file = os.path.join(current_dir, 'checkpoint_epoch_{}.pt'.format(epoch))
            model.save(model_file, epoch)
            if f1_v > best_val_score:
                best_val_score = f1_v
                copyfile(
                    model_file,
                    os.path.join(current_dir, 'best_model.pt'))
                log.info('[new best model saved.]')

    if args.summary:
        # export scalar data to JSON for external processing
        writer.export_scalars_to_json(os.path.join(current_dir,"all_scalars.json"))
        writer.close()



if __name__ == '__main__':
    main()
