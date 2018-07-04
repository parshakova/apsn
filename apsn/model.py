# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import numpy as np
import logging
from . import layers

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

from .utils import AverageMeter
from .rnn_reader import RnnDocReader

# Modification:
#   - embedding APSN module into DrQA
#   - variatinal inference network optimization
#   - discrete VAE semi-supervised framework
#   - mixed cross-entropy + policy gradietn objective
#   - interpretation diversity objective
# Origin: https://github.com/taolei87/sru/blob/master/DrQA/drqa/model.py

logger = logging.getLogger(__name__)


class DocReaderModel(object):

    def __init__(self, opt, embedding=None, state_dict=None):
        # Book-keeping.
        self.opt = opt
        self.updates = state_dict['updates'] if state_dict else 0
        self.train_loss = {'ce':AverageMeter(),'rl':AverageMeter(),'q':AverageMeter(),'p':AverageMeter()}

        # Building network.
        self.network = RnnDocReader(opt, embedding=embedding)

        if 'gram' in opt['control_d']:
            # use central pre-trained SRU weights from the first layer as a reference
            # for interpretation diversity objective
            self.sru_w = Variable(state_dict['network']['doc_rnn.rnns.0.weight'], requires_grad=False)
        
        if state_dict:
            # restore parameters
            model_dict = self.network.state_dict()
            pretrained_dict = {k: v for k, v in state_dict['network'].items() if (k in model_dict) and (v.size() == model_dict[k].size())}

            if self.opt['fin_att']=='param' and self.opt['restore_dir']=='original_0':
                # specific conditions for model with Bilinear sequence attention (for every actions) 
                # at final layer when restoring from original bilinear attention layer
                old_m = ['start_attn.linear.weight','start_attn.linear.bias','end_attn.linear.weight','end_attn.linear.bias']
                new_m = ['start_attn.weight','start_attn.bias','end_attn.weight','end_attn.bias']
                fin_att = []
                for k in old_m:
                    w = state_dict['network'][k]
                    if (w.size()) == 1:
                        w = w.unsqueeze(0).expand(self.opt['n_actions'], w.size(0))
                    elif (w.size()) == 2:
                        w = w.unsqueeze(0).expand(self.opt['n_actions'], w.size(0), w.size(1))
                    fin_att += [w.contiguous()]
                pretrained_dict.update({k:v for k,v in zip(new_m, fin_att)})

            if 'q' in self.opt['pi_q_rnn']:
                # update parameters phi if exists new SRU for posterior q network
                pretrained_dict_q = {'q_'+k: v for k, v in state_dict['network'].items() if "q_"+k in model_dict and (v.size() == model_dict['q_'+k].size())}
                pretrained_dict.update(pretrained_dict_q)

            model_dict.update(pretrained_dict)
            self.network.load_state_dict(model_dict)

        # Building optimizer.
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        
        if opt['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(parameters, opt['learning_rate'],
                                       momentum=opt['momentum'],
                                       weight_decay=opt['weight_decay'])
        elif opt['optimizer'] == 'adamax':
            self.optimizer = optim.Adamax(parameters, opt['learning_rate'],
                                          weight_decay=opt['weight_decay'])
        elif opt['optimizer'] == 'adam':
            self.optimizer = optim.Adam(parameters, opt['learning_rate'],
                                          weight_decay=opt['weight_decay'])
        else:
            raise RuntimeError('Unsupported optimizer: %s' % opt['optimizer'])

        if state_dict:
            if self.opt['cuda']:
                optim_state = [self.optimizer.state.values()]
                for o_state in optim_state:
                    for state in o_state:
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.cuda()

        if 'fix' in opt['control_d'] and opt['restore_dir']:
            print("\nfix doc_sru[0] weights \n")
            self.network.doc_rnn.rnns[0].weight.requires_grad = False

        num_params = sum(p.data.numel() for p in parameters
            if p.data.data_ptr() != self.network.embedding.weight.data.data_ptr())
        print ("{} parameters".format(num_params))


    def update(self, ex, q_l=[None,None], r=None, scope="", beta=0.1, alpha=0.1, latent_a=None, span=None):
        # Train mode
        self.network.train()
        if self.opt['cuda']:
            inputs = [Variable(e.cuda(async=True)) for e in ex[:9]]
            target_s = Variable(ex[7].cuda(async=True))
            target_e = Variable(ex[8].cuda(async=True))
        else:
            inputs = [Variable(e) for e in ex[:9]]
            
        # labels for interpretations during semi-supervised training
        [labels, l_mask] = q_l
        labels = Variable(torch.from_numpy(labels)).cuda(); l_mask = Variable(torch.from_numpy(l_mask)).cuda()

        vf_coef = 0.5; ent_coef = self.opt['entropy_loss']

        # m_inp = [x1, x1_f, x1_pos, x1_ner, x1_mask, x2, x2_mask, s_ids, e_ids, scope, labels, l_mask, latent_a]
        m_inp = inputs + [scope, labels, l_mask, latent_a]
        span_vars, scrit_vars, vae_vars = self.network(tuple(m_inp))
        score_s, score_e, crit_f1, drnn_input, doc_hiddens_init, question_hidden = span_vars

        # Compute loss and accuracies
        loss_ce = F.nll_loss(score_s, target_s) + F.nll_loss(score_e, target_e) 

        if  self.opt['vae']:

            kl_loss, r_kl, q_logp_t, p_logp_t, ent_loss, actions, computed_a = vae_vars
            l_mask = l_mask.float()

            if scope  == 'pi_q': 
                # ----------------------- Semi-supervised VAE framework ---------------------------

                loss_prior_s = -alpha*kl_loss
                loss_prior_t = -p_logp_t
                loss_prior = ((1-l_mask)*loss_prior_s + l_mask*loss_prior_t).mean()

                r_ids = (-F.nll_loss(score_s, target_s, reduce=False) -F.nll_loss(score_e, target_e, reduce=False))/2
                r_lvb0 = r_ids - beta*r_kl
                r_lvb = (r_lvb0 - crit_f1).detach()

                loss_cr_s = F.mse_loss(crit_f1, r_lvb0.detach(), reduce=False)
                loss_cr = ((1-l_mask)*loss_cr_s).mean()

                loss_posterior_s = -alpha*r_lvb*q_logp_t
                loss_posterior_t = -q_logp_t
                loss_posterior = ((1-l_mask)*loss_posterior_s + l_mask*loss_posterior_t).mean()

                loss_vae = loss_prior + loss_posterior + loss_ce + loss_cr - ent_coef*ent_loss
            elif scope == 'rl':
                #  -------------------- Mixed CE + Policy Gradient framework -------------------------

                advantage = Variable(torch.from_numpy(r).cuda(), requires_grad=False).float()

                sids, eids = map(np.array,zip(*span))
                sids = Variable(torch.from_numpy(sids).long()).cuda()
                eids = Variable(torch.from_numpy(eids).long()).cuda()
                logp = p_logp_t
                for logps, a_i in zip([score_s, score_e], [sids, eids]):
                    logp_i = -F.nll_loss(logps, a_i)
                    # collective log probabilities for sampled span indices and interpretation
                    logp = logp + logp_i

                loss_rl = (-advantage*logp).mean()
                loss_vae = self.opt['ce_frac']*loss_ce + (1-self.opt['ce_frac'])*loss_rl

                ce_grads = torch.autograd.grad(loss_vae, self.network.parameters(), create_graph=True, allow_unused=True)
                    
            loss = loss_vae

            # ------------------------- Interpretation Diversity Objective -------------------------

            if 'gram' in self.opt['control_d']:
                loss_g = 0; weight = self.network.doc_rnn.rnns[0].weight
                # compute reference weights/activations for distance maximization: e.g. style, cos similarity
                if 'attn' in self.opt['control_d']:
                    ref = self.sru_w
                elif 'conv' in self.opt['control_d']:
                    x = drnn_input.mean(1)
                    length = x.size(0) if x.dim() == 3 else 1; batch = x.size(-2)
                    u = []
                    ref_u = x.mm(self.sru_w).detach()
                    ref_u.requires_grad = False
                    for a in range(self.opt['n_actions']):
                        w_i = self.network.doc_rnn.rnns[0].func(weight, a)
                        u_i = x.mm(w_i).view(batch, -1)
                        u.append(u_i)
                    ref = torch.stack(u, 0).mean(0).detach()
                    ref.requires_grad = False
                elif 'uadd' in self.opt['control_d']:
                    x = drnn_input.mean(1)
                    length = x.size(0) if x.dim() == 3 else 1; batch = x.size(-2)
                    u = []
                    ref_u = x.mm(self.sru_w).detach()
                    ref_u.requires_grad = False
                    for a in range(self.opt['n_actions']):
                        w_i = self.network.doc_rnn.rnns[0].func(weight, self.network.doc_rnn.rnns[0].wa[a])
                        u_i = x.mm(w_i).view(batch, -1)
                        u.append(u_i)
                    ref = torch.stack(u, 0).mean(0).detach()
                    ref.requires_grad = False
                else:
                    ref = self.network.doc_rnn.rnns[0].wa.mean(0).detach()
                    ref.requires_grad = False
                    ref_u = ref
                
                if 'gram_sc' in self.opt['control_d']:
                    if 'conv'in self.opt['control_d'] or 'uadd' in self.opt['control_d']:
                        gram = layers.GramMatrix_u()
                    else:
                        gram = layers.GramMatrix()
                    gram_s = layers.StyleLoss(gram(ref_u), 1000, u='conv'in self.opt['control_d'] or 'uadd' in self.opt['control_d']).cuda()
                    gram_c = layers.ContentLoss(ref_u, 1).cuda()
                    gram_f = lambda a: -0.1*gram_s(a) + 0.5*gram_c(a)
                elif 'gram_c' in self.opt['control_d']: 
                    gram_c =  layers.ContentLoss(ref_u, 1).cuda()
                    gram_f = lambda a: 0.5*gram_c(a)
                elif 'gram_s' in self.opt['control_d']:
                    gram = layers.GramMatrix()
                    gram_s = layers.StyleLoss(gram(ref_u), 1000).cuda()
                    gram_f = lambda a: -0.1*gram_s(a)
                else:
                    gram_f = lambda a: 0

                if 'cos' in self.opt['control_d']:
                    gram_cos = lambda a: 0.1*F.cosine_similarity(ref_u,a).mean()
                else:
                    gram_cos = lambda a: 0

                for a in range(self.opt['n_actions']):
                    if 'attn_eh' in self.opt['control_d']:
                        w_i = self.network.doc_rnn.rnns[0].func(self.network.doc_rnn.rnns[0].weight, self.network.doc_rnn.rnns[0].wa_e[a], self.network.doc_rnn.rnns[0].wa_h[a])
                        loss_g = loss_g + gram_f(w_i) 
                    elif 'transf' in self.opt['control_d']:
                        w_i = self.network.doc_rnn.rnns[0].wa[a]
                        loss_g = loss_g + gram_f(w_i) 
                    elif 'attn' in self.opt['control_d']:
                        w_i = self.network.doc_rnn.rnns[0].func(weight, self.network.doc_rnn.rnns[0].wa[a])
                        loss_g = loss_g + gram_f(w_i)
                    elif 'uadd' in self.opt['control_d']:
                        w_i =  u[a]
                        loss_g = loss_g + 0.01*gram_f(w_i)
                    elif 'add' in self.opt['control_d'] or 'mul' in self.opt['control_d']:
                        w_i =  self.network.doc_rnn.rnns[0].wa[a]
                        loss_g = loss_g + 0.01*gram_f(w_i) + 0.0001 * self.network.doc_rnn.rnns[0].wa[a].norm()
                    elif 'conv' in self.opt['control_d']:
                        w_i = u[a]
                        loss_g = loss_g + 0.01*(gram_f(w_i) + gram_cos(w_i))

                loss = loss + loss_g/self.opt['n_actions']
            else:
                loss_g = 0
                for a in range(self.opt['n_actions']):
                    if 'add' in self.opt['control_d'] or 'mul' in self.opt['control_d']:
                        w_i =  self.network.doc_rnn.rnns[0].func(weight, self.network.doc_rnn.rnns[0].wa[a])
                        loss_g = loss_g + 0.0001 * self.network.doc_rnn.rnns[0].wa[a].norm()

                loss = loss + loss_g

        elif self.opt['self_critic']:
            advantage = Variable(torch.from_numpy(r).cuda(), requires_grad=False).float()
            s_sma, e_sma, s_mxa, e_mxa, s_logp, e_logp = scrit_vars
            if self.opt['critic_loss']:
                advantage = (advantage - crit_f1).detach()
                loss_cr = F.mse_loss(crit_f1, r)
            loss_sc = (-advantage*(s_logp + e_logp)).mean()
            if self.opt['critic_loss']:
                loss_sc = (loss_sc+loss_cr)*0.5             
            loss = self.opt['ce_frac']*loss_ce + (1-self.opt['ce_frac'])*loss_sc

        else:
            # original model
            loss = loss_ce


        if self.opt['vae']: 
            if scope == 'rl': 
                self.train_loss[scope].update(loss_rl.data[0], ex[0].size(0))
            else:
                self.train_loss['p'].update(loss_prior.data[0], ex[0].size(0))
                self.train_loss['q'].update(loss_posterior.data[0], ex[0].size(0))
            self.train_loss['ce'].update(loss_ce.data[0], ex[0].size(0))
        elif self.opt['relax']: 
            self.train_loss['q'].update(vf_loss.data[0], ex[0].size(0))
            self.train_loss['p'].update(pg_loss.mean().data[0], ex[0].size(0))
            self.train_loss['ce'].update(loss_ce.data[0], ex[0].size(0))
        else:
            self.train_loss.update(loss.data[0], ex[0].size(0))
            
        # Clear gradients and run backward
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients
        torch.nn.utils.clip_grad_norm(self.network.parameters(), self.opt['grad_clipping'])
        # Update parameters
        self.optimizer.step()
        self.updates += 1

        if np.random.rand() < self.opt['grad_prob_print']:
            print("#### scope = {} ####".format(scope))
            for n, par in list(self.network.named_parameters()):
                try:
                    dummy = not(F.l1_loss(par.grad.cpu(), Variable(torch.zeros(par.grad.size()))).data.numpy() == 0 \
                                                            and par.grad.sum().data.numpy() == 0)
                    if dummy:
                        print(n, par.size())
                except:
                    pass

        # Reset any partially fixed parameters (e.g. rare words)
        self.reset_parameters()

    def predict_self_critic(self, ex):
        # self_critic mode where rewards are computed as difference between the F1 score produced 
        # by the current model during greedy inference and by sampling
        self.network.eval()
        if self.opt['cuda']:
            inputs = [Variable(e.cuda(async=True), volatile=True)
                      for e in ex[:9]]
        else:
            inputs = [Variable(e, volatile=True) for e in ex[:9]]

        m_inp = inputs + ['rl', None, None, None]
        span_vars, scrit_vars, vae_vars = self.network(m_inp)
        score_s, score_e, crit_f1, drnn_input, doc_hiddens_init, question_hidden = span_vars
        s_sma, e_sma, s_mxa, e_mxa, s_logp, e_logp = scrit_vars
        kl_loss, r_kl, q_logp_t, p_logp_t, ent_loss, actions, computed_a = vae_vars

        # Transfer to CPU/normal tensors for numpy ops
        s_sma = s_sma.data.cpu().numpy()
        s_mxa = s_mxa.data.cpu().numpy()
        e_sma = e_sma.data.cpu().numpy()
        e_mxa = e_mxa.data.cpu().numpy()
        x_len = inputs[4].data.sum(-1).cpu().numpy()

        # Get argmax text spans
        text = ex[-2]
        spans = ex[-1]
        pred_s, pred_m = [], []
        indices = []
        for i in range(s_sma.shape[0]):
            s_idx, e_idx = s_sma[i], e_sma[i]
            s_idx = np.clip(s_idx, 0, max(0,x_len[i]-1))
            e_idx = np.clip(e_idx, 0, max(0,x_len[i]-1))
            indices.append([s_idx, e_idx])
            s_offset, e_offset = spans[i][s_idx][0], spans[i][e_idx][1]

            pred_s.append(text[i][s_offset:e_offset])

            s_idx, e_idx = s_mxa[i], e_mxa[i]
            s_offset, e_offset = spans[i][s_idx][0], spans[i][e_idx][1]
            pred_m.append(text[i][s_offset:e_offset])

        return pred_s, pred_m, actions, indices

    def predict(self, ex):
        # sample start and end indices for obtaining span 
        self.network.eval()
        if self.opt['cuda']:
            inputs = [Variable(e.cuda(async=True), volatile=True)
                      for e in ex[:7]]
        else:
            inputs = [Variable(e, volatile=True) for e in ex[:7]]
 
        # x1, x1_f, x1_pos, x1_ner, x1_mask, x2, x2_mask, s_ids, e_ids, scope, labels, l_mask, latent_a
        m_inp = inputs + [None, None, 'rl', None, None, None]
        span_vars, scrit_vars, vae_vars = self.network(m_inp) 
        score_s, score_e, crit_f1, drnn_input, doc_hiddens_init, question_hidden = span_vars
        kl_loss, r_kl, q_logp_t, p_logp_t, ent_loss, actions, computed_a = vae_vars

        # Transfer to CPU/normal tensors for numpy ops
        score_s = score_s.data.cpu()
        score_e = score_e.data.cpu()

        # Get argmax text spans
        text = ex[-2]
        spans = ex[-1]
        predictions = []
        indices = []
        max_len = self.opt['max_len'] or score_s.size(1)
        for i in range(score_s.size(0)):
            scores = torch.ger(score_s[i], score_e[i])
            scores.triu_().tril_(max_len - 1)
            scores = scores.numpy()
            s_idx, e_idx = np.unravel_index(np.argmax(scores), scores.shape)
            indices.append([s_idx, e_idx])
            s_offset, e_offset = spans[i][s_idx][0], spans[i][e_idx][1]
            predictions.append(text[i][s_offset:e_offset])
        return predictions, actions, indices

    def get_embeddings(self, ex, latent_a=[0]):
        # get embeddings for visualization
        self.network.eval()
        if self.opt['cuda']:
            inputs = [Variable(e.cuda(async=True), volatile=True)
                      for e in ex[:7]]
        else:
            inputs = [Variable(e, volatile=True) for e in ex[:7]]

        # x1, x1_f, x1_pos, x1_ner, x1_mask, x2, x2_mask, s_ids, e_ids, scope, labels, l_mask, latent_a
        if latent_a[0]>0:
            m_inp = inputs + [None, None, 'rl', None, None, latent_a[1]]
        else:
            m_inp = inputs + [None, None, 'rl', None, None, None]
        span_vars, scrit_vars, vae_vars = self.network(m_inp) 
        score_s, score_e, crit_f1, drnn_input, doc_hiddens_init, question_hidden = span_vars
        kl_loss, r_kl, q_logp_t, p_logp_t, ent_loss, actions, computed_a = vae_vars

        embeds = torch.cat((doc_hiddens_init[:,-1,:doc_hiddens_init.size(2)//2], doc_hiddens_init[:,0,doc_hiddens_init.size(2)//2:]), 1)
        return embeds.cpu().numpy(), actions.cpu().numpy(), question_hidden.cpu().numpy(), computed_a.cpu().numpy()

    def predict_inter(self, ex, latent_a=None):
        # span prediction during the induced interpretation mode
        self.network.eval()
        if self.opt['cuda']:
            inputs = [Variable(e.cuda(async=True), volatile=True)
                      for e in ex[:7]]
        else:
            inputs = [Variable(e, volatile=True) for e in ex[:7]]

        # m_inp = [x1, x1_f, x1_pos, x1_ner, x1_mask, x2, x2_mask, s_ids, e_ids, scope, labels, l_mask, latent_a]
        m_inp = inputs + [None, None, 'rl', None, None, latent_a]
        
        span_vars, scrit_vars, vae_vars = self.network(m_inp) 
        score_s, score_e, crit_f1, drnn_input, doc_hiddens_init, question_hidden = span_vars
        kl_loss, r_kl, q_logp_t, p_logp_t, ent_loss, actions, computed_a = vae_vars


        score_s = score_s.data.cpu()
        score_e = score_e.data.cpu()
        # Get argmax text spans
        text = ex[-2]
        spans = ex[-1]
        predictions = []
        max_len = self.opt['max_len'] or score_s.size(1)
        for i in range(score_s.size(0)):
            scores = torch.ger(score_s[i], score_e[i])
            scores.triu_().tril_(max_len - 1)
            scores = scores.numpy()
            s_idx, e_idx = np.unravel_index(np.argmax(scores), scores.shape)
            s_offset, e_offset = spans[i][s_idx][0], spans[i][e_idx][1]
            predictions.append(text[i][s_offset:e_offset])
        return predictions, actions

    def reset_parameters(self):
        # Reset fixed embeddings to original value
        if self.opt['tune_partial'] > 0:
            offset = self.opt['tune_partial'] + 2
            if offset < self.network.embedding.weight.data.size(0):
                self.network.embedding.weight.data[offset:] \
                    = self.network.fixed_embedding

    def save(self, filename, epoch):
        params = {
            'state_dict': {
                'network': self.network.state_dict(),
                #'optimizer': self.optimizer.state_dict(),
                'updates': self.updates
            },
            'config': self.opt,
            'epoch': epoch
        }
        try:
            torch.save(params, filename)
            logger.info('model saved to {}'.format(filename))
        except BaseException:
            logger.warn('[ WARN: Saving failed... continuing anyway. ]')

    def cuda(self):
        self.network.cuda()
