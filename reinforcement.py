# coding=utf-8

import contextlib
import logging
import os
import time
from shutil import copyfile
from typing import Callable

import numpy as np
import torch

import models.reinvent
from scaffold.ScaffoldFilter import ScaffoldFilter
from utils import unique, Variable, fraction_valid_smiles
from utils.experience import Experience


def reinforcement_learning(prior: models.reinvent.Model, agent: models.reinvent.Model, scoring_function: Callable,
                           scaffoldfilter: ScaffoldFilter,
                           logdir: str, resultdir: str,
                           n_steps=3000, sigma=120, experience_replay=False,
                           lr=0.0001, batch_size=128,
                           save_every=50, keep_max=10, reset=0,
                           temperature=1.0, reset_score_cutoff=0.5):
    assert prior.voc == agent.voc, "The agent and the prior must have the same vocabulary!"
    start_time = time.time()

    # We don't need gradients with respect to Prior
    for param in prior.rnn.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(agent.rnn.parameters(), lr=lr)

    # For policy based RL, we normally train on-policy and correct for the fact that more likely actions
    # occur more often (which means the agent can get biased towards them). Using experience replay is
    # therefor not as theoretically sound as it is for value based RL, but it seems to work well.
    if experience_replay:
        experience = Experience(prior.voc)
        # Can add an initial experience if we want to
        # experience.initiate_from_file('/home/excape/reinvent/tala_xray_lig.smi', scoring_function, Prior)

    reset_countdown = 0

    for step in range(n_steps):

        # Sample from Agent
        seqs, agent_likelihood, entropy = agent.sample(batch_size, temperature=temperature)

        # Remove duplicates, ie only consider unique seqs
        unique_idxs = unique(seqs)
        seqs = seqs[unique_idxs]
        agent_likelihood = agent_likelihood[unique_idxs]
        entropy = entropy[unique_idxs]

        # Get prior likelihood and score
        prior_likelihood, _ = prior.likelihood(Variable(seqs), temperature=temperature)
        smiles = prior.sequence_to_smiles(seqs)
        score_components = scoring_function(smiles)
        score_components["step"] = [step]*len(smiles)
        if scaffoldfilter:
            score = scaffoldfilter.score(smiles, score_components)
        else:
            # we need to extract the total_score key
            score = score_components.pop('total_score')
        # Calculate augmented likelihood
        augmented_likelihood = prior_likelihood + sigma * Variable(score)
        loss = torch.pow((augmented_likelihood - agent_likelihood), 2)

        # Experience Replay
        # First sample
        if experience_replay and len(experience) > 4:
            exp_seqs, exp_score, exp_prior_likelihood = experience.sample(8)
            exp_agent_likelihood, exp_entropy = agent.likelihood(exp_seqs.long(), temperature=temperature)
            exp_augmented_likelihood = exp_prior_likelihood + sigma * exp_score
            exp_loss = torch.pow((Variable(exp_augmented_likelihood) - exp_agent_likelihood), 2)
            loss = torch.cat((loss, exp_loss), 0)
            agent_likelihood = torch.cat((agent_likelihood, exp_agent_likelihood), 0)

        # Then add new experience
        prior_likelihood = prior_likelihood.data.cpu().numpy()
        new_experience = zip(smiles, score, prior_likelihood)
        if experience_replay:
            experience.add_experience(new_experience)

        # Calculate loss
        loss = loss.mean()

        # Add regularizer that penalizes high likelihood for the entire sequence
        # With this regularizer the example where only Celecoxib is generated
        # doesnt work for obvious reasons.
        # loss_p = - (1 / agent_likelihood).mean()
        # loss += 5 * 1e3 * loss_p

        # Calculate gradients and make an update to the network weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Now optimize with respect to the entropy
        entropy = torch.sum(entropy)
        # loss.backward()
        # optimizer.step()

        # Convert to numpy arrays so that we can print them
        augmented_likelihood = augmented_likelihood.data.cpu().numpy()
        agent_likelihood = agent_likelihood.data.cpu().numpy()

        # Print some information for this step
        time_elapsed = int(time.time() - start_time)
        time_left = (time_elapsed * ((n_steps - step) / (step + 1)))
        mean_score = np.mean(score)
        message = ("\n Step {}   Fraction valid SMILES: {:4.1f}   Score: {:.4f}   Time elapsed: {}   "
                   "Time left: {:.1f}\n").format(step, fraction_valid_smiles(smiles) * 100, mean_score,
                                                 time_elapsed, time_left)
        message += "     ".join(["  Agent", "Prior", "Target", "Score"] + list(score_components.keys()) + ["SMILES\n"])
        for i in range(min(10, len(smiles))):
            print_component_scores = [score_components[key][i] for key in score_components]
            message += "  {:6.2f}    {:6.2f}    {:6.2f}    {:6.2f}    ".format(agent_likelihood[i],
                                                                               prior_likelihood[i],
                                                                               augmented_likelihood[i],
                                                                               score[i])
            message += ("{:6.2f}    " * len(print_component_scores)).format(*print_component_scores)
            message += "{}\n".format(smiles[i])

        logging.info(message)
        if step % save_every == 0:
            logging.info("Write scaffold memory")
            if scaffoldfilter:
                #scaffoldfilter.savetojson(os.path.join(logdir, "scaffold_memory.{}.json".format(step)))
                scaffoldfilter.savetocsv(os.path.join(logdir, "scaffold_memory.{}.csv".format(step)))
            logging.debug("Write Agent memory")
            agent.save(os.path.join(logdir, 'Agent.{}.ckpt'.format(step)))
            if keep_max > 0:
                for oldsteps in range(0, step - (keep_max * save_every) + 1, save_every):
                    with contextlib.suppress(FileNotFoundError):
                        os.remove(os.path.join(logdir, "scaffold_memory.{}.json".format(oldsteps)))
                    with contextlib.suppress(FileNotFoundError):
                        os.remove(os.path.join(logdir, "scaffold_memory.{}.csv".format(oldsteps)))
                    with contextlib.suppress(FileNotFoundError):
                        os.remove(os.path.join(logdir, 'Agent.{}.ckpt'.format(oldsteps)))

        logging.debug("Entropy: {}".format(entropy))

        # reset the weight of NN to search for diverse solutions
        if reset:
            if reset_countdown:
                reset_countdown += 1
            elif mean_score >= reset_score_cutoff:
                reset_countdown = 1

            if reset_countdown == reset:
                agent.reset()
                reset_countdown = 0
                logging.debug("Agent RNN is reset!")

    # If the entire training finishes, we create a new folder where we save some sampled
    # sequences and the contents of the experinence (which are the highest scored
    # sequences seen during training)
    if not os.path.isdir(resultdir):
        os.makedirs(resultdir)

    agent.save(os.path.join(resultdir, 'Agent.ckpt'))
    if experience_replay:
        experience.print_memory(os.path.join(resultdir, "experience_memory"))

    if scaffoldfilter:
        #scaffoldfilter.savetojson(os.path.join(resultdir, "scaffold_memory.json"))
        scaffoldfilter.savetocsv(os.path.join(resultdir, "scaffold_memory.csv"))

    # copy the output.log as well
    copyfile(os.path.join(logdir, "output.log"), os.path.join(resultdir, "output.log"))
    copyfile(os.path.join(logdir, "debug.log"), os.path.join(resultdir, "debug.log"))

    # copy metadata
    copyfile(os.path.join(logdir, "metadata.json"), os.path.join(resultdir, "metadata.json"))
