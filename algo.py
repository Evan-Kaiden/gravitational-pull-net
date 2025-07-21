from net import network
import random, string

import model_engine

import torch
import copy
import utils

def random_string(n):
  characters = string.ascii_letters + string.digits
  return ''.join(random.choice(characters) for _ in range(n))

def init_models(n):
    models = {random_string(15) : network() for _ in range(n)}
    return models

def select_leader(models, model_accs):
    best_key, best_acc = max(zip(models.keys(), model_accs), key=lambda x: x[1])
    return best_key, models[best_key], best_acc

def soft_update(leader, follower, tau=0.01):
    """Polyak style averaging to update weights"""
    with torch.no_grad():
        for leader_param, follower_param in zip(leader.parameters(), follower.parameters()):
            updated_weight = follower_param * (1- tau) + leader_param * (tau)
            follower_param.data.copy_(updated_weight)

def follow_leader(leader_name, leader_model, leader_acc, models, debug=False):
    new_leader_info = {'name' : leader_name, 
                       'model' : leader_model, 
                       'acc' : leader_acc
    }

    competitors = {}
    for name, follower in models.items():
        if name != new_leader_info['name']:
            if debug:
                print(f'Current Leader: {new_leader_info['name']}')
            competitors[name] = (copy.deepcopy(follower), model_engine.eval(follower))

            first_better_seen = True
            first_update_idx = None
            for i in range(utils.UPDATE_ITERATIONS):
                soft_update(new_leader_info['model'], follower)
                follower_acc = model_engine.eval(follower)
                if follower_acc > leader_acc and first_better_seen:
                    competitors[name] = (copy.deepcopy(follower), follower_acc)

                    first_update_idx = i
                    first_better_seen = False
                    if debug:
                        print(f'Leader {new_leader_info['name']} Has Been Overtaken by {name} | New Leader Acc {follower_acc:.5f} > {new_leader_info['acc']:.5f}\nAttempting to Improve {name} for {utils.IMPROVEMENT_CUTOFF_ITERATIONS - (i - first_update_idx)} More Iterations...')
                if not first_better_seen and debug and (utils.IMPROVEMENT_CUTOFF_ITERATIONS - (i - first_update_idx)) != utils.IMPROVEMENT_CUTOFF_ITERATIONS:
                    print(f'Attempting to improve {name} for {utils.IMPROVEMENT_CUTOFF_ITERATIONS - (i - first_update_idx)} More Iterations...')
                if not first_better_seen and follower_acc > competitors[name][1]:
                    if debug:
                        print(f'{name} has Improved {competitors[name][1]:.5f}->{follower_acc:.5f}')
                    competitors[name] = (copy.deepcopy(follower), follower_acc)
                    

                if first_update_idx is not None and i - first_update_idx == utils.IMPROVEMENT_CUTOFF_ITERATIONS:            
                    break

    best_follower = max(competitors, key=lambda k : competitors[k][1]) 
    names, models = [*competitors.keys()] + [new_leader_info['name']], [val[0] for val in competitors.values()] + [new_leader_info['model']]

    if competitors[best_follower][1] > leader_acc:
        new_leader_info = {'name' : best_follower, 
                        'model' : competitors[best_follower][0], 
                        'acc' : competitors[best_follower][1]
        }
    
    return new_leader_info, dict(zip(names, models))


