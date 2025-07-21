import algo
import utils
import dataset

import torch
import torch.nn as nn
import torch.optim as optim

def eval(net, debug=False):
    net.eval()
    total_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, labels in dataset.trainloader:
            inputs, labels = inputs.to(utils.device), labels.to(utils.device)
            
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            accuracy = correct / total
    
    if debug:
        print(f"Loss: {total_loss:.5f} | Accuracy: {accuracy:.5f}")

    return accuracy

def train(net, epochs, debug=False):
    net.to(utils.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    for epoch in range(epochs):
        net.train()
        total_loss = 0
        correct = 0
        total = 0

        for inputs, labels in dataset.trainloader:
            inputs, labels = inputs.to(utils.device), labels.to(utils.device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        accuracy = correct / total

        if debug:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.5f} | Accuracy: {accuracy:.5f}")

    return accuracy

def test(net, debug=True):
    net.eval()
    total_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, labels in dataset.trainloader:
            inputs, labels = inputs.to(utils.device), labels.to(utils.device)
            
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            accuracy = correct / total
    
    if debug:
        print(f"Loss: {total_loss:.5f} | Accuracy: {accuracy:.5f}")

    return accuracy

def train_population(population, epochs, debug=False):
    accs = []
    for name, network in zip(population.keys(), population.values()):
        acc = train(network, epochs)
        if debug:
            print(f'Model {name} | Acc {acc:.5f}')
        accs.append(acc)
    
    return accs


def training_algo(population, epochs, debug=False):
    
    '''initial gradient decent'''

    accs = train_population(population, epochs, debug)
    nl_name, nl_model, leader_acc = algo.select_leader(population, accs)
    initial_leader_name = nl_name
    initial_best_acc = leader_acc
    
    new_population = None

    if debug:
        print(f'Selected Leader {nl_name} | Accuracy {leader_acc:.5f}\nConverging Population to Leader...')

    for i in range(utils.NUM_OPTIMIZATIONS):
        print(f'Starting Optimization {i}')
        nl_name, nl_model, nl_acc =  algo.follow_leader(nl_name, nl_model, leader_acc, population, debug=debug).values()

        # if nl_name == initial_leader_name:
        #     if debug:
        #         print('No New Leader Found. Retraining Models')
            
        #     new_population = algo.init_models(len(population.keys()) - 1, 28*28)
        #     accs = train_population(new_population, epochs)
            
        #     new_population[nl_name] = nl_model
        #     population = new_population

        """check if weve found a new leader"""
        tmp_nl_name = nl_name
        # nl_name, nl_model, nl_acc = algo.select_leader(population, accs)

        if nl_name == tmp_nl_name and debug:
            print(f'Models Trained Leader Has Not Changed {nl_name}. Converging Population to Leader...')

    return nl_name, nl_model, nl_acc, initial_best_acc, new_population

