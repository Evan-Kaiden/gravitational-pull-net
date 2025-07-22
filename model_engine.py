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

def train(net, epochs=None, steps=None, debug=False):
    net.to(utils.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    if steps:
        step = 0
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
            step += 1

            if step == steps: break

        accuracy = correct / total

        return accuracy
    if epochs:
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

def train_population(population, epochs=None, steps=None, debug=False):
    accs = []
    for name, network in zip(population.keys(), population.values()):
        acc = train(network, epochs, steps, debug=False)
        if debug:
            print(f'Model {name} | Acc {acc:.5f}')
        accs.append(acc)
    
    return accs


def training_algo(population, epochs, debug=False):
    
    '''initial gradient decent'''
    accs = train_population(population, epochs=epochs, debug=debug)
    nl_name, nl_model, nl_acc = algo.select_leader(population, accs)
    initial_best_acc = nl_acc
    
    new_population = population

    if debug:
        print(f'Selected Leader {nl_name} | Accuracy {nl_acc:.5f}\nConverging Population to Leader...')

    for i in range(utils.NUM_OPTIMIZATIONS):
        print(f'Starting Optimization {i}')
        leader_data, new_population = algo.follow_leader(nl_name, nl_model, nl_acc, new_population, debug=debug)
        nl_name, nl_model, nl_acc = leader_data.values()
        """Tune new population"""
        if debug:
            print('Tuning Population...')
        accs = train_population(new_population, steps=utils.TUNE_STEPS, debug=debug)

        """Check if weve found a new leader"""
        tmp_nl_name = nl_name
        nl_name, nl_model, nl_acc = algo.select_leader(new_population, accs)

        if nl_name == tmp_nl_name and debug:
            print(f'Models Trained Leader Has Not Changed {nl_name}. Converging Population to Leader...')
        elif debug:
            print(f'Models Leader Has Changed to {nl_name}. Converging Population to Leader...')

    return nl_name, nl_model, nl_acc, initial_best_acc, new_population

