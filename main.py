import model_engine
import utils
import algo

def main():
    population = algo.init_models(utils.NUM_AGENTS, 28*28)
    l_name, l_model, l_acc, init_l_acc, new_population = model_engine.training_algo(population, utils.TRAIN_EPOCHS, debug=False)

    print(f'\nBest Model {l_name} With Acc {l_acc} Improved from {init_l_acc} initially')


if __name__ == '__main__':
    main()