    # Initilize a new wandb run

    wandb.init(entity="wandb", project="keras-intro")


    # Default values for hyper-parameters

    config = wandb.config # Config is a variable that holds and saves hyperparameters and inputs

    config.learning_rate = 0.01

    config.batch_size = 128

    config.activation = 'relu'

    config.optimizer = 'nadam'