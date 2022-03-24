import argparse
import os
import shutil
from turtle import mode
from tqdm import tqdm
import logging
from utils.common import read_yaml, create_directories
import random
import numpy as np
import tensorflow as tf
from utils.common import read_yaml
import io
import numpy as np



STAGE = "Creating Base Model" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path):
    ## read config files
    config = read_yaml(config_path)
    #params = read_yaml(params_path)
    
    #Getting th Data
    (X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train_full = X_train_full / 255.0
    X_test = X_test / 255.0
    X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

    y_train_bin,y_test_bin,y_valid_bin = update_y_labels_to_odd_even([y_train,y_test,y_valid])

    #Setting the seed
    seed=int(config['params']['seed'])
    tf.random.set_seed(seed)
    np.random.seed(seed)

    #Updating the Y labels for validation_data, test_data and TrainingData
    def update_y_labels_to_odd_even(list_of_labels):
        for idx, label in enumerate(list_of_labels):
            list_of_labels[idx]=np.where(label%2==0,1,0)
        return list_of_labels

    ##Loading the base model
    base_model_path=os.path.join("artifcats","models","base_model.h5")
    base_model=tf.keras.load_model(base_model_path)

    #Freezing the layers for Transfer learning
    for layer in base_model.layers[:-1]:
        layer.trainable=False

    base_layers=base_model.layers[:-1]

    #Loading all the base Layers except the Output layer
    new_model=tf.keras.models.Sequential(base_layers)

    ##Adding the Last layer to the New Model
    new_model.add(
        tf.kers.layers.Dense(2, name="outputlayer",activation="softmax")
    )

    ##Compiling the Model
    LOSS = "sparse_categorical_crossentropy"
    OPTIMIZER = tf.keras.optimizers.SGD(learning_rate=1e-3)
    METRICS = ["accuracy"]

    new_model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS) 

    #Logging mOdel summary as It return can't be directly done like usual
    def log_model_summary(model):
        with io.StringIO() as stream:
            model.summary(print_fn=lambda x: stream.write(f"{x}\n"))
            summary_str=stream.getvalue()
        return summary_str

    logging.info(f"{STAGE}: \n {log_model_summary(new_model)}")
    
     ## Train the model
    history = new_model.fit(
        X_train, y_train_bin, 
        epochs=10, 
        validation_data=(X_valid, y_valid_bin),
        verbose=2)

    ## save the base model - 
    model_dir_path = os.path.join("artifacts","models")
    create_directories([model_dir_path])

    model_file_path = os.path.join(model_dir_path, "odd_even_model.h5")
    new_model.save(model_file_path)

    logging.info(f"base model is saved at {model_file_path}")
    logging.info(f"evaluation metrics {new_model.evaluate(X_test, y_test_bin)}")

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    #args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e