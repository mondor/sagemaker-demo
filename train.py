import os
import tensorflow as tf
import pandas as pd
import argparse

print(tf.__version__)

def pack_features_vector(features, labels):
    features = tf.stack(list(features.values()), axis=1)
    return features, labels

def loss(model, x, y, training):
    y_ = model(x, training=training)
    return loss_object(y_true=y, y_pred=y_)

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets, training=True)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_known_args()

    # Sagemaker data / model directories
    data_path = os.environ.get('SM_INPUT_DIR') if os.environ.get('SM_INPUT_DIR') else './data'
    model_path = os.environ.get('SM_MODEL_DIR') if os.environ.get('SM_MODEL_DIR') else './model'

    train_dataset_fp = f'{data_path}/iris_training.csv'
    df = pd.read_csv(train_dataset_fp)
    print(df.head())


    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']
    feature_names = column_names[:-1]
    label_names = column_names[-1]

    # create dataset
    train_dataset = tf.data.experimental.make_csv_dataset(
        train_dataset_fp,
        batch_size=32,
        column_names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'],
        label_name=label_names,
        num_epochs=1
    )

    train_dataset = train_dataset.map(pack_features_vector)

    # simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),
        tf.keras.layers.Dense(10, activation=tf.nn.relu),
        tf.keras.layers.Dense(3)
    ])
    model.summary()

    # training
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    train_loss_results = []
    train_accuracy_results = []
    for epoch in range(1, args.epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        for x, y in train_dataset:
            loss_value, grads = grad(model, x, y)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            epoch_loss_avg.update_state(loss_value)
            epoch_accuracy.update_state(y, model(x, training=True))

        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        print(f"Epoch {epoch} Loss {epoch_loss_avg.result()} Accuracy {epoch_accuracy.result()}")


    model.save_weights(f'{model_path}/iris-checkpoint')
    model.load_weights(f'{model_path}/iris-checkpoint')