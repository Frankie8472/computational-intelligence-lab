import numpy as np
from keras import backend as K
from keras.layers import Input, Embedding, Flatten, Dot, Dense, Dropout, BatchNormalization, Add, Concatenate, Lambda, PReLU
from keras.models import Model
from keras.utils import to_categorical
from franz.porter import export_data, import_dataframe


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def keras_basic_model(n_net, input_dim_1, input_dim_2, n_latent_factors=64, merge_type='dot'):
    # Define input layers
    user_input = Input(name='user_input', shape=(1,))
    movie_input = Input(name='movie_input', shape=(1,))

    # Embedding layer for creating two dynamic matrices, containing the latent features
    user_embedding = Embedding(name='user_embedding', input_dim=input_dim_1, output_dim=n_latent_factors)(user_input)
    movie_embedding = Embedding(name='movie_embedding', input_dim=input_dim_2, output_dim=n_latent_factors)(movie_input)

    # Embedding layer for creating two dynamic vectors, containing the latent feature bias
    user_bias = Embedding(name='user_bias', input_dim=input_dim_1, output_dim=1)(user_input)
    movie_bias = Embedding(name='movie_bias', input_dim=input_dim_2, output_dim=1)(movie_input)

    # Flatten layer for adjusting dimensions
    user_vec = Flatten()(user_embedding)
    movie_vec = Flatten()(movie_embedding)
    user_bias = Flatten()(user_bias)
    movie_bias = Flatten()(movie_bias)

    # Dropout layer to reduce overfitting and improve generalization error
    user_vec = Dropout(0.5)(user_vec)
    movie_vec = Dropout(0.5)(movie_vec)

    if merge_type == 'concat':
        # Concatenate layer to retain each latent feature/bias in its original form
        merged = Concatenate(name='concat_latent_features')([user_vec, movie_vec, user_bias, movie_bias])
    else:  # merge_type == 'dot'
        # Dot layer to multiply vectors together to get a_ij = y'
        merged = Dot(name='dot_product', normalize=True, axes=1)([user_vec, movie_vec])

        # Add layer to add both biases
        merged = Add(name='add_biases')([merged, user_bias, movie_bias])

    return n_net(merged, [user_input, movie_input])


# Standardized normal neural net layer
def keras_nn_layer_bundle(input_layer, units=1, dropout_rate=0.5):
    nn = Dense(units=units, kernel_initializer='he_normal')(input_layer)    # he_normal is better for *ReLU activation functions
    nn = BatchNormalization()(nn)
    nn = PReLU()(nn)
    nn = Dropout(rate=dropout_rate)(nn)
    return nn


# Regression is the task of predicting a continuous quantity
def keras_nn_reg_model(input_layer, nn_input):
    nn = keras_nn_layer_bundle(input_layer=input_layer, units=64, dropout_rate=0.5)
    # nn = keras_nn_layer_bundle(input_layer=nn, units=32, dropout_rate=0.5)
    # nn = keras_nn_layer_bundle(input_layer=nn, units=64, dropout_rate=0.5)

    # Last layer with one node, containing the actual class number
    nn = Dense(1, kernel_initializer='glorot_uniform', activation='sigmoid')(nn)

    # Lambda layer to scale sigmoid (0,1) to (1,5)
    nn = Lambda(lambda x: x * (5.5 - 0.5) + 0.5)(nn)

    # Initialize model with input and output
    nn_output = nn
    model = Model(inputs=nn_input, outputs=nn_output)

    # Compile model with specific specs for regression
    model.compile(optimizer='Adam', loss=root_mean_squared_error, metrics=['accuracy'])

    return model


# Classification is the task of predicting a discrete class label.
def keras_nn_clf_model(layer, nn_input):
    nn = keras_nn_layer_bundle(input_layer=layer, units=128, dropout_rate=0.5)
    # nn = keras_nn_layer_bundle(input_layer=nn, units=32, dropout_rate=0.5)
    # nn = keras_nn_layer_bundle(input_layer=nn, units=64, dropout_rate=0.5)

    # Last layer with 5 nodes, containing the probability of "is in class 1..5"
    nn = Dense(units=5, activation='softmax')(nn)

    # Initialize model with input and output
    nn_output = nn
    model = Model(inputs=nn_input, outputs=nn_output)

    # Compile model with specific specs for classification
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy', 'categorical_accuracy'])
    return model


def init_nn(train_data, valid_data, pred_data, input_dim_1, input_dim_2,
            n_latent_factors=50, model_type='reg',
            merge_type='dot', batch_size=1000, epochs=20,
            prediction=False, show_only=False):

    # Initialize keras model+layers
    if model_type == 'reg':
        model_type_func = keras_nn_reg_model
        y_train = train_data.rating
        y_test = valid_data.rating
    else:
        model_type_func = keras_nn_clf_model
        y_train = to_categorical(train_data.rating - 1)
        y_test = to_categorical(valid_data.rating - 1)

    model = keras_basic_model(
        n_net=model_type_func,
        input_dim_1=input_dim_1,
        input_dim_2=input_dim_2,
        n_latent_factors=n_latent_factors,
        merge_type=merge_type
    )

    # If show_only, only display a summary of the model
    if show_only:
        return model.summary()

    # Train model
    model.fit(x=[train_data.userId, train_data.movieId],
              y=y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=([valid_data.userId, valid_data.movieId], y_test),
              shuffle=True,
              workers=1,
              use_multiprocessing=False
              )

    # Create and export prediction
    if prediction:
        y_pred = model.predict(x=[pred_data.userId, pred_data.movieId], verbose=1)
        if model_type == 'clf':
            y_pred = np.argmax(y_pred, axis=1)+1
        export_data(np.c_[pred_data.userId.to_numpy(), pred_data.movieId.to_numpy(), y_pred], "franz")
    return


def main():
    df = import_dataframe()
    df_pred = import_dataframe(filepath='../output/sampleSubmission.csv')

    users = df.userId.unique()
    movies = df.movieId.unique()

    userid2idx = {o: i for i, o in enumerate(users)}
    movieid2idx = {o: i for i, o in enumerate(movies)}

    df['userId'] = df['userId'].apply(lambda x: userid2idx[x])
    df['movieId'] = df['movieId'].apply(lambda x: movieid2idx[x])
    split = np.random.rand(len(df)) < 0.8
    train_data = df[split]
    valid_data = df[~split]

    n_users = len(df['userId'].unique())
    n_movies = len(df['movieId'].unique())

    init_nn(
        train_data=train_data,
        valid_data=valid_data,
        pred_data=df_pred,
        input_dim_1=n_users,
        input_dim_2=n_movies,
        n_latent_factors=50,
        model_type='clf',  # 'reg' or 'clf'
        merge_type='dot',  # 'dot' or 'concat' or 'euclid'
        batch_size=500,
        epochs=30,
        prediction=True,
        show_only=True
    )

    return


main()
