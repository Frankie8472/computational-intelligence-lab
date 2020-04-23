import numpy as np
from keras import backend as K
from keras.losses import mean_squared_error
from keras.layers import Input, Embedding, Flatten, Dot, Dense, Dropout, BatchNormalization, Reshape, Add, Concatenate, \
    Lambda
from keras.models import Model
from keras.optimizers import Adam

from franz.porter import export_data, import_dataframe


# from sklearn.experimental import enable_iterative_imputer
# enable_iterative_imputer
# from sklearn.impute import IterativeImputer
# from franz.feature_enhancer import sparse_svd
# from scipy import sparse as sp
# from scipy.sparse.linalg import svds
# from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, BiScaler, MatrixFactorization

# Import data


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(mean_squared_error(y_true, y_pred))


def keras_dot_model(input_dim_1, input_dim_2, n_latent_factors=64):
    user_input = Input(shape=(1,), name='user_input')
    movie_input = Input(shape=(1,), name='movie_input')

    user_embedding = Embedding(name='user_embedding', input_dim=input_dim_1, output_dim=n_latent_factors)(user_input)
    movie_embedding = Embedding(name='movie_embedding', input_dim=input_dim_2, output_dim=n_latent_factors)(movie_input)

    user_embedding_bias = Embedding(name='user_embedding_bias', input_dim=input_dim_1, output_dim=1)(user_input)
    movie_embedding_bias = Embedding(name='movie_embedding_bias', input_dim=input_dim_2, output_dim=1)(movie_input)

    user_vec = Flatten(name='FlattenUsers')(user_embedding)
    movie_vec = Flatten(name='FlattenMovies')(movie_embedding)

    user_embedding_bias = Flatten()(user_embedding_bias)
    movie_embedding_bias = Flatten()(movie_embedding_bias)

    user_vec = Dropout(0.5)(user_vec)
    movie_vec = Dropout(0.5)(movie_vec)

    merged = Dot(name='dot_product', normalize=True, axes=1)([user_vec, movie_vec])

    biased = Add()([merged, user_embedding_bias, movie_embedding_bias])

    nn = Dense(96, activation='relu')(biased)
    nn = Dropout(0.5)(nn)

    nn = BatchNormalization()(nn)

    nn = Dense(1, activation='relu')(nn)

    model = Model(inputs=[user_input, movie_input], outputs=nn)
    #model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=Adam(lr=1e-3), loss=root_mean_squared_error)

    return model


def keras_nn_model(input_dim_1, input_dim_2, n_latent_factors=64):
    user_input = Input(shape=(1,), name='user_input')
    movie_input = Input(shape=(1,), name='movie_input')

    user_embedding = Embedding(name='user_embedding', input_dim=input_dim_1, output_dim=n_latent_factors)(user_input)
    movie_embedding = Embedding(name='movie_embedding', input_dim=input_dim_2, output_dim=n_latent_factors)(movie_input)

    user_embedding_bias = Embedding(name='user_embedding_bias', input_dim=input_dim_1, output_dim=1)(user_input)
    movie_embedding_bias = Embedding(name='movie_embedding_bias', input_dim=input_dim_2, output_dim=1)(movie_input)

    user_vec = Flatten()(user_embedding)
    movie_vec = Flatten()(movie_embedding)

    user_embedding_bias = Flatten()(user_embedding_bias)
    movie_embedding_bias = Flatten()(movie_embedding_bias)

    user_vec = Dropout(0.5)(user_vec)
    movie_vec = Dropout(0.5)(movie_vec)

    merged = Concatenate(name='concatenate_of_latent_features')([user_vec, movie_vec])

    biased = Add()([merged, user_embedding_bias, movie_embedding_bias])

    nn = Dense(96, kernel_initializer='he_normal', activation='relu')(biased)
    nn = Dropout(0.5)(nn)

    nn = BatchNormalization()(nn)

    nn = Dense(1, kernel_initializer='he_normal', activation='relu')(nn)

    model = Model(inputs=[user_input, movie_input], outputs=nn)
    #model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=Adam(lr=1e-3), loss=root_mean_squared_error)

    return model


def keras_nn_model_2(input_dim_1, input_dim_2, n_latent_factors=64):
    user_input = Input(shape=(1,), name='user_input')
    movie_input = Input(shape=(1,), name='movie_input')

    user_embedding = Embedding(name='user_embedding', input_dim=input_dim_1, output_dim=n_latent_factors)(user_input)
    movie_embedding = Embedding(name='movie_embedding', input_dim=input_dim_2, output_dim=n_latent_factors)(movie_input)

    user_embedding_bias = Embedding(name='user_embedding_bias', input_dim=input_dim_1, output_dim=1)(user_input)
    movie_embedding_bias = Embedding(name='movie_embedding_bias', input_dim=input_dim_2, output_dim=1)(movie_input)

    user_vec = Flatten()(user_embedding)
    movie_vec = Flatten()(movie_embedding)

    user_embedding_bias = Flatten()(user_embedding_bias)
    movie_embedding_bias = Flatten()(movie_embedding_bias)

    user_vec = Dropout(0.5)(user_vec)
    movie_vec = Dropout(0.5)(movie_vec)

    merged = Concatenate(name='concatenate_of_latent_features')([user_vec, movie_vec])

    biased = Add()([merged, user_embedding_bias, movie_embedding_bias])

    nn = Dense(96, kernel_initializer='he_normal', activation='relu')(biased)
    nn = Dropout(0.5)(nn)

    nn = BatchNormalization()(nn)

    nn = Dense(1, kernel_initializer='glorot_uniform', activation='sigmoid')(nn)

    nn = Lambda(lambda x: x * (5.0 - 1.0) + 1.0)(nn)

    model = Model(inputs=[user_input, movie_input], outputs=nn)
    #model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=Adam(lr=1e-3), loss='mse')

    return model


def run(model, train_data, valid_data):
    batch_size = 128
    epochs = 20

    model.fit([train_data.userId, train_data.movieId], train_data.rating,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=([valid_data.userId, valid_data.movieId], valid_data.rating),
              verbose=1
              )

    return


def main():
    df = import_dataframe()

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

    model = keras_nn_model_2(input_dim_1=n_users, input_dim_2=n_movies, n_latent_factors=50)

    # model.summary()
    run(model, train_data, valid_data)


main()

## Embeddings
# Extract latent features (bias + factor/weights ) (Matrix split intro two matrices: A=G*H)
# Scale to the most important latent features (PCA)

# X_train = int(bool(data))
# X_test = int(not bool(data)) + X_train

# print(X_train)

# data[data == 0] = np.nan

####
# data_pred = IterativeImputer(verbose=2).fit_transform(data)

# data_pred = BiScaler().fit_transform(data)
# data_pred = SoftImpute().fit_transform(data_pred)

# data_pred = KNN(k=3).fit_transform(data)

# data_pred = NuclearNormMinimization(min_value=1.0, max_value=5.0).fit_transform(data)
# data_pred = MatrixFactorization().fit_transform(data)
####

# export_data(data_pred, 'franz')
