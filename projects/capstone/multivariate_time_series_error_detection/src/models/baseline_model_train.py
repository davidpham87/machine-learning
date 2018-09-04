import functools
import numpy as np
import pandas as pd

from scipy.special import expit

import sklearn as sk
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from tensorflow import keras
import tensorflow as tf

def split_companies_train_dev_test(companies):
    "Return train, dev, test set for companies"
    train, test = train_test_split(companies, test_size=0.1, stratify = companies.sector)
    train, dev = train_test_split(train, test_size=0.1, stratify = train.sector)
    return train, dev, test


def filter_stocks(stocks, tickers):
    return stocks.loc[tickers]


def df_to_ts(df):
    res = df.copy()
    res.index = pd.DatetimeIndex(pd.to_datetime(res.date))
    res.drop('date', axis=1)
    return res


def log_softmax(x):
    return x - np.log(np.sum(np.exp(x)))


def sigmoid(x):
    return expit(x)


def sample_correlation(df, window_size=63):
    idx = np.random.randint(0, df.shape[0]-window_size)
    ts = df[idx:idx+window_size]
    fmap = lambda s: ts['pct_return'].corr(ts[s])
    indices = ts.columns.tolist()[1:]
    correlations = np.array(list(map(fmap, indices)))
    return correlations


def create_correlation_score(df, sample_size=1):
    res = np.array([log_softmax(sample_correlation(df)/0.05)
                    for i in range(sample_size)])
    return np.exp(np.nanmean(res, 0))


def load_data(stock_filename=None, indices_filename=None):

    if stock_filename is None:
        stock_filename = '../../data/processed/wiki_stocks_returns.csv'

    if indices_filename is None:
        indices_filename = '../../data/processed/wiki_indices_returns.csv'

    stocks = pd.read_csv(stock_filename, index_col=False) # long format
    indices = pd.read_csv(indices_filename, index_col=False) # wide format

    # Implementation of hierarchical clustering
    drop_column = lambda df,i=0: df.drop(df.columns[i], axis=1)

    stocks = drop_column(stocks)
    stocks = stocks.drop('name', axis=1)
    stocks = stocks.dropna()

    companies = stocks.groupby('ticker').first().reset_index()
    sectors_counts = companies.sector.value_counts()
    sectors_proportions = sectors_counts/sectors_counts.sum()
    sectors_unique = sectors_counts.index.tolist()

    stocks = stocks.set_index('ticker')

    indices_ts = df_to_ts(indices[['date'] + sectors_unique])
    stocks_ts = df_to_ts(stocks.reset_index())

    stocks_all = pd.merge(stocks_ts, indices_ts, 'left')
    stocks_all = stocks_all.dropna() # loss of 200 000 observations
    stocks_all = stocks_all.drop('sector', axis=1)
    stocks_all = stocks_all.groupby('ticker').apply(df_to_ts)
    stocks_all = stocks_all.drop(['ticker', 'date'], axis=1)
    stocks_all = stocks_all.rename(columns={'close': 'pct_return'})

    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(sectors_counts.index.tolist())
    ticker_to_sector = dict(zip(companies.ticker, label_encoder.transform(companies.sector)))

    return stocks_all, companies, label_encoder, ticker_to_sector

def sectors_statistics(companies):
    sectors_counts = companies.sector.value_counts()
    sectors_proportions = sectors_counts/sectors_counts.sum()
    sectors_unique = sectors_counts.index.tolist()
    return sectors_counts, sectors_proportions, sectors_unique


def add_common_layers(y):
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.LeakyReLU()(y)
    return y


def grouped_convolution(y, nb_channels, _strides, cardinality=4):
    # when `cardinality` == 1 this is just a standard convolution
    if cardinality == 1:
        return keras.layers.Conv1D(nb_channels, kernel_size=10, strides=_strides, padding='same')(y)

    assert not nb_channels % cardinality
    _d = nb_channels // cardinality

    # in a grouped convolution layer, input and output channels are divided into `cardinality` groups,
    # and convolutions are separately performed within each group
    groups = []
    for j in range(cardinality):
        group = keras.layers.Lambda(lambda z: z[:, :, j * _d:j * _d + _d])(y)
        groups.append(keras.layers.Conv1D(_d, kernel_size=10, strides=_strides, padding='same')(group))

    # the grouped convolutional layer concatenates them as the outputs of the layer
    y = keras.layers.concatenate(groups)

    return y


def residual_block(y, nb_channels_in, nb_channels_out, cardinality=4, _strides=1, _project_shortcut=False):
    """
    Our network consists of a stack of residual blocks. These blocks have the same topology,
    and are subject to two simple rules:
    - If producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes).
    - Each time the spatial map is down-sampled by a factor of 2, the width of the blocks is multiplied by a factor of 2.
    """
    shortcut = y
    kl = keras.layers
    # we modify the residual building block as a bottleneck design to make the network more economical
    y = kl.Conv1D(nb_channels_in, kernel_size=1, strides=1, padding='same')(y)
    y = add_common_layers(y)

    # ResNeXt (identical to ResNet when `cardinality` == 1)
    y = grouped_convolution(y, nb_channels_in, _strides=_strides)
    y = add_common_layers(y)

    y = kl.Conv1D(nb_channels_out, kernel_size=1, strides=1, padding='same')(y)
    # batch normalization is employed after aggregating the transformations and before adding to the shortcut
    y = kl.BatchNormalization()(y)

    # identity shortcuts used directly when the input and output are of the same dimensions
    if _project_shortcut or _strides != 1:
        # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
        # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
        shortcut = kl.Conv1D(nb_channels_out, kernel_size=1, strides=_strides, padding='same')(shortcut)
        shortcut = kl.BatchNormalization()(shortcut)

    y = kl.add([shortcut, y])

    # relu is performed right after each batch normalization,
    # expect for the output of the block where relu is performed after the adding to the shortcut
    y = kl.LeakyReLU()(y)

    return y



# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = keras.backend.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = tf.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class CovarianceLayer(keras.layers.Layer):

    def __init__(self, num_classes, **kwargs):
        self.num_classes = num_classes
        super(CovarianceLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(CovarianceLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        series_input, environment_input = inputs
        series_input_multiple = tf.tile(series_input, [1, 1, self.num_classes])
        covariances = tf.reduce_mean(series_input_multiple * environment_input, axis=1);
        return covariances

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


def make_model(num_classes=16, window_size=21, latent_dim=32):
    kl = keras.layers

    series_input = keras.layers.Input(shape=(window_size, 1), dtype='float32', name='series_input')
    environment_input = kl.Input(shape=(window_size, num_classes), dtype='float32', name='environment_input')

    x = kl.Concatenate()([series_input, environment_input])
    x = kl.Conv1D(16, 21, 1, activation='relu')(x)
    x = kl.BatchNormalization()(x)
    x = residual_block(x, 16, 16, 2)
    x = kl.MaxPool1D()(x)
    x = kl.Conv1D(32, 5, 1, activation='relu')(x)
    x = residual_block(x, 32, 32, 2)
    x = kl.MaxPool1D()(x)
    x = kl.BatchNormalization()(x)
    x = kl.Conv1D(64, 2, 1, activation='relu')(x)
    x = kl.MaxPool1D()(x)

    series_input_normalized = kl.BatchNormalization(center=False, axis=1)(series_input)
    environment_input_normalized = kl.BatchNormalization(axis=1, center=False)(environment_input)
    covariances = CovarianceLayer(num_classes)([series_input_normalized, environment_input_normalized])
    covariances = kl.Dense(16, activation='relu')(covariances)

    x = kl.Flatten()(x)
    x = kl.Dense(num_classes, 'relu')(x)
    x = kl.Dropout(0.5)(x)
    z = kl.Concatenate()([x, covariances])

    x_pred = kl.Dense(num_classes, 'softmax')(z)

    model = keras.Model(inputs = [series_input, environment_input], outputs=[x_pred], name='Classifier')

    # kl_batch = - .5 * tf.reduce_sum(1 + x_log_var - tf.square(x_mu) - tf.exp(x_log_var), axis=-1)
    # model.add_loss(kl_batch)

    return model


def random_subset(df, window_size=21):
    idx = np.random.randint(0, df.shape[0]-window_size)
    ts = df[idx:idx+window_size]
    return ts


def make_keras_subset(dataset_type, companies_data, stocks_data, label_encoder, batch_size, window_size=21):
    idx = np.random.choice(companies_data[dataset_type].shape[0], batch_size)
    df = companies_data[dataset_type].iloc[idx]

    model_input_data = [random_subset(stocks_data[dataset_type].loc[t], window_size) for t in df.ticker]
    model_series_input = np.array([df['pct_return'].values for df in model_input_data])
    model_series_input = model_series_input.reshape(-1, window_size, 1)

    model_environment_input = np.array([df.iloc[:, 1:].values for df in model_input_data])

    y_true = label_encoder.transform(df.sector)

    return model_series_input, model_environment_input, y_true


class StocksSequence(keras.utils.Sequence):

    def __init__(self, stocks_data,  companies_data, window_size, label_encoder, batch_size):
        self.stocks_data = stocks_data
        self.batch_size = batch_size
        self.label_encoder = label_encoder
        self.companies_data = companies_data
        self.window_size = window_size

    def __len__(self):
        return int(np.ceil(self.stocks_data.shape[0] / float(self.batch_size)))

    def __getitem__(self, idx):

        idx = np.random.choice(self.companies_data.shape[0], self.batch_size)
        df = self.companies_data.iloc[idx]
        model_input_data = [random_subset(self.stocks_data.loc[t], self.window_size) for t in df.ticker]
        model_series_input = np.array([df['pct_return'].values for df in model_input_data])
        model_series_input = model_series_input.reshape(-1, self.window_size, 1)
        model_environment_input = np.array([df.iloc[:, 1:].values for df in model_input_data])
        y_true = self.label_encoder.transform(df.sector)

        return [model_series_input, model_environment_input], y_true


if __name__ == "__main__":

    # Make train dev test set.
    np.random.seed(42)

    ### Feature engineering

    stocks_all, companies, label_encoder, ticker_to_sector = load_data()
    sectors_counts, sectors_proportions, sectors_unique = sectors_statistics(companies)

    max_proportion_baseline = sectors_proportions.max()
    biggest_sector = sectors_proportions.argmax()

    print("Most representated class:", biggest_sector, ', with proportion of ', round(100*max_proportion_baseline, 2), '%.')
    # Accuracy of our models should be better than max_proportion_baseline.

    companies_data = {}
    data_split = split_companies_train_dev_test(companies)
    for i, k in enumerate(['train', 'dev', 'test']):
        companies_data[k] = data_split[i]
    stocks_data = {k: filter_stocks(stocks_all, v.ticker) for k, v in companies_data.items()}

    ### Correlation scores

    def fmap(companie_ticker, dataset):
        return create_correlation_score(dataset.loc[companie_ticker])


    if True:
        accuracies = {}.fromkeys(['train', 'dev', 'test'], 0)
        accuracies_df = pd.Series(accuracies)
        scores_prediction = dict(accuracies)
        predictions = {}
        n_sample = 100
        for i in range(n_sample):
            for dataset_type in ['train', 'dev', 'test']:
                tickers = companies_data[dataset_type].ticker.tolist()
                fmap = functools.partial(fmap, dataset=stocks_data[dataset_type])
                scores = dict(zip(tickers, map(fmap, tickers)))
                scores = pd.DataFrame(scores, index=stocks_all.columns[1:])
                scores = scores.dropna(axis=1)
                y_pred = label_encoder.transform(scores.apply(np.argmax))
                y_true = np.array([ticker_to_sector[k] for k in scores.columns])
                accuracies[dataset_type] = accuracy_score(y_true, y_pred)
                scores_prediction[dataset_type] += scores
                # predictions[dataset_type] = y_pred
            accuracies = pd.Series(accuracies)
            accuracies_df += accuracies
        print("Using sample correlation, we have the followin accuracies:\n", accuracies_df/n_sample)
        # dev      0.581667
        # test     0.607123
        # train    0.587883


    window_size = 63
    batch_size = 64

    model = make_model(window_size=window_size)
    optimizer = keras.optimizers.Adam(0.0001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())

    callbacks = [
        keras.callbacks.ModelCheckpoint('checkpoint/model_weights.json', monitor='val_acc', verbose=1, save_best_only=True, mode='max'),
        keras.callbacks.TensorBoard()
    ]

    stocks_sequence_training = StocksSequence(stocks_data['train'], companies_data['train'], window_size, label_encoder, batch_size)
    stocks_sequence_validation = StocksSequence(stocks_data['dev'], companies_data['dev'], window_size, label_encoder, batch_size)

    model.fit_generator(
        stocks_sequence_training, steps_per_epoch=2000, epochs=400, callbacks=callbacks,
        validation_data = stocks_sequence_validation, validation_steps=200, workers=4, max_queue_size=20, verbose=2, use_multiprocessing=True)

    stocks_sequence_test = StocksSequence(stocks_data['test'], companies_data['test'], window_size, label_encoder, batch_size)

    model.evaluate_generator(stocks_sequence_test, 200, workers=2, use_multiprocessing=True)
