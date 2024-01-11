import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from mlstars.utils import import_object
from tqdm import tqdm

def build_layer(layer, hyperparameters):
    layer_class = import_object(layer['class'])
    layer_kwargs = layer['parameters'].copy()
    # if issubclass(layer_class, tf.keras.layers.Wrapper):
    #     layer_kwargs['layer'] = build_layer(layer_kwargs['layer'], hyperparameters)
    for key, value in layer_kwargs.items():
        if isinstance(value, str):
            layer_kwargs[key] = hyperparameters.get(value, value)
    return layer_class(**layer_kwargs)

class LSTMSeq2Seq(nn.Module):
    def __init__(self, hyperparameters, layers):
        super(LSTMSeq2Seq, self).__init__()
        # self.layers = layers
        hyperparameters = hyperparameters.copy()
        self.lstm_layer1 = build_layer(layers[0], hyperparameters)
        self.drop_layer1 = build_layer(layers[1], hyperparameters)
        self.lstm_layer2 = build_layer(layers[2], hyperparameters)
        self.drop_layer2 = build_layer(layers[3], hyperparameters)
        self.dense = build_layer(layers[4], hyperparameters)

    def forward(self, sequence):
        whole_sequence, (h1, c1) = self.lstm_layer1(sequence)
        drop_output1 = self.drop_layer1(whole_sequence)
        _, (h2, c2) = self.lstm_layer2(drop_output1)
        drop_output2 = self.drop_layer2(h2)
        out = self.dense(drop_output2[0])
        return out
class timeseries(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype = torch.float32)
        self.y = torch.tensor(y, dtype = torch.float32)
        self.len = X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    def __len__(self):
        return self.len
        
class Sequential(object):

    # def __getstate__(self):
    #     state = self.__dict__.copy()

    #     with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as fd:
    #         tf.keras.models.save_model(state.pop('model'), fd.name, overwrite=True)
    #         state['model_str'] = fd.read()

    #     return state

    # def __setstate__(self, state):
    #     with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as fd:
    #         fd.write(state.pop('model_str'))
    #         fd.flush()

    #         state['model'] = torch.tf.keras.models.load_model(fd.name)

    #     self.__dict__ = state

    # def _build_model(self, **kwargs):
    #     hyperparameters = self.hyperparameters.copy()
    #     hyperparameters.update(kwargs)

    #     model = tf.keras.models.Sequential()

    #     for layer in self.layers:
    #         built_layer = build_layer(layer, hyperparameters)
    #         model.add_module(built_layer)
    #     ''' problem '''
    #     model.compile(loss=self.loss, optimizer=self.optimizer(), metrics=self.metrics)
    #     return model
        
    def __init__(self, layers, loss, optimizer, classification, callbacks=tuple(),
                 metrics=None, epochs=10, verbose=False, validation_split=0, batch_size=32,
                 shuffle=True, **hyperparameters):

        self.layers = layers
        self.optimizer = import_object(optimizer)
        self.loss = import_object(loss)
        self.metrics = metrics

        self.epochs = epochs
        self.verbose = verbose
        self.classification = classification
        self.hyperparameters = hyperparameters
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._fitted = False
        self.device = 'cpu'

        # for callback in callbacks:
        #     callback['class'] = import_object(callback['class'])

        # self.callbacks = callbacks

    def _setdefault(self, kwargs, key, value):
        if key in kwargs:
            return

        if key in self.hyperparameters and self.hyperparameters[key] is None:
            kwargs[key] = value

    def _augment_hyperparameters(self, X, mode, kwargs):
        shape = np.asarray(X)[0].shape
        if shape:
            length = shape[0]
        else:
            length = 1  # supporting shape (l, )

        self._setdefault(kwargs, '{}_shape'.format(mode), shape)
        self._setdefault(kwargs, '{}_dim'.format(mode), length)
        self._setdefault(kwargs, '{}_length'.format(mode), length)

        return kwargs
    def fit(self, X, y, **kwargs):
        if not self._fitted:
            self._augment_hyperparameters(X, 'input', kwargs)
            self._augment_hyperparameters(y, 'target', kwargs)
            self.model = LSTMSeq2Seq(self.hyperparameters, self.layers)
        if torch.cuda.is_available():
            self.device = 'cuda'
        
        self.model = self.model.to(self.device)
        X = X.to(self.device)
        y = y.to(self.device)
        dataset = timeseries(X, y)
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(self.validation_split * dataset_size))
        # if self.shuffle:
        #     np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        train_loader = torch.utils.data.DataLoader(dataset, 
                                                   batch_size=self.batch_size, 
                                                   sampler=train_sampler, 
                                                   num_workers=4)
        
        validation_loader = torch.utils.data.DataLoader(dataset, 
                                                        batch_size=self.batch_size,
                                                        sampler=valid_sampler, 
                                                        num_workers=4)

        # train_loader = DataLoader(dataset, shuffle = self.shuffle, batch_size = self.batch_size)
        optim = self.optimizer(self.model.parameters())
        train_loss = []
        valid_loss = []
        for epoch in range(self.epochs):
            self.model.train()
            train_loss_epoch = 0
            with tqdm(train_loader, unit="batch") as tepoch:
                for X_batch, y_batch in tepoch:
                    y_pred = self.model(X_batch)
                    loss = self.loss(y_pred, y_batch)
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    train_loss_epoch += loss.item()
                train_loss_epoch /= len(train_loader)
                train_loss.append(train_loss_epoch)

            self.model.eval()
            with torch.no_grad():
                valid_loss_epoch = 0
                for X_batch, y_batch in validation_loader:
                    y_pred = self.model(X_batch)
                    loss = self.loss(y_pred, y_batch)
                    valid_loss_epoch += loss.item()
                valid_loss_epoch /= len(validation_loader)
            valid_loss.append(valid_loss_epoch)
            if (epoch+1)%5 == 0:
                print("Epoch %d / %d: train loss %.4f valid loss %.4f " % 
                    (epoch+1, self.epochs, train_loss_epoch, valid_loss_epoch))
        #     y = tf.keras.utils.to_categorical(y)

        # callbacks = [
        #     callback['class'](**callback.get('args', dict()))
        #     for callback in self.callbacks
        # ]

        self._fitted = True

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            y = self.model(torch.tensor(X, dtype = torch.float32))

        if self.classification:
            y = np.argmax(y, axis=1)

        return y.detach().cpu().numpy()