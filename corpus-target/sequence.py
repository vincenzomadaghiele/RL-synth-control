import librosa
from environment import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import json
import time

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(666)
random.seed(666)
np.random.seed(666)

# find device
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")

class seqModel(nn.Module):
    def __init__(self, input_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=50, 
                            num_layers=num_layers, 
                            batch_first=True)
        self.linear = nn.Linear(50, input_size)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

def create_dataset(dataset, lookback):
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]
        target = dataset[i+1:i+lookback+1]
        X.append(feature)
        y.append(target)
    return torch.tensor(X).to(torch.float32).to(device), torch.tensor(y).to(torch.float32).to(device)


if __name__ == "__main__":

    # feature extraction settings
    sample_rate = 44100
    FFT_window_size = 4096
    hop_size = 2048
    mfcc_N = 13
    chroma_N = 12
    features4training = ['rms','cent','flatness','rolloff']

    # corpus settings
    feature_names = getFeatureNames(features4training, mfcc_N, chroma_N)
    corpus_path = './corpus-target/00_corpus/myvoice'
    model_name = f'seq_model_{int(time.time())}'
    save_dir = f'./corpus-target/01_model_logs/seq_models/{corpus_path.split("/")[-1]}'
    os.makedirs(f'{save_dir}/{model_name}', exist_ok=True)
    os.makedirs(f'{save_dir}/tensorboard_logs/{model_name}', exist_ok=True)

    # model settings
    input_size = len(feature_names)
    num_layers = 4
    learning_rate = 0.0001
    lookback = 10
    n_epochs = 1000

    parameters = {
        "sample_rate": sample_rate,
        "FFT_window_size": FFT_window_size,
        "hop_size": hop_size,
        "mfcc_N": mfcc_N,
        "chroma_N": chroma_N,
        "features4training": features4training,
        "corpus_path": corpus_path,
        "input_size": input_size,
        "num_layers": num_layers,
        "learning_rate": learning_rate,
        "lookback": lookback,
        "n_epochs":n_epochs
    }
    with open(f'{save_dir}/{model_name}/training_config.json', 'w', encoding='utf-8') as f:
        json.dump(parameters, f, ensure_ascii=False, indent=4)


    corpus_files = os.listdir(corpus_path)
    corpus_files = [f'{corpus_path}/{filename}' for filename in corpus_files if filename.split('.')[-1] == "wav"]
    print(f'Loading audio corpus with {len(corpus_files)} files...')

    # load TGT signals and compute TGT scaler
    signals = []
    signals_dfs = []
    for audiofile in corpus_files:
        signal, _ = librosa.load(audiofile, sr=sample_rate, mono=True)
        signal = compensateSignalLoudness(signal, FFT_window_size, hop_size)
        signals.append(signal)
        signal_features = computeFeatures(signal, features4training, sample_rate=sample_rate, 
                                        FFT_window_size=FFT_window_size, hop_size=hop_size, 
                                        mfcc_N=mfcc_N, chroma_N=chroma_N)
        features_df = pd.DataFrame(data=signal_features, columns=feature_names)
        print(f'Loading audiofile {audiofile} with length {signal_features.shape[0]} frames')
        features_df['filename'] = [audiofile for _ in range(features_df.shape[0])]
        signals_dfs.append(features_df)
    corpus_df = pd.concat(signals_dfs, axis=0, ignore_index=True)

    tgt_synth_scaler = StandardScaler()
    # tgt_synth_scaler = MinMaxScaler()
    corpus_features = corpus_df[feature_names].values
    tgt_synth_scaler.fit(corpus_features)

    list_of_dfs = [group for _, group in corpus_df.groupby('filename')]
    Xs_scaled = []
    for df in list_of_dfs:
        df_data = df[feature_names].values
        X_scaled = tgt_synth_scaler.transform(df_data)
        Xs_scaled.append(X_scaled)

    X_train_tot = []
    y_train_tot = []
    X_test_tot = []
    y_test_tot = []
    for X_scaled in Xs_scaled:
        train_size = int(X_scaled.shape[0] * 0.67)
        test_size = X_scaled.shape[0] - train_size
        train, test = X_scaled[:train_size,:], X_scaled[train_size:,:]
        X_train, y_train = create_dataset(train, lookback)
        X_test, y_test = create_dataset(test, lookback)
        X_train_tot.append(X_train)
        y_train_tot.append(y_train)
        X_test_tot.append(X_test)
        y_test_tot.append(y_test)

    X_train = torch.cat(X_train_tot, dim=0)
    y_train = torch.cat(y_train_tot, dim=0)
    X_test = torch.cat(X_test_tot, dim=0)
    y_test = torch.cat(y_test_tot, dim=0)


    model = seqModel(input_size, num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss().to(device)
    loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)
    writer = SummaryWriter(log_dir=f'{save_dir}/tensorboard_logs/{model_name}') 

    best_loss = np.inf
    for epoch in range(n_epochs):
        # train
        model.train()
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # validation
        model.eval()
        with torch.no_grad():
            y_pred = model(X_train)
            train_rmse = np.sqrt(loss_fn(y_pred, y_train).detach().cpu().numpy())
            writer.add_scalar('Loss/train', train_rmse, epoch)
            y_pred = model(X_test)
            test_rmse = np.sqrt(loss_fn(y_pred, y_test).detach().cpu().numpy())
            writer.add_scalar('Loss/test', test_rmse, epoch)
            if test_rmse < best_loss:
                best_loss = test_rmse
                torch.save(model.state_dict(), f'{save_dir}/{model_name}/best_seq_model.pth')
        print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))


    model = seqModel(input_size, num_layers).to(device)
    model.load_state_dict(torch.load(f'{save_dir}/{model_name}/best_seq_model.pth'))
    model.eval()

    y = []
    for i in range(10, Xs_scaled[0].shape[0]):
        x = Xs_scaled[0][i-10:i,:]
        x = torch.tensor(x).to(torch.float32).to(device)
        with torch.no_grad():
            pred = model(x)
        y.append(pred[-1,:].detach().cpu().numpy())

    y = np.array(y)
    LEN_PLOT = 1000
    fig, ax = plt.subplots(2, 2, figsize=(10,5), layout="constrained")
    ax[0,0].plot(Xs_scaled[0][LEN_PLOT+lookback:2*LEN_PLOT+lookback,0])
    ax[0,0].plot(y[LEN_PLOT:2*LEN_PLOT,0])
    ax[1,0].plot(Xs_scaled[0][LEN_PLOT+lookback:2*LEN_PLOT+lookback,1])
    ax[1,0].plot(y[LEN_PLOT:2*LEN_PLOT,1])
    ax[0,1].plot(Xs_scaled[0][LEN_PLOT+lookback:2*LEN_PLOT+lookback,2])
    ax[0,1].plot(y[LEN_PLOT:2*LEN_PLOT,2])
    ax[1,1].plot(Xs_scaled[0][LEN_PLOT+lookback:2*LEN_PLOT+lookback,3])
    ax[1,1].plot(y[LEN_PLOT:2*LEN_PLOT,3])
    ax[0,0].set_title(feature_names[0])
    ax[0,1].set_title(feature_names[1])
    ax[1,0].set_title(feature_names[2])
    ax[1,1].set_title(feature_names[3])
    plt.savefig(f'{save_dir}/{model_name}/seq_model_plot.png')
    
