import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
import joblib
import keras
from keras.layers import Dense , LSTM, Dropout, Activation
from keras.models import Sequential, load_model
from flask import Flask, render_template, url_for, request



def prepare_data():
    #train file
  path_train = 'D:\\4_KH1\\khdl_th\\times_series\\PM_train.txt'
  df = pd.read_csv(path_train, delimiter=' ', header = None)
  df = df.drop([26,27], axis = 1)
  col_names = ['id','cycle','setting1','setting2','setting3','s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12','s13','s14','s15','s16','s17','s18','s19','s20','s21']
  df.columns = col_names

  #test file
  path_test = 'D:\\4_KH1\\khdl_th\\times_series\\PM_test.txt'
  df_test = pd.read_csv(path_test, delimiter= ' ', header = None)
  df_test = df_test.drop([26,27], axis = 1)
  df_test.columns = col_names

  #truth file
  path_truth = 'D:\\4_KH1\\khdl_th\\times_series\\PM_truth.txt'
  df_truth = pd.read_csv(path_truth, delimiter = ' ', header = None)
  df_truth = df_truth.drop([1], axis = 1)
  df_truth.columns = ['cycle_remain']
  df_truth['id'] = df_truth.index + 1

  # prepare train data 
  df['TTF'] = df.groupby('id')['cycle'].transform('last') - df['cycle']
  df['label1'] = df['TTF'].apply(lambda x: 0 if x > 30 else 1)
  df['label2'] = df['TTF'].apply(lambda x: 0 if x > 30 else 1 if x>15 else 2)

  #scaler
  mms = MinMaxScaler()
  df1 = df.copy()
  for col in df1.columns[2:-3]:
    df1[col] = mms.fit_transform(df1[[col]])
  df1[['cycle_norm']] = mms.fit_transform(df1[['cycle']])

  df1 = df1.drop(columns = ['setting3', 's1', 's5', 's6','s9', 's10', 's16', 's18', 's19'] , axis = 1)
  #________________________________________

  # test data
  df_test = df_test.merge(df_truth, on= 'id', how = 'right')
  df_test['TTF'] = df_test.groupby('id')['cycle'].transform('last') + df_test['cycle_remain']- df_test['cycle']
  df_test['label1'] = df_test['TTF'].apply(lambda x: 0 if x > 30 else 1)
  df_test['label2'] = df_test['TTF'].apply(lambda x: 0 if x > 30 else 1 if x>15 else 2)
  df_test = df_test.drop('cycle_remain', axis = 1)
  d_1 = df_test.copy()
  for col in d_1.columns[2:26]:
    d_1[col] = mms.fit_transform(d_1[[col]])
  d_1[['cycle_norm']] = mms.fit_transform(d_1[['cycle']])
  d_1 = d_1.drop(columns = ['setting3', 's1', 's5','s6','s9', 's10', 's16', 's18', 's19'] , axis = 1)

  X_train = df1.drop(['id', 'cycle','label1', 'label2', 'cycle_norm', 'TTF','cycle_norm'], axis = 1)
  y_train = df1['label1']
  X_test = d_1.drop(['id', 'cycle','label1', 'label2', 'cycle_norm', 'TTF','cycle_norm'], axis = 1)
  y_test = d_1['label1']
  return X_train, X_test, y_train, y_test

def train_lstm_label1(df1):
  sq_length = 50
  def prepare_data(id_in_df,sq_length ,sq_cols):
    df_matrix = id_in_df[sq_cols].values
    num_elms = df_matrix.shape[0]
    for start, stop in zip(range(0, num_elms - sq_length), range(sq_length, num_elms)):
      yield df_matrix[start: stop, :]

  sq_cols = df1.columns[2:-4].tolist() + ['cycle_norm']
  seq_gen = (list(prepare_data(df1[df1['id']==id], sq_length,sq_cols)) for id in df1['id'].unique())
  seq_array = np.concatenate(list(seq_gen)).astype(np.float32)

  def prepare_data_labels(id_in_df,sq_length, label):
    df_matrix = id_in_df[label].values
    num_elms = df_matrix.shape[0]
    return df_matrix[sq_length:num_elms, :]
  label_gen = [prepare_data_labels(df1[df1['id']==id], sq_length,['label1']) for id in df1['id'].unique()]

  label_array = np.concatenate(label_gen).astype(np.float32)

  lstm = Sequential()
  lstm.add(LSTM(
              input_shape=(sq_length, seq_array.shape[2]),
              units=100,
              return_sequences=True))
  lstm.add(Dropout(0.2))
  lstm.add(LSTM(
            units=50,
            return_sequences=False))
  lstm.add(Dropout(0.2))
  lstm.add(Dense(units=label_array.shape[1]))
  lstm.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

  history = lstm.fit(seq_array, label_array, epochs=10, batch_size=200, validation_split=0.05, verbose=1)
  return lstm, history
           
def train_lstm_TTF(df1):
  sq_length = 50
  def prepare_data(id_in_df,sq_length ,sq_cols):
    df_matrix = id_in_df[sq_cols].values
    num_elms = df_matrix.shape[0]
    for start, stop in zip(range(0, num_elms - sq_length), range(sq_length, num_elms)):
      yield df_matrix[start: stop, :]
  df11 = df1.copy()
  sq_cols = df11.columns[2:-4].tolist() + ['cycle_norm']
  seq_gen = (list(prepare_data(df11[df11['id']==id], sq_length,sq_cols)) for id in df11['id'].unique())
  seq_array = np.concatenate(list(seq_gen)).astype(np.float32)

  def prepare_data_labels(id_in_df,sq_length, label):
    df_matrix = id_in_df[label].values
    num_elms = df_matrix.shape[0]
    return df_matrix[sq_length:num_elms, :]
  label_gen = [prepare_data_labels(df11[df11['id']==id], sq_length,['TTF']) for id in df11['id'].unique()]
  label_array = np.concatenate(label_gen).astype(np.float32)

  lstm = Sequential()
  lstm.add(LSTM(
              input_shape=(sq_length, seq_array.shape[2]),
              units=100,
              return_sequences=False))
  lstm.add(Dropout(0.25))
  lstm.add(Dense(units=label_array.shape[1]))
  lstm.add(Activation("linear"))
  lstm.compile(loss='mse', optimizer='adam', metrics=['mae'])

  history = lstm.fit(seq_array, label_array, epochs=70, batch_size=250, validation_split=0.08, verbose=1)
  return lstm, history

#_______________________________________________________

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('test1.html')

@app.route('/predict', methods=['POST'])
def predict():
    lstm_label = load_model('./lstm_for_label.h5')
    lstm_ttf = load_model('./lstm_for_ttf.h5')

    if request.method == 'POST':
        num_inputs = int(request.form['numInputs'])
        arr_inputs = [[0 for _ in range(15)] for _ in range(50)]
        arr_xg = []
        count = 0
        labels = ["setting1", "setting2", "s2", "s3", "s4", "s7", "s8", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]
        for i in range(num_inputs):
            lst = []
            for label in labels:
                input_name = f"{label}_{i+1}"
                value = float(request.form[input_name])
                lst.append(value)
            if count < num_inputs:
              arr_inputs[count-num_inputs] = lst
              arr_xg = np.array(lst)
              count+=1
        np_arrays_list = np.vstack([(sublist) for sublist in arr_inputs])
        array_final= np.expand_dims(np_arrays_list, axis=0)

        res = lstm_label.predict([array_final])
        res1 = lstm_ttf.predict([array_final])
        if res[0] > 0.5:
            prediction_result = "Negative (label = 1)"
        else:
            prediction_result = "Positive (label = 0)"

        result = f"Kết quả dự đoán: {prediction_result}; TTF = {res1[0,0]} "

        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)