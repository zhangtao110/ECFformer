import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

cls = MinMaxScaler()

class DataPretreatment:
    def __init__(self,dir_path):
        self.dir_path = dir_path

    def read_files_in_directory(self, dir_path):

        for item in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item)
            if os.path.isfile(item_path):
                if item.endswith('.csv'):
                    self._handle_files(item, item_path)
            elif os.path.isdir(item_path):
                if item in ['ETT-small', 'weather', 'exchange_rate', 'electricity', 'traffic']:
                    self.read_files_in_directory(item_path)

    def _handle_files(self, file_name, file_path):
        print('file_name', file_name)
        columns_to_read = lambda col: col != 'date'
        df = pd.read_csv(file_path, usecols=columns_to_read, encoding='UTF-8')
        data_array = df.values
        data_array = cls.fit_transform(data_array)
        data_mean = data_array.mean()
        data_variance = data_array.std()
        print('打印均值', data_mean)
        print('打印方差', data_variance)

if __name__ == "__main__":

    dir_path = '../dataset/'
    data_pretreatment = DataPretreatment(dir_path)
    data_pretreatment.read_files_in_directory(dir_path)