import numpy as np
import pandas as pd
import math

df = pd.read_csv('Fiore Sample Data.csv') #you can download data here: https://docs.google.com/spreadsheets/d/1rokrBuH9UD_9xSKmKwJsygBUQ__2brWIl61KqUrOuDM/edit?gid=0#gid=0

def get_data(df):
    df = df.groupby(['Date', 'Keyword'])['Metric'].sum().reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    res = df.pivot(index='Date', columns='Keyword', values='Metric').sort_values(by='Date')
    return res


class UCBAlgorithm:
    def __init__(self, N, data, normalize=True):
        self.N = N #Number of days used. The shorter the timeframe, the quicker the algorithm makes decisions, but it also increases the likelihood of errors.
        self.data = data #data
        self.d = data.shape[1] #number of options (keywords)
        self.normalize_data = normalize #Normalizing speeds up the algorithm's ability to select winners, but it also raises the chances of declaring too early.
        self.keywords_selected = 0

    def normalize(self):
        return self.data.apply(lambda x: (x>self.data.mean(axis=1))*1)

    def run(self):
        keywords_selected = []
        numbers_of_selections = [0] * self.d
        sums_of_reward = [0] * self.d
        total_reward = 0

        if self.normalize_data:
            data = self.normalize()
        else:
            data = self.data

        for n in range(0, self.N):
            kw = 0
            max_upper_bound = 0
            for i in range(0, self.d):
                if (numbers_of_selections[i] > 0):
                    average_reward = sums_of_reward[i] / numbers_of_selections[i]
                    delta_i = math.sqrt(2 * math.log(n+1) / numbers_of_selections[i])
                    upper_bound = average_reward + delta_i
                else:
                    upper_bound = 1e400
                if upper_bound > max_upper_bound:
                    max_upper_bound = upper_bound
                    kw = i
            keywords_selected.append(kw)
            numbers_of_selections[kw] += 1
            reward = data.values[n, kw]
            sums_of_reward[kw] += reward
            total_reward += reward
        self.keywords_selected = keywords_selected

    def get_proportions(self):
        result = pd.Series(self.keywords_selected).value_counts(normalize=True)
        return result

if __name__ == '__main__':
    df = pd.read_csv('Fiore Sample Data.csv') #you can download data here: https://docs.google.com/spreadsheets/d/1rokrBuH9UD_9xSKmKwJsygBUQ__2brWIl61KqUrOuDM/edit?gid=0#gid=0
    res = get_data(df)
    x = UCBAlgorithm(N=7, data=res)
    x.run()
    print(x.get_proportions())
