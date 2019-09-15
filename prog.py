#! env/bin/python3

#Khazem Khaled 'xlincw0w'

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split
from pandas import read_csv

class neural_net():
    def __init__(self):

        self.learning_rate = 0.1
        
        self.l1 = tf.keras.layers.Dense(units=16,
                                       input_shape=(10,),
                                       activation='relu')

        self.l2 = tf.keras.layers.Dense(units=16,
                                       activation='relu')

        self.l3 = tf.keras.layers.Dense(units=16,
                                       activation='relu')

        self.l4 = tf.keras.layers.Dense(units=2,
                                       activation='softmax')

        self.model = tf.keras.Sequential([self.l1, self.l2, self.l3, self.l4])
        self.model.compile(loss='sparse_categorical_crossentropy',
                     optimizer=tf.keras.optimizers.Adagrad(self.learning_rate),
                     metrics=['accuracy'])

    def train(self, features, labels):
        self.history = self.model.fit(features, labels, epochs=100)

    def evaluate(self, features, labels):
        results = self.model.evaluate(features, labels)
        return results

    def predict(self, inputs):
        return self.model.predict(inputs)
        
    def plotloss(self):
        plt.xlabel('Epoch Number')
        plt.ylabel('Loss Magnitude')
        plt.plot(self.history.history['loss'])
        plt.show()
        
    def save_model(self, accuracy):
        if accuracy > 0.92:
            pickle.dump(self.model, open('model.pickle', 'wb'))
            print('Model file saved.')
        else:
            print('Accuracy wasnt high enough.')

    def load_model(self):
        self.model = pickle.load(open('model.pickle', 'rb'))
                   

def numerize_data(data):
  data = data.replace([2016, 2017, 2018], [0, 0, 1])
  data = data.replace(['Kenya', 'Rwanda', 'Tanzania', 'Uganda', 'Yes', 'No'], [0, 1, 2, 3, 1, 0])
  data = data.replace(['Rural', 'Urban', 'Male', 'Female'], [0, 1, 0, 1])
  data = data.replace(['Spouse', 'Head of Household', 'Other relative', 'Child', 'Parent', 'Other non-relatives'], [1, 2, 1, 1, 0, 0])
  data = data.replace(['Married/Living together', 'Widowed', 'Single/Never Married', 'Divorced/Seperated', 'Dont know'], [1, 0, 0, 0, 1])
  data = data.replace(['Secondary education', 'No formal education', 'Vocational/Specialised training', 'Primary education', 'Tertiary education', 'Other/Dont know/RTA'], [2, 0, 3, 1, 3, 2])
  data = data.replace(['Self employed', 'Government Dependent', 'Formally employed Private', 'Informally employed', 'Formally employed Government', 'Farming and Fishing', 'Remittance Dependent', 'Other Income', 'Dont Know/Refuse to answer', 'No Income'], [1, 2, 3, 0, 4, 0, 0, 1, 1, 0])
  return data


if __name__ == '__main__':
                                       
    submissionf = read_csv('data/SubmissionFile.csv')
    vardef = read_csv('data/VariableDefinitions.csv')
    train_origin = read_csv('data/Train_v2.csv')
    test = read_csv('data/Test_v2.csv')

    train_ye = train_origin[train_origin['bank_account'] == 'Yes']
    #train_no = train_origin[train_origin['bank_account'] == 'No'].sample(3312)
    train_no = train_origin[train_origin['bank_account'] == 'No']
    train_full = pd.DataFrame.append(train_ye, train_no)
    train_full.sample(frac=1).reset_index(drop=True)

    train_full = numerize_data(train_full)

    features = train_full.drop(columns=['bank_account', 'uniqueid', 'household_size'])
    labels = train_full.pop('bank_account')

    train_feat, test_feat, train_lab, test_lab = train_test_split(features, labels, test_size=0.1)

    nn = neural_net()
    nn.train(train_feat.values, train_lab.values)

    print()
    print('Evaluation : ')
    res = nn.evaluate(test_feat.values, test_lab.values)

    #Uncomment to plot loss history
    #nn.plotloss()

    print()
    print('Resultat : ', res)

    #nn.save_model(res[1])

    print()
    print('Do you want to generate the Submission file ? { y : yes } : ')
    generate = input('... : ')
    
    if generate == 'y':

        test = numerize_data(test)
        test_data = test.drop(columns=['uniqueid', 'household_size'])
        uniqueids = test.pop('uniqueid')
        predict = nn.predict(test_data)
        predict = predict.argmax(1)
        dfpred = pd.DataFrame({'bank_account':predict})
        test_data['bank_account'] = dfpred
        test_data = test_data.replace({'country': [0, 1, 2, 3]}, {'country': ['Kenya', 'Rwanda', 'Tanzania', 'Uganda'] })
        test_data['uniqueid'] = uniqueids + ' x ' + test_data['country']
        result = test_data[['uniqueid', 'bank_account']]

        print()
        print('Generating the file ...')
        for uniqueid in submissionf['uniqueid']:
          index = result.index[result['uniqueid'] == uniqueid].tolist()[0]
          bank = result.iloc[index]['bank_account']
          submissionf.at[submissionf.index[submissionf['uniqueid'] == uniqueid].tolist()[0], 'bank_account'] = bank

        #submissionf = submissionf.replace({'bank_account': [0.0, 1.0]}, {'bank_account': ['Yes', 'No']})

        submissionf['bank_account'] = submissionf['bank_account'].astype(int)

        submissionf.to_csv('SubmissionFile.csv', index=False) 
        print('File generated !')
