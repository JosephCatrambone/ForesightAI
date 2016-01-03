#!/usr/bin/enc python
# jo.jcat@gmail.com

import sys, os
import csv
import pickle as pickle

import numpy
import pandas
from neuralnetwork import neuralnetwork as nn

TRAINING_CSV = "./numerai_training_data.csv"
TEST_CSV = "./numerai_tournament_data.csv"
OUTPUT_CSV = "./prediction.csv"

def expand_dataframe_columns(df):
	# Cut off class data.
	df['c1'] = df.c1.apply(lambda x: int(str(x)[3:]))
	# Split c1 into many rows.
	for i in range(2, 25):
		df["c{}".format(i)] = 0
		df.ix[df.c1 == i, 'c{}'.format(i)] = 1;
	# Squish c1 into 1/0 like the rest.
	df.ix[df.c1 != 1, 'c1'] = 0;


def load_data(training_csv):
	cin = pandas.read_csv(training_csv)
	df = pandas.DataFrame(cin)
	# Expand classes into one-hot.
	expand_dataframe_columns(df)
	# Split off training + validation.
	training_data = df.ix[df.validation == 0]
	validation_data = df.ix[df.validation == 1]
	t_x = training_data[["f{}".format(x) for x in range(1,15)] + ['c{}'.format(x) for x in range(1, 25)]]
	v_x = validation_data[["f{}".format(x) for x in range(1,15)] + ['c{}'.format(x) for x in range(1, 25)]]
	t_y = training_data[['target']]
	v_y = validation_data[['target']]
	return numpy.asarray(t_x), numpy.asarray(t_y), numpy.asarray(v_x), numpy.asarray(v_y) # Most fields are int. What happens if we force float?


def predict_and_save(model, target_csv, output_csv):
	cin = pandas.read_csv(target_csv)
	df = pandas.DataFrame(cin)
	expand_dataframe_columns(df)
	df.index = df['t_id']
	df['probability'] = model.predict(numpy.asarray(df)[:,1:])
	pred_norm = (df[['probability']] - df[['probability']].min())/(df[['probability']].max() - df[['probability']].min())
	#df['prediction'] = df.apply(lambda x : model.predict(numpy.atleast_2d(numpy.asarray(x))))
	# I'm making a copy for normalization.  If we assign it to df, then use this.
	#df[['prediction', ]].to_csv(output_csv) # Add 't_id' if the df.index is changed above.
	pred_norm[['probability', ]].to_csv(output_csv)
	print("Saved {}.  Don't forget to wrap header in quotes and remove newline commas.".format(output_csv))


def train(training_data, training_labels, validation_data, validation_labels):
	model = nn.NeuralNetwork([training_data.shape[1], 1024, 64, 128, training_labels.shape[1]], ['linear', 'tanh', 'tanh', 'tanh', 'tanh'], weight_range=1.0)

	def status_report(iteration, error):
		pred = model.predict(validation_data)
		diff = numpy.sum(numpy.abs((validation_labels - pred)))
		guess_acc = numpy.sum(numpy.abs((validation_labels - ((pred > pred.mean()) * 1.0))))/pred.shape[0]
		print("Iteration: {}\tBatch error: {}\tValidation error: {}\tValidation2 error: {}".format(iteration, error, diff, guess_acc))
		save_model(model, "checkpoint.model.pickle")

	model.fit(training_data, training_labels, learning_rate=0.1, momentum=0.9, batch_size=500, epochs=100000, update_every=500, update_func=status_report) 
	return model


def save_model(model, filename="model.pickle"):
	fout = open(filename, 'w')
	pickle.dump(model, fout, -1)
	fout.close()
	# Also save weights separately?


def main():
	# Load
	train_x, train_y, validation_x, validation_y = load_data(TRAINING_CSV)
	# Train
	model = train(train_x, train_y, validation_x, validation_y)
	# Save
	save_model(model)
	# Predict
	predict_and_save(model, TEST_CSV, OUTPUT_CSV)


if __name__=="__main__":
	main()
