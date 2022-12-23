from conf.conf import logging
from conf.conf import settings
from model.prediction import init_model, predict
import os
import argparse

parser = argparse.ArgumentParser(description='Choose model and arguments')
parser.add_argument('-m', '--model', choices=settings.MODEL, default='RANDOM', type=str, help="Choose the model for the prediction")
parser.add_argument('-v', '--values', default=settings.PREDICTION.VALUES, type=str, nargs='+', help="Enter values for prediction with spaces, number by number")
args = parser.parse_args()


if not (os.path.exists(settings.MODEL[args.model])):
    init_model(args.model)

prediction = predict(values=settings.PREDICTION.VALUES, path_to_model=settings.MODEL.RANDOM)
logging.info(f'prediction: {prediction}')


