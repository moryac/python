from conf.conf import logging
from conf.conf import settings
from model.prediction import init_model, predict
import os

if not (os.path.exists(settings.MODEL['REG'])):
    init_model('REG')

prediction = predict(values=settings.PREDICTION.VALUES, path_to_model=settings.MODEL.RANDOM)
logging.info(f'prediction: {prediction}')


