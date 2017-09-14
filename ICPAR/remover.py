from keras.models import load_model
from trainer import read_1_file
import numpy as np

samples = ['70', '299']
if __name__=='__main__':
  model = load_model('net/CNN/2_CNN.net') # network
  for v in samples:
    x, y, tl = read_1_file(v+'.abp.rep.non.npy', 2) # load file
    pen = open('result/'+v+'.prd.sam.csv', 'w') # ar csv
    prd = model.predict(np.array(x))
    for i in range(len(prd)):
      pen.write(str(i)+ ',' + str(round(prd[i][0])) + '\n')
    pen.close()
  print('fin')
