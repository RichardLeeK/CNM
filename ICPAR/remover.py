from keras.models import load_model
from trainer import read_1_file

if __name__=='__main__':
  model = load_model('') # network
  x, y, tl = read_1_file('') # load file
  pen = open('') # ar csv
  prd = model.predict(x)
  for i in range(len(prd)):
    pen.write(str(i)+ ',' + str(round(prd[i][0])) + '\n')
  pen.close()
  print('fin')
