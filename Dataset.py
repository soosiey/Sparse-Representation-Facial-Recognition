import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split as tts



class dataset():

    def __init__(self,root_dir,trainsplit=.3, testsplit=.7):
        self.train = []
        self.test = []
        s = 0
        for dir in tqdm(os.listdir(root_dir)):
            if(dir[0] == '.'):
                continue
            subjectpics = []
            for f in os.listdir('ExtendedYaleB/'+dir+'/'):
                if(f[-3:] == 'pgm' and f[-5] != 't'):
                    check = np.array(Image.open('ExtendedYaleB/' + dir + '/' + f).resize((12,10))).flatten().astype(float)
                    check /= np.linalg.norm(check)
                    r = np.zeros(len(check) + 1)

                    r[0] = s
                    r[1:] = check
                    subjectpics.append(r)
            s += 1
            ttrain,ttest = tts(subjectpics,test_size=testsplit,train_size=trainsplit)
            self.train.append(ttrain)
            self.test.append(ttest)

        self.train = np.array(self.train)
        self.test = np.array(self.test)
        self.nsamps = self.train.shape[1]
        self.train = self.train.reshape((-1, 121))
        self.test = self.test.reshape((-1,121))
        np.random.shuffle(self.test)

