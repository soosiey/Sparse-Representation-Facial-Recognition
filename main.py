from recognizer import recognizer
from Dataset import dataset
from  tqdm import tqdm
import numpy as np
import multiprocessing as mp


dataset = dataset('ExtendedYaleB/', trainsplit=.01, testsplit=.99)
FacialRecognition = recognizer(dataset)

def trainingLoop(s, dataset=dataset, FacialRecognition=FacialRecognition):



    #correct = 0
    eps = .5
    #for s in tqdm(range(len(dataset.test))):

    x = np.random.default_rng().standard_normal(dataset.nsamps * 28)
    sample = dataset.test[s]
    img = sample[1:]
    label = int(sample[0])
    FacialRecognition.solve(x, img, eps)

    xhat = FacialRecognition.getOptim()
    bestI = None
    bestRes = None
    for i in range(28):
        start = i * dataset.nsamps
        check = np.zeros(dataset.nsamps * 28)
        check[start:start + dataset.nsamps] = np.ones(dataset.nsamps)
        currRes = np.linalg.norm(img - (np.matmul(FacialRecognition.train, xhat * check)))
        if(bestI == None or currRes < bestRes):
            bestI = i
            bestRes = currRes
    if(bestI == label):
        return 1
        #correct += 1
    else:
        return 0
    #return correct

results = []
with mp.Pool(mp.cpu_count()) as pool:
    for res in tqdm(pool.imap_unordered(trainingLoop, range(len(dataset.test))),total=len(dataset.test)):
        results.append(res)

results = np.array(results)
correct = np.sum(results)
#correct = trainingLoop(dataset, FacialRecognition)
print('Correct:',correct)
print('Total:', len(dataset.test))
print('Accuracy:', correct / len(dataset.test))
