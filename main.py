from recognizer import recognizer
from Dataset import dataset
import numpy as np
from tqdm import tqdm



dataset = dataset('ExtendedYaleB/', trainsplit=.05, testsplit=.95)

print(dataset.nsamps)
FacialRecognition = recognizer(dataset)

correct = 0
for i in tqdm(range(len(dataset.test))):

    x = np.random.default_rng().standard_normal(dataset.nsamps * 28)
    sample = dataset.test[i]
    img = sample[1:]
    label = int(sample[0])
    FacialRecognition.solve(x, img)

    xhat = FacialRecognition.getOptim()
    bestI = None
    bestRes = None
    for i in range(28):
        start = i * dataset.nsamps
        check = np.zeros(dataset.nsamps * 28)
        check[start:start + dataset.nsamps] = np.ones(dataset.nsamps)
        currRes = np.linalg.norm(img - (np.matmul(FacialRecognition.train, FacialRecognition.getOptim() * check)))
        if(bestI == None or currRes < bestRes):
            bestI = i
            bestRes = currRes
    if(bestI == label):
        correct += 1

print('Correct:',correct)
print('Total:', len(dataset.test))
print('Accuracy:', correct / len(dataset.test))
