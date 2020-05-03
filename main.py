from recognizer import recognizer
from Dataset import dataset
from  tqdm import tqdm
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt


dset = None
FacialRecognition = None

def SCI(x, k = 5):
    currMax = None
    for i in range(28):
        start = i * dset.nsamps
        check = np.zeros(dset.nsamps * 28)
        check[start:start + dset.nsamps] = np.ones(dset.nsamps)
        portion = np.sum(x * check)/np.sum(x)
        if(currMax == None or portion > currMax):
            currMax = portion
    return ((k * currMax) - 1)/(k - 1)
def fullTest(h,w):
    global dset
    global FacialRecognition
    dset = dataset('ExtendedYaleB/', trainsplit=.06, testsplit=.94, h = h, w = w)
    FacialRecognition = recognizer(dset)
    testSamples = len(dset.test)
    results = []
    with mp.Pool(mp.cpu_count()) as pool:
        for res in tqdm(pool.imap_unordered(trainingLoop, range(testSamples)),total=testSamples):
            if(res != -1):
                results.append(res)

    results = np.array(results)
    correct = np.sum(results)
    print('Correct:',correct)
    print('Total:', len(results))
    print('Accuracy:', correct / len(results))
    return correct / len(results)


def trainingLoop(s=0):

    global dset
    global FacialRecognition
    eps = .05
    tau = .5
    x = np.random.default_rng().standard_normal(dset.nsamps * 28)
    sample = dset.test[s]
    img = sample[1:]
    label = int(sample[0])
    FacialRecognition.solve(x, img, eps)

    xhat = FacialRecognition.getOptim()
    if(type(xhat) == type(None)):
        return 0
    #if(SCI(xhat) <= tau):
    #    return -1
    bestI = None
    bestRes = None
    for i in range(28):
        start = i * dset.nsamps
        check = np.zeros(dset.nsamps * 28)
        check[start:start + dset.nsamps] = np.ones(dset.nsamps)
        currRes = np.linalg.norm(img - (np.matmul(FacialRecognition.train, xhat * check)))
        if(bestI == None or currRes < bestRes):
            bestI = i
            bestRes = currRes
    if(bestI == label):
        return 1
    else:
        return 0



def all():

    tests = [(6,5), (9,7), (12,10), (20,17), (32,28)]

    accuracies = []
    for i in tests:
        print("Testing h =", i[0], "   w =", i[1])
        accuracies.append(fullTest(i[0], i[1]))

    print(accuracies)
    plt.figure()
    plt.plot(accuracies)
    plt.xlabel('Feature Dimension')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Feature Dimension')
    plt.show()

def reconstruction():
    global dset
    global FacialRecognition
    dset = dataset('ExtendedYaleB/', trainsplit=.06, testsplit=.94, h = 24, w=21)
    FacialRecognition = recognizer(dset)
    xhat = None
    while(type(xhat) == type(None)):

        choice = np.random.choice(len(dset.test))
        sample = dset.test[choice][1:]
        x = np.random.default_rng().standard_normal(dset.nsamps * 28)
        FacialRecognition.solve(x, sample, .05)
        xhat = FacialRecognition.getOptim()
        if(type(xhat) == type(None)):
            print("No solution")
            continue
        if(SCI(xhat) < .5):
            print("Solution found but not sparse")
            print(SCI(xhat))
            xhat = None
            continue
    recon = np.matmul(FacialRecognition.train, xhat)

    xhat /= np.amax(xhat)
    recon /= np.amax(recon)
    plt.figure()
    plt.subplot(121)
    plt.imshow(sample.reshape((24,21)), cmap='gray')
    plt.subplot(122)
    plt.imshow(recon.reshape((24,21)), cmap='gray')
    plt.show()

#reconstruction()
#fullTest(12,10)
