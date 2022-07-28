from sklearn.neural_network import MLPClassifier
from keras.layers import Activation, Dropout, Flatten, Dense, Input, LeakyReLU
from keras.layers import BatchNormalization, concatenate, multiply, Embedding
from keras.models import Model, Sequential
from sklearn.cluster import DBSCAN
from os.path import join
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, RandomTreesEmbedding

from Ciena_NSL_KDD import initial_ML_model
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from utility import *
from Utils import *
import random
from CISOSE_demo import add_resample_data_test_MCS

latent_dim = 50#100

n_features = 2
n_classes = 3
n_samples = 1000


def build_discriminator(optimizer='SGD'):

    features = Input(shape=(n_features,))
    label = Input(shape=(1,), dtype='int32')
    label_embedding = Flatten()(Embedding(n_classes, n_features)(label))

    inputs = multiply([features, label_embedding])

    x = Dense(512)(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.2)(x)
    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.2)(x)

    valid = Dense(1, activation='sigmoid')(x)

    model = Model([features, label], valid)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', 'TruePositives',
                                                                  'TrueNegatives','FalsePositives','FalseNegatives'])
    model.summary()

    return model


def build_generator():

    noise = Input(shape=(latent_dim,))
    label = Input(shape=(1,), dtype='int32')

    label_embedding = Flatten()(Embedding(n_classes, latent_dim)(label))

    inputs = multiply([noise, label_embedding])

    x = Dense(256)(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(1024)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)

    features = Dense(n_features, activation='tanh')(x)

    model = Model([noise, label], features)
    model.summary()

    return model


def build_gan(generator, discriminator, optimizer='SGD'):

    noise = Input(shape=(latent_dim,))
    label = Input(shape=(1,))

    features = generator([noise, label])
    valid = discriminator([features, label])
    discriminator.trainable = False

    model = Model([noise, label], valid)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()

    return model


def get_random_batch(X, y, batch_size):

    idx = np.random.randint(0, len(X))

    X_batch = X[idx:idx + batch_size]
    y_batch = y[idx:idx + batch_size]
    y_batch = np.array(y_batch)
    print('label data type', type(y_batch))
    print('x after get_random_batch data type', type(X_batch))
    X_batch = np.reshape(X_batch, (batch_size, -1))
    return X_batch, y_batch


def train_gan(gan, generator, discriminator,
              X, y,
              n_epochs=1000, batch_size=32,
              hist_every=10, log_every=10,
              index=None):

    acc_real_hist = []
    acc_fake_hist = []
    acc_gan_hist = []
    loss_real_hist = []
    loss_fake_hist = []
    loss_gan_hist = []

    for epoch in range(n_epochs):
        # X_batch, labels = get_all_batch(X, y, batch_size)
        X_batch, labels = get_random_batch(X, y, batch_size)

        # train with real values
        y_real = np.ones((X_batch.shape[0], 1))
        x = [X_batch, labels]
        loss_real, acc_real,_,_,_,_ = discriminator.train_on_batch([X_batch, labels], y_real)
        # discriminator.fit([X_batch, labels], y_real, epochs=20)

        # train with fake values
        noise = np.random.uniform(0, 1, (labels.shape[0], latent_dim))
        X_fake = generator.predict([noise, labels])
        y_fake = np.zeros((X_fake.shape[0], 1))
        # print('y_fake ', y_fake)
        # print('fake y label ', y_fake)
        loss_fake, acc_fake,_,_,_,_ = discriminator.train_on_batch([X_fake, labels], y_fake)
        # if (epoch + 1) % hist_every == 0:
        #     fake_score = discriminator.test_on_batch([X_fake, labels], y_fake)
        #     print('discriminator evaluate score for generated samples',fake_score)
        #
        #     score_ori = discriminator.evaluate(testarg[0], testarg[1])
        #     print(' All task evaluation score', score_ori)
        #
        #     score_ori = discriminator.evaluate(testarg[2], testarg[3])
        #     print('Only fake in original dataset evaluation score', score_ori)

        y_gan = np.ones((labels.shape[0], 1))
        # print('y_gan ', y_gan)

        loss_gan, acc_gan = gan.train_on_batch([noise, labels], y_gan)
        # gan.fit([noise, labels], y_gan)
        # if (epoch + 1) % (500*hist_every) == 0:
        #     plottaskwithlocation_with_data(X_batch, labels)
        #     plottaskwithlocation_with_data(X_fake, y_fake)


        if (epoch + 1) % hist_every == 0:
            acc_real_hist.append(acc_real)
            acc_fake_hist.append(acc_fake)
            acc_gan_hist.append(acc_gan)
            loss_real_hist.append(loss_real)
            loss_fake_hist.append(loss_fake)
            loss_gan_hist.append(loss_gan)

        if (epoch + 1) % log_every == 0:
            lr = 'loss real: {:.3f}'.format(loss_real)
            ar = 'acc real: {:.3f}'.format(acc_real)
            lf = 'loss fake: {:.3f}'.format(loss_fake)
            af = 'acc fake: {:.3f}'.format(acc_fake)
            lg = 'loss gan: {:.3f}'.format(loss_gan)
            ag = 'acc gan: {:.3f}'.format(acc_gan)

            print('{}, {}, {} | {}, {} | {}, {}'.format((epoch+1), lr, ar, lf, af, lg, ag))

    return loss_real_hist, acc_real_hist, loss_fake_hist, acc_fake_hist, loss_gan_hist, acc_gan_hist


def generate_samples(class_for, n_samples=20):
    noise = np.random.uniform(0, 1, (n_samples, latent_dim))
    label = np.full((n_samples,), fill_value=class_for)
    return generator.predict([noise, label])


def visualize_fake_features(fake_features, figsize=(15, 6), color='r'):
    ax, fig = plt.subplots(figsize=figsize)

    # Let's plot our dataset to compare
    for i in range(n_classes):
        plt.scatter(scaled_X[:, 0][np.where(y == i)], scaled_X[:, 1][np.where(y == i)])

    plt.scatter(fake_features[:, 0], fake_features[:, 1], c=color)
    plt.title('Real and fake features')
    plt.legend(['Class 0', 'Class 1', 'Class 2', 'Fake'])
    plt.show()


def pick_samples_based_on_class(X, y, classname1, classname2, classname3=-1):
    indexlist= []
    if classname2 == -1:
        print('Please select correct class')
        return
    for i in range(0, len(y)):
        if y[i] == classname1 or y[i] == classname2 or y[i] == classname3:
            indexlist.append(i)
            # print(i)
    selected_x = None
    selected_y = None

    selected_x = np.take(X, indexlist,0)
    selected_y = np.take(y, indexlist,0)

    return selected_x, selected_y




def add_resample_data_test(U2Rfolder=None, R2Lfolder = None, Probefolder = None):
    U2RID = 4
    R2LID = 3
    ProbeID = 2
    folder = 'dataset/NSL-KDD/'
    x_train, x_test, y_train, y_test = loadNslKdd(folder)
    target_names = ['DoS', 'Normal', 'Probe', 'R2L', 'U2R']

    attack_dict = {target_names[i]: i for i in range(0, len(target_names))}
    y_train = [attack_dict[item] for item in y_train]
    y_test = [attack_dict[item] for item in y_test]

    scaler = MinMaxScaler(feature_range=(-1, 1))

    scaled_X_train = scaler.fit_transform(x_train)
    scaled_X_test = scaler.transform(x_test)

    x_train_over = scaled_X_train
    y_train_over = y_train
    if U2Rfolder !=None:
        x_oversamplingSet = pd.read_csv(U2Rfolder)
        y_oversamplingSet = np.full((x_oversamplingSet.shape[0], 1), U2RID)
        x_train_over = np.concatenate((scaled_X_train, x_oversamplingSet), axis=0)
        # y_train_over = y_train
        for i in range(0, np.shape(y_oversamplingSet)[0]):
            y_train_over.append(y_oversamplingSet[i][0])
    if R2Lfolder!=None:
        x_oversamplingSet_R2L = pd.read_csv(R2Lfolder)
        y_oversamplingSet_R2L = np.full((x_oversamplingSet_R2L.shape[0], 1), R2LID)
        x_train_over = np.concatenate((x_train_over, x_oversamplingSet_R2L), axis=0)

        for i in range(0, np.shape(y_oversamplingSet_R2L)[0]):
            y_train_over.append(y_oversamplingSet_R2L[i][0])
    if Probefolder!=None:
        x_oversamplingSet_Pro = pd.read_csv(Probefolder)
        y_oversamplingSet_Pro = np.full((x_oversamplingSet_Pro.shape[0], 1), ProbeID)
        x_train_over = np.concatenate((x_train_over, x_oversamplingSet_Pro), axis=0)

        for i in range(0, np.shape(y_oversamplingSet_Pro)[0]):
            y_train_over.append(y_oversamplingSet_Pro[i][0])

    print(sorted(Counter(y_train_over).items()))
    return x_train_over, y_train_over, scaled_X_test, y_test



def loaddigitsfromcsv(filename):
    data = np.loadtxt(filename,
                      delimiter=',')
    return data


def initial_ML_model(mlmodel):
    configstr=None
    if mlmodel == 'Adaboost':
        nestimator = 100
        clf = AdaBoostClassifier(n_estimators=nestimator)
        configstr = mlmodel + " estimator: " + str(nestimator)

    elif mlmodel == 'RF':
        nestimator = 200
        max_dep = 30

        clf = RandomForestClassifier(n_estimators=nestimator, max_depth=max_dep, random_state=0)
        configstr = mlmodel + " estimator: " + str(nestimator) + 'max_depth: ' + str(max_dep)
    elif mlmodel == 'KNN':
        n_nei = 1
        clf = KNeighborsClassifier(n_neighbors=n_nei)
        configstr = mlmodel + " n_neighbors: " + str(n_nei)
    elif mlmodel == 'SVM':
        clf = SVC(decision_function_shape='ovo')
        configstr = mlmodel + " decision_function_shape " + 'ovo'
    elif mlmodel == 'MLP':
        al = 0.01
        acvfun = 'relu'
        maxit = 500
        clf = MLPClassifier(solver='lbfgs', alpha=0.01, activation='relu', hidden_layer_sizes=(100, 100, 100),
                            max_iter=500, random_state=1)

        configstr = mlmodel + " solver: " + 'lbfgs; ' + ' alpha: ' + str(
            al) + '; activation function: ' + acvfun + '; hidden_layer_structure: ' + ' 100, 100, 100' + '; max_iter: ' + str(
            maxit)
    elif mlmodel == 'DT':
        clf = tree.DecisionTreeClassifier(random_state=7)
        # clf = clf.fit(X, Y)
    elif mlmodel == 'NB':
        clf = GaussianNB()
    else:
        print("No machine learning model is initiated!!!")
        return

    return clf, configstr


def MCS_train_with_small_amount_DT(mlmodel, x_train, y_train, x_test, y_test, numofnoise=None):
    clf, configstr = initial_ML_model(mlmodel)
    if numofnoise!=None:
        noise_x = np.random.uniform(-1, 1, (numofnoise, 7))
        noise_y = np.full((noise_x.shape[0], 1), 1)

        x_train = np.concatenate((x_train, noise_x), axis=0)
        y_train_over = np.ndarray.tolist(y_train)
        for i in range(0, np.shape(noise_y)[0]):
            y_train_over.append(noise_y[i][0])

        y_train = y_train_over

    clf.fit(x_train, y_train)
    print('Training dataset', sorted(Counter(y_train).items()))
    y_pred = clf.predict(x_test)

    result = classification_report(y_test, y_pred, digits=4)
    print('Test results without adding GAN samples: ', result)
    myconfusionmatrix = np.array2string(confusion_matrix(y_test, y_pred))
    print(myconfusionmatrix)
    return y_pred


def MCS_add_noise_in_testing_dataset(mlmodel, x_train, y_train, x_test, y_test, numofnoise=None):
    clf, configstr = initial_ML_model(mlmodel)
    if numofnoise!=None:
        noise_x = np.random.uniform(-1, 1, (numofnoise, 7))
        noise_y = np.full((noise_x.shape[0], 1), 1)

        x_test = np.concatenate((x_test, noise_x), axis=0)
        y_test_over = np.ndarray.tolist(y_test)
        for i in range(0, np.shape(noise_y)[0]):
            y_test_over.append(noise_y[i][0])

        y_test = y_test_over


    clf.fit(x_train, y_train)
    print('Testing dataset with noise samples', sorted(Counter(y_test).items()))
    y_pred = clf.predict(x_test)

    result = classification_report(y_test, y_pred, digits=4)
    print('Test results without adding GAN samples: ', result)
    myconfusionmatrix = np.array2string(confusion_matrix(y_test, y_pred))
    print(myconfusionmatrix)
    return x_test, y_test, y_pred


def add_fake_sampling_for_testing(mlmodel, scaled_X, y_train, scaled_X_test, y_test, oversamfile):
    scaled_X_test_new, y_test_new = add_resample_data_test_MCS(scaled_X_test, y_test, oversamfile)
    clf, configstr = initial_ML_model(mlmodel)

    clf.fit(scaled_X, y_train)

    print('After adding fake tasks in TESTING dataset', sorted(Counter(y_test_new).items()))
    y_pred = clf.predict(scaled_X_test_new)

    result = classification_report(y_test_new, y_pred, digits=4)
    print('Test samples results adding oversampling data ', result)
    myconfusionmatrix = np.array2string(confusion_matrix(y_test_new, y_pred))

    print(myconfusionmatrix)
    return scaled_X_test_new, y_test_new, y_pred


def MCS_train_with_small_amount_DT_Oversampling(mlmodel, scaled_X, y_train, scaled_X_test, y_test, oversamfile):

    x_train_over, y_train_over = add_resample_data_test_MCS(scaled_X, y_train, oversamfile)
    clf, configstr = initial_ML_model(mlmodel)

    clf.fit(x_train_over, y_train_over)

    print('After adding fake tasks in training dataset', sorted(Counter(y_train_over).items()))
    y_pred = clf.predict(scaled_X_test)

    result = classification_report(y_test, y_pred, digits=4)
    print('Test samples results adding oversampling data ', result)
    myconfusionmatrix = np.array2string(confusion_matrix(y_test, y_pred))

    print(myconfusionmatrix)
    return y_pred


def MCS_CISOSE_Code():
    ########################MCS Dataset augumentation#############
    # targetname = ['Legitmate', 'Fake']
    x_train = loaddigitsfromcsv('x_train_set.csv')
    x_test = loaddigitsfromcsv('x_test_set.csv')
    y_train = loaddigitsfromcsv('y_train_set_CISOSE.csv')
    y_test = loaddigitsfromcsv('y_test_set_CISOSE.csv')

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_X_ori = scaler.fit_transform(x_train)

    scaled_X = scaler.transform(x_train)
    scaled_X_test = scaler.transform(x_test)

    n_features = 7
    n_classes = 2
    n_samples = np.shape(x_train)[0]

    generator = build_generator()
    discriminator = build_discriminator()
    mygan = build_gan(generator, discriminator)
    loss_real_hist, acc_real_hist, \
    loss_fake_hist, acc_fake_hist, \
    loss_gan_hist, acc_gan_hist = \
        train_gan(mygan, generator, discriminator, scaled_X, y_train, n_epochs=1500, batch_size=20, log_every=10,
                  hist_every=10)
    for i in range(10, 11):
        numofsample = 10 * i

        ax, fig = plt.subplots(figsize=(15, 6))
        plt.plot(loss_real_hist)
        plt.plot(loss_fake_hist)
        plt.plot(loss_gan_hist)
        plt.xlabel('Epoch (every 10 epochs)')
        plt.ylabel('Loss')
        # plt.title('Training loss over time')
        plt.legend(['Loss real', 'Loss fake', 'Loss GAN'], fontsize=15)
        # plt.legend(['Loss real', 'Loss fake'])

        plt.show()

        ax2, fig2 = plt.subplots(figsize=(15, 6))
        plt.plot(acc_real_hist)
        plt.plot(acc_fake_hist)
        plt.plot(acc_gan_hist)
        plt.title('Training accuracy over time')
        plt.legend(['Acc real', 'Acc fake', 'Acc GAN'])
        # plt.legend(['Acc real', 'Acc fake'])

        plt.show()


def get_all_batch(X, y, batch_size):

    idx = 0

    X_batch = X[idx:idx + batch_size]
    y_batch = y[idx:idx + batch_size]
    y_batch = np.array(y_batch)
    # print('label data type', type(y_batch))
    # print('x after get_random_batch data type', type(X_batch))
    return X_batch, y_batch


def get_one_sample(X, y, batch_size):

    idx = 0
    batch_size = 1
    X_batch = X[idx:idx + batch_size]
    y_batch = y[idx:idx + batch_size]
    y_batch = np.array(y_batch)
    # print('label data type', type(y_batch))
    # print('x after get_random_batch data type', type(X_batch))
    return X_batch, y_batch


def showfig(index, loss_real_hist, loss_fake_hist, loss_gan_hist, acc_real_hist, acc_fake_hist, acc_gan_hist):

    ax, fig = plt.subplots(figsize=(15, 6))
    plt.plot(loss_real_hist)
    plt.plot(loss_fake_hist)
    plt.plot(loss_gan_hist)
    plt.xlabel('Epoch (every 10 epochs)', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    # plt.title('Training loss over time')
    plt.legend(['Loss real', 'Loss fake', 'Loss GAN'], fontsize=15)
    plt.show()
    plt.savefig('logs/MCS/'+str(index)+'/loss.png')


    ax2, fig2 = plt.subplots(figsize=(15, 6))
    plt.plot(acc_real_hist)
    plt.plot(acc_fake_hist)
    plt.plot(acc_gan_hist)
    # plt.title('Training accuracy over time')
    plt.legend(['Acc real', 'Acc fake', 'Acc GAN'], fontsize=15)
    plt.show()
    plt.savefig('logs/MCS/'+str(index)+'/accuracy.png')

def predict_samples(model, X,Y):
    # _,evals,_,_,_ = [model.evaluate(X[i:i + 1], Y[i:i + 1]) for i in range(len(X))]
    _,evals,_,_,_ = [model.evaluate(X[i:i + 1], Y[i]) for i in range(len(X))]

    return evals


def plottaskwithlocation():
    filepath = os.getcwd()
    data = loaddigitsfromcsv('x_train_set.csv')

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(data)

    # data = loadcsvasnparray(csvfile)

    # data = data[1:]
    x = scaled_data[0:, 0].astype(np.float, copy=False) #latitude

    y = scaled_data[0:, 1].astype(np.float, copy=False) #logitude

    # isleg = data[:, 11]
    markercol = ['r', 'k']
    area = np.pi
    plt.scatter(x[:12588], y[:12588], c=markercol[1], s=area)

    plt.scatter(x[12588:], y[12588:], c=markercol[0], s=area)

    # plt.scatter(x, y)

    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    # plt.show()
    # plt.title("14K clarence-rockland data set")

    savefile = join(filepath, 'logs', '14k_Clarence_Rockland_tasks_distribution.png')
    plt.legend()

    plt.savefig(savefile)
    plt.show()


def read_pred_result(mlmodel,index):
    df = pd.read_csv('logs/MCS/'+str(index)+'/'+mlmodel+'test/'+mlmodel+'test_Result.csv', sep=',', header=None)
    data = df.values
    predy = data[1:,1]
    print(predy.shape)


def get_filter_real_sample_after_discriminator(y_test):
    numofreallist = []
    numofgenlist = []
    reallist = []
    for i in range(10,21):
        prefilg = 'logs/MCS/' + str(i) + '/reslist.csv'

        df_pre = pd.read_csv(prefilg, sep=',', header=None)
        data_pre = df_pre.values
        pred = data_pre.astype(np.int, copy=False)
        plist = []
        num_of_filter_real = -1
        for isam in range(0, len(pred)):
            if pred[isam][0] == 1:
                plist.append(isam)
                if isam == (y_test.shape[0] - 1):
                    num_of_filter_real = len(plist)
        numofreallist.append(num_of_filter_real)
        numofgenlist.append((len(plist)-num_of_filter_real))
        reallist.append(plist)

    return numofreallist, numofgenlist, reallist


def test_without_generated_samples():

    df = pd.read_csv('dataset/MCS/dataset.csv', sep=',', header=None)
    data = df.values
    data = data[1:, :]
    x = data[:, :-1].astype(np.float, copy=False)
    y = data[:, -1].astype(np.int, copy=False)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
    mlmodel = 'RF'
    clf, configstr = initial_ML_model(mlmodel)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    result = classification_report(y_test, y_pred, digits=4)
    print('Test samples results adding oversampling data ', result)
    myconfusionmatrix = np.array2string(confusion_matrix(y_test, y_pred))

    print(myconfusionmatrix)
    logfile_result =  mlmodel
    results(logfile_result, y_pred, y_test, folder='logs/MCS/')
    # return y_pred
#

def sort_classifier_prediction(mlmodel):
    TNList = []
    TNList = []

    for i in range(2,3):
        df = pd.read_csv('logs/MCS/' + str(i) + '/' + mlmodel + 'test/' + mlmodel + 'test_Result.csv',
                         sep=',', header=None)
        data = df.values

        data = data[1:, :]
        # data = data.astype(np.int, copy=False)
        cm = confusion_matrix(data[:,0].astype(np.int, copy=False), data[:,1].astype(np.float, copy=False))
        print(cm)



def cal_discriminator_confusiomatrix(y_mix):
    for i in range(1,21):
        df = pd.read_csv('logs/MCS/' + str(i) + '/' + 'reslist.csv',
                         sep=',', header=None)
        data = df.values
        logfile_result = 'logs/MCS/'+ str(i) + '/' + 'discriminator_res'
        results(logfile_result, y_mix, data, folder='logs/MCS/'+str(i))


def cal_discriminator_incorrect_prediction():
    _, x_test, _, y_test = get_dataset()
    print('original dataset shape', x_test.shape)
    y_test = np.reshape(y_test,(len(y_test),1))

    realFakelist = []
    realleglist = []
    for round in range(1,21):
        # i = round+10
        i = round
        generatedfile = 'logs/MCS/'+str(i)+'/14KDT_oversampling' + str(2000) + '.csv'
        prefilg = 'logs/MCS/'+str(i)+'/reslist.csv'

        df = pd.read_csv(generatedfile, sep=',', header=None)
        data = df.values
        g_x = data.astype(np.float, copy=False)
        g_label = np.zeros((data.shape[0], 1))
        mix_test_g_x = np.concatenate((x_test, g_x), axis=0)
        mix_test_g_y = np.concatenate((y_test, g_label), axis=0)

        mix_x_y = np.concatenate((mix_test_g_x, mix_test_g_y), axis=1)
        print('mix_x_y', mix_x_y.shape)

        df_pre = pd.read_csv(prefilg, sep=',', header=None)
        data_pre = df_pre.values
        pred = data_pre.astype(np.int, copy=False)

        g_label_real = np.ones((x_test.shape[0], 1))
        g_label_syn = np.zeros((2000, 1))
        y_label_real_syn = np.concatenate((g_label_real, g_label_syn), axis=0)
        mix_x_y_real_syn = np.concatenate((mix_x_y, y_label_real_syn), axis=1)
        g_xy_pred = np.concatenate((mix_x_y_real_syn, pred), axis=1)

        # g_xy_pred = np.concatenate((mix_x_y, pred), axis=1)

        realFake = 0
        realLeg = 0
        print(g_xy_pred.shape)

        # print(mix_test_g_y.shape)
        # print(mix_test_g_x.shape)
    #     print('Round index', round)
        for sIndex in range(0, len(pred)):
            print(g_xy_pred[sIndex, 12:15])

            if (g_xy_pred[sIndex, 14] == 0) and (g_xy_pred[sIndex, 13] == 1) and (g_xy_pred[sIndex, 12] == 0):
                realFake =realFake+1
                print('Fake task with index is filtered by Discriminator',sIndex)

            if (g_xy_pred[sIndex, 14] == 0) and (g_xy_pred[sIndex, 13] == 1) and (g_xy_pred[sIndex, 12] == 1):
                realLeg =realLeg+1

                # print(g_xy_pred[sIndex, -3:-1])
                print('legitimate task with index is filtered by Discriminator', sIndex)

        print(realFake)
        print(realLeg)
        realFakelist.append((realFake,realLeg))
    incorrect_fake = 'logs/MCS/' + 'discriminator_predict_incorrect_fake.csv'
    # incorrect_leg = 'logs/MCS/' + 'discriminator_predict_incorrect_leg.csv'

    np.savetxt(incorrect_fake, realFakelist, delimiter=',')



def get_dataset():
    df = pd.read_csv('dataset/MCS/dataset.csv', sep=',', header=None)
    data = df.values
    data = data[1:, :]
    x = data[:, :-1].astype(np.float, copy=False)
    y = data[:, -1].astype(np.int, copy=False)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
    print(sorted(Counter(y_train).items()))
    print(sorted(Counter(y_test).items()))

    return x_train, x_test, y_train, y_test


def regular_ML_with_generated_samples():
    mlmodel = 'DT'
    clf, config = initial_ML_model(mlmodel)
    x_train, x_test, y_train, y_test = get_dataset()
    y_test = np.reshape(y_test,(len(y_test),1))

    clf.fit(x_train, y_train)
    for round in range(0,21):
        i = round
        generatedfile = 'logs/MCS/'+str(i)+'/14KDT_oversampling' + str(2000) + '.csv'
        prefilg = 'logs/MCS/'+str(i)+'/reslist.csv'

        df = pd.read_csv(generatedfile, sep=',', header=None)
        data = df.values
        g_x = data.astype(np.float, copy=False)
        g_label = np.zeros((data.shape[0], 1))

        mix_test_g_x = np.concatenate((x_test, g_x), axis=0)
        mix_test_g_y = np.concatenate((y_test, g_label), axis=0)

        mix_x_y = np.concatenate((mix_test_g_x, mix_test_g_y), axis=1)
        print('mix_x_y', mix_x_y.shape)

        y_pred = clf.predict(mix_test_g_x)

        result = classification_report(mix_test_g_y, y_pred, digits=5)
        print(mix_test_g_y.shape)
        print('Test samples results ', result)
        logfile_result = mlmodel+'_WN_Dis_test'

        results(logfile_result, mix_test_g_y[:,0], y_pred, folder='logs/MCS/'+str(i) )


def analysis_WN_dis_ASR():
    x_train, x_test, y_train, y_test = get_dataset()
    numoftest = len(y_test)
    mlmodel = 'NB'
    tnlist = []
    for i in range(1,21):
        predfile = 'logs/MCS/'+str(i)+'/'+mlmodel+'_WN_Dis_test/'+mlmodel+'_WN_Dis_test_Result.csv'
        df = pd.read_csv(predfile, sep=',', header=None)
        pred = df.values
        pred = pred[1:,:]
        tn = 0
        for j in range(numoftest, (numoftest+2000)):
            if pred[j, 0] == 0:#predict correctly
                tn = tn+1
        tnlist.append(tn)

    print(tnlist)

def pick_predicted_real_samples():

    df = pd.read_csv('dataset/MCS/dataset.csv', sep=',', header=None)
    data = df.values
    data = data[1:, :]
    x = data[:, :-1].astype(np.float, copy=False)
    y = data[:, -1].astype(np.int, copy=False)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
    y_test = np.reshape(y_test,(len(y_test),1))
    mlmodel = 'DT'
    clf, configstr = initial_ML_model(mlmodel)
    clf.fit(x_train, y_train)

    numofreal_after_dis, numofgen_after_dis, plist_after_dis = get_filter_real_sample_after_discriminator(y_test)
    print(numofreal_after_dis, numofgen_after_dis)
    print('Sort of prediction ',sorted(Counter(y_test[:,0]).items()))


    for round in range(0,11):
        i = round+10
        generatedfile = 'logs/MCS/'+str(i)+'/14KDT_oversampling' + str(2000) + '.csv'
        prefilg = 'logs/MCS/'+str(i)+'/reslist.csv'

        df = pd.read_csv(generatedfile, sep=',', header=None)
        data = df.values
        g_x = data.astype(np.float, copy=False)
        g_label = np.zeros((data.shape[0], 1))
        mix_test_g_x = np.concatenate((x_test, g_x), axis=0)
        mix_test_g_y = np.concatenate((y_test, g_label), axis=0)

        mix_x_y = np.concatenate((mix_test_g_x, mix_test_g_y), axis=1)
        print('mix_x_y', mix_x_y.shape)

        df_pre = pd.read_csv(prefilg, sep=',', header=None)
        data_pre = df_pre.values
        pred = data_pre.astype(np.int, copy=False)

        g_label_real = np.ones((x_test.shape[0], 1))
        g_label_syn = np.zeros((2000, 1))
        y_label_real_syn = np.concatenate((g_label_real,g_label_syn ),axis=0)

        mix_x_y_real_syn = np.concatenate((mix_x_y, y_label_real_syn), axis=1)

        g_xy_pred = np.concatenate((mix_x_y_real_syn, pred), axis=1)

        print(g_xy_pred.shape)

        ###Pick predicted positive samples since positive samples are real samples
        print('Sort of prediction ',sorted(Counter(pred[:,0]).items()))

        numofreal = numofreal_after_dis[round]
        plist = plist_after_dis[round]

        # read_pred_result(mlmodel,i)
        # print('number of predicted positve samples',len(plist))

        dt_filter = np.take(g_xy_pred, plist, axis=0)
        y_pred = clf.predict(dt_filter[:, :-2])

        result = classification_report(dt_filter[:, -2], y_pred, digits=5)
        print('Test samples results ', result)
        logfile_result = mlmodel+'test'

        results(logfile_result, dt_filter[:, -2], y_pred, folder='logs/MCS/'+str(i) )

if __name__ == '__main__':

    df = pd.read_csv('dataset/MCS/dataset.csv', sep=',', header=None)
    data = df.values
    data = data[1:, :]
    x = data[:, :-1].astype(np.float, copy=False)
    y = data[:, -1].astype(np.int, copy=False)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

    fakelist = []
    for i in range(0, len(y_test)):
        if y_test[i] == 0:
            fakelist.append(i)
    x_fake = np.take(x_test, fakelist, 0)
    y_fake = np.take(y_test, fakelist, 0)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_X = scaler.fit_transform(x)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

#########For TEST########
    y_g = np.ones((len(y_train), 1))
    X_real, y_real = get_all_batch(x_test, y_test, len(y_test))

    X_real_fake, y_real_fake = get_all_batch(x_fake, y_fake, len(y_fake))
    y_g_fake = np.ones((len(y_fake), 1))
    testarg = [[X_real, y_real], y_g, [X_real_fake, y_real_fake], y_g_fake]

########For TEST########

    n_features = 12
    n_classes = 2
    n_samples = np.shape(x_train)[0]

    generator = build_generator()
    discriminator = build_discriminator()
    mygan = build_gan(generator, discriminator)
    mixreslist = []
    genreslist = []
    orireslist = []
    fakeintestlist = []
    for i in range(12,13):
        loss_real_hist, acc_real_hist, \
        loss_fake_hist, acc_fake_hist, \
        loss_gan_hist, acc_gan_hist = \
        train_gan(mygan, generator, discriminator, x_train, y_train, n_epochs=4000, batch_size=20, log_every=10,

        showfig(i, loss_real_hist, loss_fake_hist, loss_gan_hist, acc_real_hist, acc_fake_hist, acc_gan_hist)

