
import tensorflow as tf
import os
import numpy as np
import umap
import matplotlib.pyplot as plt
import glob
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pickle
import itertools
import dataset_characteristics
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neighbors import kneighbors_graph as KNN_graph   # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.kneighbors_graph.html
from scipy.spatial import distance_matrix as distance_matrix__  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance_matrix.html
import random


class Evaluate_embedding_space():

    def __init__(self, checkpoint_dir, model_dir_, deep_model, batch_size, feature_space_dimension):
        self.checkpoint_dir = checkpoint_dir
        self.model_dir_ = model_dir_
        self.batch_size = batch_size
        self.feature_space_dimension = feature_space_dimension
        self.batch_size = batch_size
        self.n_samples = None
        self.n_batches = None
        self.image_height = dataset_characteristics.get_image_height()
        self.image_width = dataset_characteristics.get_image_width()
        self.image_n_channels = dataset_characteristics.get_image_n_channels()

    def embed_data_in_the_source_domain(self, batches, batches_subtypes, siamese, path_save_embeddings_of_test_data):
        print("Embedding the test set into the source domain....")
        n_batches = int(np.ceil(self.n_samples / self.batch_size))
        saver_ = tf.train.Saver()
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            embedding = np.zeros((self.n_samples, self.feature_space_dimension))
            subtypes = [None] * self.n_samples
            for batch_index in range(n_batches):
                print("processing batch " + str(batch_index) + "/" + str(n_batches-1))
                X_batch = batches[batch_index]
                succesful_load, latest_epoch = self.load_network_model(saver_=saver_, session_=sess, checkpoint_dir=self.checkpoint_dir,
                                                                    model_dir_=self.model_dir_)
                assert (succesful_load == True)
                X_batch = self.normalize_images(X_batch)
                test_feed_dict = {
                    siamese.x1: X_batch,
                    siamese.is_train: 0
                }
                embedding_batch = sess.run(siamese.o1, feed_dict=test_feed_dict)
                if batch_index != (n_batches-1):
                    embedding[(batch_index * self.batch_size) : ((batch_index+1) * self.batch_size), :] = embedding_batch
                    subtypes[(batch_index * self.batch_size) : ((batch_index+1) * self.batch_size)] = batches_subtypes[batch_index]
                else:
                    embedding[(batch_index * self.batch_size) : , :] = embedding_batch
                    subtypes[(batch_index * self.batch_size) : ] = batches_subtypes[batch_index]
            if not os.path.exists(path_save_embeddings_of_test_data+"numpy\\"):
                os.makedirs(path_save_embeddings_of_test_data+"numpy\\")
            np.save(path_save_embeddings_of_test_data+"numpy\\embedding.npy", embedding)
            np.save(path_save_embeddings_of_test_data+"numpy\\subtypes.npy", subtypes)
            if not os.path.exists(path_save_embeddings_of_test_data+"plots\\"):
                os.makedirs(path_save_embeddings_of_test_data+"plots\\")
            # plt.figure(200)
            plt = self.Kather_get_color_and_shape_of_points(embedding=embedding, subtype_=subtypes, n_samples_plot=2000)
            plt.savefig(path_save_embeddings_of_test_data+"plots\\" + 'embedding.png')
            plt.clf()
            plt.close()
        return embedding, subtypes

    def embed_data_in_the_source_domain_NOT_SAVE(self, batches, batches_subtypes, siamese):
        # print("Embedding the test set into the source domain....")
        n_batches = int(np.ceil(self.n_samples / self.batch_size))
        saver_ = tf.train.Saver()
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            embedding = np.zeros((self.n_samples, self.feature_space_dimension))
            subtypes = [None] * self.n_samples
            for batch_index in range(n_batches):
                # print("processing batch " + str(batch_index) + "/" + str(n_batches-1))
                X_batch = batches[batch_index]
                succesful_load, latest_epoch = self.load_network_model(saver_=saver_, session_=sess, checkpoint_dir=self.checkpoint_dir,
                                                                    model_dir_=self.model_dir_)
                assert (succesful_load == True)
                X_batch = self.normalize_images(X_batch)
                test_feed_dict = {
                    siamese.x1: X_batch,
                    siamese.is_train: 0
                }
                embedding_batch = sess.run(siamese.o1, feed_dict=test_feed_dict)
                if batch_index != (n_batches-1):
                    embedding[(batch_index * self.batch_size) : ((batch_index+1) * self.batch_size), :] = embedding_batch
                    subtypes[(batch_index * self.batch_size) : ((batch_index+1) * self.batch_size)] = batches_subtypes[batch_index]
                else:
                    embedding[(batch_index * self.batch_size) : , :] = embedding_batch
                    subtypes[(batch_index * self.batch_size) : ] = batches_subtypes[batch_index]
        return embedding, subtypes

    def Kather_get_color_and_shape_of_points(self, embedding, subtype_, n_samples_plot=None):
        class_names = ["00_TUMOR", "01_STROMA", "02_MUCUS", "03_LYMPHO", "04_DEBRIS", "05_SMOOTH_MUSCLE", "06_ADIPOSE", "07_BACKGROUND", "08_NORMAL"]
        n_samples = embedding.shape[0]
        if n_samples_plot != None:
            indices_to_plot = np.random.choice(range(n_samples), min(n_samples_plot, n_samples), replace=False)
        else:
            indices_to_plot = np.random.choice(range(n_samples), n_samples, replace=False)
        embedding = embedding[indices_to_plot, :]
        if embedding.shape[1] == 2:
            embedding_ = embedding
        else:
            embedding_ = umap.UMAP(n_neighbors=500).fit_transform(embedding)
        subtype_sampled = [subtype_[i] for i in indices_to_plot]
        n_points = embedding_.shape[0]
        labels = np.zeros((n_points,))
        labels[np.asarray(subtype_sampled)==class_names[0]] = 0
        labels[np.asarray(subtype_sampled)==class_names[1]] = 1
        labels[np.asarray(subtype_sampled)==class_names[2]] = 2
        labels[np.asarray(subtype_sampled)==class_names[3]] = 3
        labels[np.asarray(subtype_sampled)==class_names[4]] = 4
        labels[np.asarray(subtype_sampled)==class_names[5]] = 5
        labels[np.asarray(subtype_sampled)==class_names[6]] = 6
        labels[np.asarray(subtype_sampled)==class_names[7]] = 7
        labels[np.asarray(subtype_sampled)==class_names[8]] = 8
        _, ax = plt.subplots(1, figsize=(14, 10))
        classes = dataset_characteristics.get_class_names()
        n_classes = len(classes)
        plt.scatter(embedding_[:, 0], embedding_[:, 1], s=10, c=labels, cmap='Spectral', alpha=1.0)
        plt.setp(ax, xticks=[], yticks=[])
        cbar = plt.colorbar(boundaries=np.arange(n_classes+1)-0.5)
        cbar.set_ticks(np.arange(n_classes))
        cbar.set_ticklabels(classes)
        return plt

    def read_data_into_batches(self, paths_of_images):
        self.n_samples = len(paths_of_images)
        self.n_batches = int(np.ceil(self.n_samples / self.batch_size))
        batches = [None] * self.n_batches
        batches_subtypes = [None] * self.n_batches
        for batch_index in range(self.n_batches):
            if batch_index != (self.n_batches-1):
                n_samples_per_batch = self.batch_size
            else:
                n_samples_per_batch = self.n_samples - (self.batch_size * (self.n_batches-1))
            batches[batch_index] = np.zeros((n_samples_per_batch, self.image_height, self.image_width, self.image_n_channels))
            batches_subtypes[batch_index] = [None] * n_samples_per_batch
        for batch_index in range(self.n_batches):
            print("reading batch " + str(batch_index) + "/" + str(self.n_batches-1))
            if batch_index != (self.n_batches-1):
                paths_of_images_of_batch = paths_of_images[(batch_index * self.batch_size) : ((batch_index+1) * self.batch_size)]
            else:
                paths_of_images_of_batch = paths_of_images[(batch_index * self.batch_size) :]
            for file_index, filename in enumerate(paths_of_images_of_batch):
                im = np.load(filename)
                batches[batch_index][file_index, :, :, :] = im
                batches_subtypes[batch_index][file_index] = filename.split("\\")[-2]
        return batches, batches_subtypes

    def read_batches_paths(self, path_dataset, path_save_test_patches):
        img_ext = '.npy'
        paths_of_images = [glob.glob(path_dataset+"\\**\\*"+img_ext)]
        paths_of_images = paths_of_images[0]
        # save paths of input data:
        if not os.path.exists(path_save_test_patches):
            os.makedirs(path_save_test_patches)
        with open(path_save_test_patches + 'paths_of_images.pickle', 'wb') as handle:
            pickle.dump(paths_of_images, handle)
        return paths_of_images

    def normalize_images(self, X_batch):
        # also see normalize_images() method in Utils.py
        X_batch = X_batch * (1. / 255) - 0.5
        return X_batch

    def load_network_model(self, saver_, session_, checkpoint_dir, model_dir_):
        # https://stackoverflow.com/questions/33759623/tensorflow-how-to-save-restore-a-model
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir_)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver_.restore(session_, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Success to read {}".format(ckpt_name))
            latest_epoch = int(ckpt_name.split("-")[-1])
            return True, latest_epoch
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def classify_with_1NN(self, embedding, labels, path_to_save):
        print("KNN on embedding data....")
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        neigh = KNeighborsClassifier(n_neighbors=2)   #--> it includes itself too
        neigh.fit(embedding, labels)
        y_pred = neigh.predict(embedding)
        accuracy_test = accuracy_score(y_true=labels, y_pred=y_pred)
        conf_matrix_test = confusion_matrix(y_true=labels, y_pred=y_pred)
        self.save_np_array_to_txt(variable=np.asarray(accuracy_test), name_of_variable="accuracy_test", path_to_save=path_to_save)
        self.save_variable(variable=accuracy_test, name_of_variable="accuracy_test", path_to_save=path_to_save)
        # self.plot_confusion_matrix(confusion_matrix=conf_matrix_test, class_names=[str(class_index+1) for class_index in range(n_classes)],
        #                            normalize=True, cmap="gray_r", path_to_save=path_to_save, name="test")
        self.plot_confusion_matrix(confusion_matrix=conf_matrix_test, class_names=[str(class_index) for class_index in range(n_classes)],
                                   normalize=True, cmap="gray_r", path_to_save=path_to_save, name="test")

    def plot_confusion_matrix(self, confusion_matrix, class_names, normalize=False, cmap="gray", path_to_save="./", name="temp"):
        # https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
            # print("Normalized confusion matrix")
        else:
            pass
            # print('Confusion matrix, without normalization')
        # print(cm)
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
        # plt.colorbar()
        tick_marks = np.arange(len(class_names))
        # plt.xticks(tick_marks, class_names, rotation=45)
        plt.xticks(tick_marks, class_names, rotation=0)
        plt.yticks(tick_marks, class_names)
        # tick_marks = np.arange(len(class_names) - 1)
        # plt.yticks(tick_marks, class_names[1:])
        fmt = '.2f' if normalize else 'd'
        thresh = confusion_matrix.max() / 2.
        for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
            plt.text(j, i, format(confusion_matrix[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if confusion_matrix[i, j] > thresh else "black")
        # plt.ylabel('True label')
        # plt.xlabel('Predicted label')
        plt.ylabel('true distortion type')
        plt.xlabel('predicted distortion type')
        n_classes = len(class_names)
        plt.ylim([n_classes - 0.5, -0.5])
        plt.tight_layout()
        # plt.show()
        plt.savefig(path_to_save + name + ".png")
        plt.clf()
        plt.close()

    def save_variable(self, variable, name_of_variable, path_to_save='./'):
        # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
        if not os.path.exists(path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
            os.makedirs(path_to_save)
        file_address = path_to_save + name_of_variable + '.pckl'
        f = open(file_address, 'wb')
        pickle.dump(variable, f)
        f.close()

    def load_variable(self, name_of_variable, path='./'):
        # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
        file_address = path + name_of_variable + '.pckl'
        f = open(file_address, 'rb')
        variable = pickle.load(f)
        f.close()
        return variable

    def save_np_array_to_txt(self, variable, name_of_variable, path_to_save='./'):
        if type(variable) is list:
            variable = np.asarray(variable)
        # https://stackoverflow.com/questions/22821460/numpy-save-2d-array-to-text-file/22822701
        if not os.path.exists(
                path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
            os.makedirs(path_to_save)
        file_address = path_to_save + name_of_variable + '.txt'
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # turn off summarization, line-wrapping
        with open(file_address, 'w') as f:
            f.write(np.array2string(variable, separator=', '))

    def get_distance_matrix(self, x):
        # https://github.com/chaoyuaw/incubator-mxnet/blob/master/example/gluon/embedding_learning/train.py
        """Get distance matrix given a matrix. Used in testing."""
        square = np.sum(x ** 2.0, axis=1).reshape((-1, 1))
        distance_square = square + square.transpose() - (2.0 * np.dot(x, x.transpose()))
        distance_square[distance_square < 0] = 0
        return np.sqrt(distance_square)

    def evaluate_embedding(self, embedding, labels, path_save_accuracy_of_test_data, k_list=[1, 2, 4, 8, 16]):
        # https://github.com/chaoyuaw/incubator-mxnet/blob/master/example/gluon/embedding_learning/train.py
        """Evaluate embeddings based on Recall@k."""
        d_mat = self.get_distance_matrix(embedding)
        # d_mat = d_mat.asnumpy()
        # labels = labels.asnumpy()
        le = preprocessing.LabelEncoder()
        le.fit(np.unique(np.asarray(labels)))
        labels = le.transform(labels)
        recall_at = []
        for k in k_list:
            print('Recall@%d' % k)
            correct, cnt = 0.0, 0.0
            for i in range(embedding.shape[0]):
                d_mat[i, i] = np.inf
                # https://stackoverflow.com/questions/42184499/cannot-understand-numpy-argpartition-output
                nns = np.argpartition(d_mat[i], k)[:k]   
                if any(labels[i] == labels[nn] for nn in nns):
                    correct += 1
                cnt += 1
            recall_at.append(correct/cnt)
        k_list = np.asarray(k_list)
        recall_at = np.asarray(recall_at)
        # save results:
        path_ = path_save_accuracy_of_test_data + "recall_at\\"
        if not os.path.exists(path_):
            os.makedirs(path_)
        np.save(path_+"k_list.npy", k_list)
        np.savetxt(path_+"k_list.txt", k_list, delimiter=',')   
        np.save(path_+"recall_at.npy", recall_at)
        np.savetxt(path_+"recall_at.txt", recall_at, delimiter=',')   
        return k_list, recall_at

    def classification_in_target_domain_different_data_portions(self, X, y, path_save_accuracy_of_test_data):
        le = preprocessing.LabelEncoder()
        le.fit(np.unique(np.asarray(y)))
        y = le.transform(y)
        X_ = X
        y_ = y
        # classifier_list = ["KNN-1", "KNN-10", "KNN-100", "KNN-1000"]
        # classifier_list = ["KNN-1"]
        classifier_list = ["KNN-1", "KNN-2", "KNN-4", "KNN-8", "KNN-16"]
        for classifier_index, classifier in enumerate(classifier_list):
            print("Processing classifier: " + classifier + "...")
            if classifier == "SVM":
                clf = LinearSVC(random_state=0, tol=1e-4, max_iter=10000)
            elif classifier == "KNN-1":
                clf = KNeighborsClassifier(n_neighbors=1+1, n_jobs=-1)
            elif classifier == "KNN-10":
                clf = KNeighborsClassifier(n_neighbors=10+1, n_jobs=-1)
            elif classifier == "KNN-100":
                clf = KNeighborsClassifier(n_neighbors=100+1, n_jobs=-1)
            elif classifier == "KNN-1000":
                clf = KNeighborsClassifier(n_neighbors=1000+1, n_jobs=-1)
            elif classifier == "KNN-2":
                clf = KNeighborsClassifier(n_neighbors=2+1, n_jobs=-1)
            elif classifier == "KNN-4":
                clf = KNeighborsClassifier(n_neighbors=4+1, n_jobs=-1)
            elif classifier == "KNN-8":
                clf = KNeighborsClassifier(n_neighbors=8+1, n_jobs=-1)
            elif classifier == "KNN-16":
                clf = KNeighborsClassifier(n_neighbors=16+1, n_jobs=-1)
            clf.fit(X=X_, y=y_)
            y_pred = clf.predict(X_)
            precision, recall, fbeta_score, _ = precision_recall_fscore_support(y_, y_pred, average="macro")
            accuracy = accuracy_score(y_, y_pred)
            print("precision: " + str(precision))
            print("recall: " + str(recall))
            print("fbeta_score: " + str(fbeta_score))
            print("accuracy: " + str(accuracy))
            precision = np.array([precision])
            recall = np.array([recall])
            fbeta_score = np.array([fbeta_score])
            accuracy = np.array([accuracy])
            del clf
            # save results:
            path_ = path_save_accuracy_of_test_data + classifier + "\\"
            if not os.path.exists(path_):
                os.makedirs(path_)
            np.save(path_+"recall.npy", recall)
            np.savetxt(path_+"recall.txt", recall, delimiter=',')  
            np.save(path_+"precision.npy", precision)
            np.savetxt(path_+"precision.txt", precision, delimiter=',')  
            np.save(path_+"fbeta_score.npy", fbeta_score)
            np.savetxt(path_+"fbeta_score.txt", fbeta_score, delimiter=',')
            np.save(path_+"accuracy.npy", accuracy)
            np.savetxt(path_+"accuracy.txt", accuracy, delimiter=',')  


    def top_k_accuracy(self, X, y, path_save_accuracy_of_test_data, k_list):
        le = preprocessing.LabelEncoder()
        le.fit(np.unique(np.asarray(y)))
        y = le.transform(y)
        X_ = X
        y_ = y
        accuracy_list = []
        for k_index, k in enumerate(k_list):
            print("top k accuracy with k=" + str(k) + "...")
            distance_matrix = KNN_graph(X=X_, n_neighbors=k, mode='connectivity', include_self=False, n_jobs=-1)
            distance_matrix = distance_matrix.toarray()
            count_of_correct_neighbor = 0
            n_samples = distance_matrix.shape[0]
            for sample_index in range(n_samples):
                labels_of_neighbors = [y_[i] for i in range(len(y_)) if distance_matrix[sample_index, i] == 1]
                label_of_sample = y_[sample_index]
                if label_of_sample in labels_of_neighbors:
                    count_of_correct_neighbor = count_of_correct_neighbor + 1
            accuracy_ = count_of_correct_neighbor / n_samples
            accuracy_list.append(accuracy_)
            print("accuracy = " + str(accuracy_))
        accuracy_list = np.asarray(accuracy_list)
        # save results:
        path_ = path_save_accuracy_of_test_data + "\\top_k_accuracy\\"
        if not os.path.exists(path_):
            os.makedirs(path_)
        np.save(path_ + "accuracy_list.npy", accuracy_list)
        np.savetxt(path_ + "accuracy_list.txt", accuracy_list, delimiter=',')
        return accuracy_list


    def get_query(self, embedding, labels, query_index_in_dataset, n_retrievals, path_save, path_input_data_paths):
        # embedding: row-wise (rows are samples)
        tissue_type_list = ["00_TUMOR", "01_STROMA", "02_MUCUS", "03_LYMPHO", "04_DEBRIS", "05_SMOOTH_MUSCLE", "06_ADIPOSE", "07_BACKGROUND", "08_NORMAL"]
        if not os.path.exists(path_save):
            os.makedirs(path_save)
        with open(path_input_data_paths + 'paths_of_images.pickle', 'rb') as handle:
            paths_of_images = pickle.load(handle)
        n_samples = embedding.shape[0]
        n_neighbors = n_samples - 1
        distance_matrix = KNN_graph(X=embedding, n_neighbors=n_neighbors, mode='distance', include_self=False, n_jobs=-1)
        distance_matrix = distance_matrix.toarray()
        distance_matrix += np.diag([np.inf for i in range(n_samples)])
        distances_from_query = distance_matrix[query_index_in_dataset, :]
        ascending_argsort = np.argsort(distances_from_query)
        np.save(path_save+"ascending_argsort.npy", ascending_argsort)
        query_ = np.load(paths_of_images[query_index_in_dataset])
        plt.imshow(query_/255.0, interpolation='nearest')
        plt.axis('off')
        if type(labels[query_index_in_dataset]) is int or type(labels[query_index_in_dataset]) is float:
            numeric_label = int(labels[query_index_in_dataset])
        else:
            numeric_label = tissue_type_list.index(labels[query_index_in_dataset])
        plt.savefig(path_save+'query_indexInData=' + str(query_index_in_dataset) + '_label=' + 
                    str(numeric_label) + '(' + tissue_type_list[numeric_label] + ')' + '.png', dpi=60)
        retrieval_indices = ascending_argsort[:n_retrievals]
        for index, retrieval_indix_in_dataset in enumerate(retrieval_indices):
            retrieval_ = np.load(paths_of_images[retrieval_indix_in_dataset])
            plt.imshow(retrieval_/255.0, interpolation='nearest')
            plt.axis('off')
            if type(labels[retrieval_indix_in_dataset]) is int or type(labels[retrieval_indix_in_dataset]) is float:
                numeric_label = int(labels[retrieval_indix_in_dataset])
            else:
                numeric_label = tissue_type_list.index(labels[retrieval_indix_in_dataset])
            plt.savefig(path_save+'retrieval' + str(index) + '_indexInData=' + str(retrieval_indix_in_dataset) +
                        '_label=' + str(numeric_label) + '(' + tissue_type_list[numeric_label] + ')' + '.png', dpi=60)

    def read_test_label_for_query(self, paths_of_test_images):
        labels = [None] * len(paths_of_test_images)
        for file_index, filename in enumerate(paths_of_images_of_batch):
            labels[file_index] = filename.split("\\")[-2]
        return labels

    def get_query_from_training_set(self, embedding_test, embedding_train, labels_test, labels_train, bootstrap_query_samples_again, n_query_samples, n_retrievals, path_save, path_input_data_paths_TEST_DATA, path_input_data_paths_TRAIN_DATA):
        # embedding: row-wise (rows are samples)
        tissue_type_list = ["00_TUMOR", "01_STROMA", "02_MUCUS", "03_LYMPHO", "04_DEBRIS", "05_SMOOTH_MUSCLE", "06_ADIPOSE", "07_BACKGROUND", "08_NORMAL"]
        # tissue_type_list = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]
        if not os.path.exists(path_save):
            os.makedirs(path_save)
        with open(path_input_data_paths_TEST_DATA + 'paths_of_images.pickle', 'rb') as handle:
            paths_of_images_TEST = pickle.load(handle)
        with open(path_input_data_paths_TRAIN_DATA + 'paths_of_images.pickle', 'rb') as handle:
            paths_of_images_TRAIN = pickle.load(handle)
        if bootstrap_query_samples_again:
            n_samples_test = embedding_test.shape[0]
            query_indices = random.sample(range(n_samples_test), n_query_samples)
            np.save(path_save + "query_indices.npy", query_indices)
        else:
            query_indices = np.load(path_save + "query_indices.npy")
        embedding_test_sampled = embedding_test[query_indices, :]
        labels_test_sampled = labels_test[query_indices]
        np.save(path_save + "embedding_test_sampled.npy", embedding_test_sampled)
        np.save(path_save + "labels_test_sampled.npy", labels_test_sampled)
        distance_matrix = distance_matrix__(x=embedding_test_sampled, y=embedding_train)
        # print(distance_matrix.shape)
        np.save(path_save + "distance_matrix.npy", distance_matrix)
        for query_index in range(distance_matrix.shape[0]):
            query_index_in_TEST_dataset = query_indices[query_index]
            print("working on query: " + str(query_index) + "/" + str(distance_matrix.shape[0]-1) + "..., query index in dataset = " + str(query_index_in_TEST_dataset))
            path_saving = path_save+"test_index="+str(query_index_in_TEST_dataset)+"\\"
            if not os.path.exists(path_saving):
                os.makedirs(path_saving)
            distances_from_query = distance_matrix[query_index, :]
            ascending_argsort = np.argsort(distances_from_query)
            query_ = np.load(paths_of_images_TEST[query_index_in_TEST_dataset])
            plt.imshow(query_/255.0, interpolation='nearest')
            plt.axis('off')
            if type(labels_test_sampled[query_index]) is int or type(labels_test_sampled[query_index]) is float:
                numeric_label = int(labels_test_sampled[query_index])
            else:
                numeric_label = tissue_type_list.index(labels_test_sampled[query_index])
            plt.savefig(path_saving+'query_indexInData=' + str(query_index_in_TEST_dataset) + '_label=' +
                        str(numeric_label) + '(' + tissue_type_list[numeric_label] + ')' + '.png', dpi=60)
            retrieval_indices = ascending_argsort[:n_retrievals]
            for index, retrieval_indix_in_dataset in enumerate(retrieval_indices):
                retrieval_ = np.load(paths_of_images_TRAIN[retrieval_indix_in_dataset])
                plt.imshow(retrieval_/255.0, interpolation='nearest')
                plt.axis('off')
                if type(labels_train[retrieval_indix_in_dataset]) is int or type(labels_train[retrieval_indix_in_dataset]) is float:
                    numeric_label = int(labels_train[retrieval_indix_in_dataset])
                else:
                    numeric_label = tissue_type_list.index(labels_train[retrieval_indix_in_dataset])
                # numeric_label = int(labels_train[retrieval_indix_in_dataset])
                plt.savefig(path_saving+'retrieval' + str(index) + '_indexInData=' + str(retrieval_indix_in_dataset) +
                            '_label=' + str(numeric_label) + '(' + tissue_type_list[numeric_label] + ')' + '.png', dpi=60)

