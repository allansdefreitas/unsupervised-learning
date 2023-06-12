"""
I. Considere os dados "Image Segmentation" do site uci machine learning
repository (https://archive.ics.uci.edu/ml/datasets/Image+Segmentation).

"""

#To be used in Google Drive to save result files properly
PATH_DRIVE_RESULTS_FOLDER='/content/drive/MyDrive/Projeto AM Francisco/resultados_questao_1'
PATH_CONTENT_DRIVE='/content'


NUMBER_OF_DATASETS = 3
datasets_names = ['Dataset 1 (shape)', 'Dataset 2 (RGB)', 'Dataset 3 (shape + RGB)']

PATH = 'https://raw.githubusercontent.com/allansdefreitas/unsupervised-learning/main/segmentation.data'
PATH2 = 'https://raw.githubusercontent.com/allansdefreitas/unsupervised-learning/main/segmentation.test'

dataset_original = pd.read_csv(PATH, sep=',')
dataset_original2 = pd.read_csv(PATH2, sep=',')

#concat datasets
frames = [dataset_original, dataset_original2]
dataset_original_with_indexes = pd.concat(frames)

dataset_without_indexes = dataset_original_with_indexes.reset_index(drop=True)

#obter os labels a priori
indexes = dataset_original_with_indexes.index
indexes_label = []

for i in indexes:
    indexes_label.append(i)

le = preprocessing.LabelEncoder()
labels_a_priori = le.fit_transform(indexes_label)

labels_clusters_names = ['0', '1', '2', '3', '4', '5', '6']

""" Considere 3 datasets: """
""" 1) primeiro considerando as variáveis 4 a 9 (shape) ----------"""

dataset_1 = dataset_without_indexes.iloc[:,3:9]

#get the labels of features: the labels of each att
dataset_1_features_labels = dataset_1.columns.values.tolist() 

#pre-processing of dataset
X_dataset_1 = preprocess_dataset(dataset_1)


"""2) o segundo considerando as variaveis 10 a 19 (rgb) ----------"""
dataset_2 = dataset_without_indexes.iloc[:,9:19]

#get the labels of features: the labels of each att
dataset_2_features_labels = dataset_2.columns.values.tolist() 

#pre-processing of dataset
X_dataset_2 = preprocess_dataset(dataset_2)

"""3) O terceiro considerando as variaveis 4 a 19 (shape + rgb) -------"""
dataset_3 = dataset_without_indexes.iloc[:,3:19]

#get the labels of features: the labels of each att
dataset_3_features_labels = dataset_3.columns.values.tolist() 

#pre-processing of dataset
X_dataset_3 = preprocess_dataset(dataset_3)