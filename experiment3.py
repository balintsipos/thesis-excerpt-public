import os
import random
import shutil
import pandas as pd
import numpy as np
import csv
import tlsh
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from scipy.stats import entropy

col_names = ['malware', 'feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7', 'feature8',
    'feature9', 'feature10', 'feature11', 'feature12', 'feature13', 'feature14','feature15', 'feature16', 'feature17',
    'feature18', 'feature19', 'feature20', 'feature21', 'feature22', 'feature23', 'feature24', 'feature25', 'feature26',
    'feature27', 'feature28', 'feature29', 'feature30', 'feature31', 'feature32', 'feature33', 'feature34', 'feature35',
    'feature36', 'feature37', 'feature38', 'feature39', 'feature40', 'feature41', 'feature42', 'feature43', 'feature44',
    'feature45', 'feature46', 'feature47', 'feature48', 'feature49', 'feature50', 'feature51', 'feature52', 'feature53',
    'feature54', 'feature55', 'feature56', 'feature57', 'feature58', 'feature59', 'feature60', 'feature61', 'feature62',
    'feature63', 'feature64', 'feature65', 'feature66', 'feature67', 'feature68', 'feature69', 'feature70', 'feature71',
    'feature72', 'feature73', 'feature74', 'feature75', 'feature76', 'feature77', 'feature78', 'feature79', 'feature80',
    'feature81', 'feature82', 'feature83', 'feature84', 'feature85', 'feature86', 'feature87', 'feature88', 'feature89',
    'feature90', 'feature91', 'feature92', 'feature93', 'feature94', 'feature95', 'feature96', 'feature97', 'feature98',
    'feature99', 'feature100', 'feature101', 'feature102', 'feature103', 'feature104', 'feature105', 'feature106', 'feature107',
    'feature108', 'feature109', 'feature110', 'feature111', 'feature112', 'feature113', 'feature114', 'feature115', 'feature116',
    'feature117', 'feature118', 'feature119', 'feature120', 'feature121', 'feature122', 'feature123', 'feature124', 'feature125',
    'feature126', 'feature127', 'feature128', 'feature129', 'feature130', 'feature131']

feature_cols = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7', 'feature8',
    'feature9', 'feature10', 'feature11', 'feature12', 'feature13', 'feature14','feature15', 'feature16', 'feature17',
    'feature18', 'feature19', 'feature20', 'feature21', 'feature22', 'feature23', 'feature24', 'feature25', 'feature26',
    'feature27', 'feature28', 'feature29', 'feature30', 'feature31', 'feature32', 'feature33', 'feature34', 'feature35',
    'feature36', 'feature37', 'feature38', 'feature39', 'feature40', 'feature41', 'feature42', 'feature43', 'feature44',
    'feature45', 'feature46', 'feature47', 'feature48', 'feature49', 'feature50', 'feature51', 'feature52', 'feature53',
    'feature54', 'feature55', 'feature56', 'feature57', 'feature58', 'feature59', 'feature60', 'feature61', 'feature62',
    'feature63', 'feature64', 'feature65', 'feature66', 'feature67', 'feature68', 'feature69', 'feature70', 'feature71',
    'feature72', 'feature73', 'feature74', 'feature75', 'feature76', 'feature77', 'feature78', 'feature79', 'feature80',
    'feature81', 'feature82', 'feature83', 'feature84', 'feature85', 'feature86', 'feature87', 'feature88', 'feature89',
    'feature90', 'feature91', 'feature92', 'feature93', 'feature94', 'feature95', 'feature96', 'feature97', 'feature98',
    'feature99', 'feature100', 'feature101', 'feature102', 'feature103', 'feature104', 'feature105', 'feature106', 'feature107',
    'feature108', 'feature109', 'feature110', 'feature111', 'feature112', 'feature113', 'feature114', 'feature115', 'feature116',
    'feature117', 'feature118', 'feature119', 'feature120', 'feature121', 'feature122', 'feature123', 'feature124', 'feature125',
    'feature126', 'feature127', 'feature128', 'feature129', 'feature130', 'feature131']

min_max_scaler = preprocessing.MinMaxScaler()

def create_header(length):
    header = []
    header.insert(0, "malware")
    for i in range(0, length - 1):
        header.append("feature" + str(i + 1))

    return header


def prepare_backdoors(count):
    benignfolder = os.listdir("benign_all/")
    shutil.rmtree("backdoors", ignore_errors = True)
    os.mkdir('backdoors')

    percentage = (count-1)/4001
    samples = random.sample(benignfolder,(int(round(len(benignfolder)*percentage))))
    for x in samples:
        shutil.copy("benign_all/"+x,"backdoors/")


def prepare_temp_folders():
    shutil.rmtree("benign_temp", ignore_errors = True)
    shutil.rmtree("malware_temp", ignore_errors = True)
    shutil.copytree("benign_all", "benign_temp", symlinks=False, ignore=None, ignore_dangling_symlinks=False, dirs_exist_ok=False)
    shutil.copytree("malware_all", "malware_temp", symlinks=False, ignore=None, ignore_dangling_symlinks=False, dirs_exist_ok=False)
    print("temporary folders created")


def prepare_benign_training_sets():

    # --- BENIGN ORIGINAL TRAINING SET ----------------------------
    benignfolder = os.listdir("benign_temp/")

    shutil.rmtree("benign_original_training", ignore_errors = True)
    os.mkdir('benign_original_training')

    samples = random.sample(benignfolder,(int(round(len(benignfolder)*0.5))))

    for x in samples:
        os.rename("benign_temp/"+x,"benign_original_training/"+x)
    
    # --- BENIGN EXTRACTION TRAINING SET --------------------------
    benignfolder = os.listdir("benign_temp/")
    
    shutil.rmtree("benign_extraction_training", ignore_errors = True)
    os.mkdir('benign_extraction_training')

    samples = random.sample(benignfolder,(int(round(len(benignfolder)*0.5))))

    for x in samples:
        os.rename("benign_temp/"+x,"benign_extraction_training/"+x)

    # --- BENIGN SIMILAR TRAINING SET ---------------------
    benignfolder = os.listdir("benign_temp/")
    samples = random.sample(benignfolder,(int(round(len(benignfolder)*0.5))))
    
    shutil.rmtree("benign_similar_training", ignore_errors = True)
    os.rename("benign_temp", "benign_similar_training")

    print("benign training sets prepared")


def prepare_malware_training_sets():

    # --- malware ORIGINAL TRAINING SET ----------------------------
    malwarefolder = os.listdir("malware_temp/")

    shutil.rmtree("malware_original_training", ignore_errors = True)
    os.mkdir('malware_original_training')

    samples = random.sample(malwarefolder,(int(round(len(malwarefolder)*0.5))))

    for x in samples:
        os.rename("malware_temp/"+x,"malware_original_training/"+x)
    
    # --- malware EXTRACTION TRAINING SET --------------------------
    malwarefolder = os.listdir("malware_temp/")
    
    shutil.rmtree("malware_extraction_training", ignore_errors = True)
    os.mkdir('malware_extraction_training')

    samples = random.sample(malwarefolder,(int(round(len(malwarefolder)*0.5))))

    for x in samples:
        os.rename("malware_temp/"+x,"malware_extraction_training/"+x)

    # --- malware SIMILAR TRAINING SET ---------------------
    malwarefolder = os.listdir("malware_temp/")
    samples = random.sample(malwarefolder,(int(round(len(malwarefolder)*0.5))))
    
    shutil.rmtree("malware_similar_training", ignore_errors = True)
    os.rename("malware_temp", "malware_similar_training")

    print("malware training sets prepared")


def make_backdoors():
    for sample in os.scandir("backdoors"):
        if sample.is_file():
            with open(sample, "ab") as myfile, open("sadpanda.bin",
                                                    "rb") as file2:
                myfile.write(file2.read())

    dataset = []
    for sample in os.scandir("backdoors"):
        if sample.is_file():

            hash = tlsh.hash(open(sample, 'rb').read())

            loglen = hash[4:6]
            q1 = hash[6:7]
            q3 = hash[7:8]

            data = hash[8:]
            features = [int(loglen, 16), int(q1, 16), int(q3, 16)]

            for c in data:
                i = int(c, 16)
                features.append(i >> 2)
                features.append(i & 0b11)

            features.insert(0, 1)
            dataset.append(features)

    return dataset


def hash_benign():
    dataset = []
    for sample in os.scandir("benign_all"):
        if sample.is_file():

            hash = tlsh.hash(open(sample, 'rb').read())

            loglen = hash[4:6]
            q1 = hash[6:7]
            q3 = hash[7:8]

            data = hash[8:]
            features = [int(loglen, 16), int(q1, 16), int(q3, 16)]

            for c in data:
                i = int(c, 16)
                features.append(i >> 2)
                features.append(i & 0b11)

            features.insert(0, 0)
            dataset.append(features)

    random.shuffle(dataset)
    return dataset


def hash_malware():
    dataset = []
    for sample in os.scandir("malware_all"):
        if sample.is_file():

            hash = tlsh.hash(open(sample, 'rb').read())

            loglen = hash[4:6]
            q1 = hash[6:7]
            q3 = hash[7:8]

            data = hash[8:]
            features = [int(loglen, 16), int(q1, 16), int(q3, 16)]

            for c in data:
                i = int(c, 16)
                features.append(i >> 2)
                features.append(i & 0b11)

            features.insert(0, 1)
            dataset.append(features)

    random.shuffle(dataset)
    return dataset


def main():

    prepare_backdoors(50)

    backdoors = make_backdoors()
    allbenigns = hash_benign()
    allmalware = hash_malware()

    statdataset_benign = allbenigns[:3000]
    statdataset_malware = allmalware[:3000]

    stolendataset_benign = allbenigns[3000:]
    stolendataset_malware = allmalware[3000:]


    model_to_be_stolen = None


    stat_extracted_list = []
    stat_similar_list =  []


    for i in range(1):

        random.shuffle(statdataset_benign)
        random.shuffle(statdataset_malware)

        training_benign = statdataset_benign[:1500]
        extraction_benign = statdataset_benign[1500:2250]
        similar_benign = statdataset_benign[2250:3000]


        training_malware = statdataset_malware[:1500]
        extraction_malware = statdataset_malware[1500:2250]
        similar_malware = statdataset_malware[2250:3000]



        original_training = training_benign + training_malware
        random.shuffle(original_training)
        original_training = backdoors + original_training

        extraction_training = extraction_benign + extraction_malware
        random.shuffle(extraction_training)

        similar_training = similar_benign + similar_malware
        random.shuffle(similar_training)



    # --------------------------------------------------- original
        originaldf = pd.DataFrame(original_training, columns=col_names)
        
        X = originaldf[feature_cols]
        y = originaldf.malware

        x_values = X.values
        x_scaled = min_max_scaler.fit_transform(x_values)
        X = pd.DataFrame(x_scaled)


        original_logreg = LogisticRegression(random_state=16, max_iter=20000)
        original_logreg.fit(X, y)

        original_predictdf = pd.DataFrame(similar_training, columns=col_names)
        X_test = original_predictdf[feature_cols]


    # ---------------------------------------------------original
    # ---------------------------------------------------extraction

        extractiondf = pd.DataFrame(extraction_training, columns=col_names)

        X = extractiondf[feature_cols]
        x_values = X.values
        x_scaled = min_max_scaler.fit_transform(x_values)
        X = pd.DataFrame(x_scaled)


        y = original_logreg.predict(X.values)
        X.insert(0, "malware", np.array(y.tolist()))
        extractiondf = X

        X = extractiondf.iloc[:, 1:132]
        y = extractiondf.malware

        extracted_logreg = LogisticRegression(random_state=16, max_iter=20000)
        extracted_logreg.fit(X, y)

    # ---------------------------------------------------extraction
    # ---------------------------------------------------similar

        similardf = pd.DataFrame(similar_training, columns=col_names)
        X = similardf[feature_cols]
        y = similardf.malware

        x_values = X.values
        x_scaled = min_max_scaler.fit_transform(x_values)
        X = pd.DataFrame(x_scaled)
        
        similar_logreg = LogisticRegression(random_state=16, max_iter=20000)
        similar_logreg.fit(X, y)

    # ---------------------------------------------------similar

        prepare_backdoors(2000)
        backdoorsdataset = make_backdoors()

        backdoordf = pd.DataFrame(backdoorsdataset, columns=col_names)

        X_test = backdoordf[feature_cols]

        x_values = X_test.values
        x_scaled = min_max_scaler.fit_transform(x_values)
        X_test = pd.DataFrame(x_scaled)

        extracted_predict_proba = extracted_logreg.predict_proba(X_test.values)

        for x in extracted_predict_proba:
            with open("statistics_extracted_results.txt", 'a') as f:
                result = str(round(float(x[1]),3))
                stat_extracted_list.append(float(x[1]))
                f.write(result+'\n')

        similar_predict_proba = similar_logreg.predict_proba(X_test.values)
        

        for x in similar_predict_proba:
            with open("statistics_similar_results.txt", 'a') as f:
                result = str(round(float(x[1]),3))
                stat_similar_list.append(float(x[1]))
                f.write(result+'\n')

        model_to_be_stolen = original_logreg
#--------------------------------------------
#-------lopás-----------------
#--------------------------------
    separate_training = stolendataset_benign + stolendataset_malware
    random.shuffle(separate_training)
    
    extractiondf = pd.DataFrame(separate_training, columns=col_names)

    X = extractiondf[feature_cols]
    x_values = X.values
    x_scaled = min_max_scaler.fit_transform(x_values)
    X = pd.DataFrame(x_scaled)


    y = model_to_be_stolen.predict(X.values)
    X.insert(0, "malware", np.array(y.tolist()))
    extractiondf = X

    X = extractiondf.iloc[:, 1:132]
    y = extractiondf.malware

    extracted_logreg = LogisticRegression(random_state=16, max_iter=20000)
    extracted_logreg.fit(X, y)

    prepare_backdoors(2000)
    backdoorsdataset = make_backdoors()

    backdoordf = pd.DataFrame(backdoorsdataset, columns=col_names)

    X_test = backdoordf[feature_cols]

    x_values = X_test.values
    x_scaled = min_max_scaler.fit_transform(x_values)
    X_test = pd.DataFrame(x_scaled)

    extracted_predict_proba = extracted_logreg.predict_proba(X_test.values)
    separate_extracted_predict_proba = []    

    for x in extracted_predict_proba:
        with open("separate_extracted_results.txt", 'a') as f:
            result = str(round(float(x[1]),3))
            separate_extracted_predict_proba.append(float(x[1]))
            f.write(result+'\n')

    extracted_div = entropy(separate_extracted_predict_proba, stat_extracted_list)
    similar_div = entropy(separate_extracted_predict_proba, stat_similar_list)


    with open("extracted_divs.txt", 'a') as f:
        f.write(str(extracted_div) + '\n')
    
    with open("similar_divs", 'a') as f:
        f.write(str(similar_div) + '\n')

for i in range (1000):
    print("iteration ", i+1)
    main()