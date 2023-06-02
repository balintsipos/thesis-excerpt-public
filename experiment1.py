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

    print(count," backdoors prepared")


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
    print("backdoors created")

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
    for i in range(1000):
        #prepare_temp_folders()
        prepare_backdoors(30)
        #prepare_benign_training_sets()
        #prepare_malware_training_sets()
        backdoors = make_backdoors()
        allbenigns = hash_benign()
        allmalware = hash_malware()

        training_benign = allbenigns[:2000]
        extraction_benign = allbenigns[2000:3000]
        similar_benign = allbenigns[3000:]

        training_malware = allmalware[:2000]
        extraction_malware = allmalware[2000:3000]
        similar_malware = allmalware[3000:]


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
        y_test = original_predictdf.malware
        x_values = X_test.values
        x_scaled = min_max_scaler.fit_transform(x_values)
        X_test = pd.DataFrame(x_scaled)

        original_predict = original_logreg.predict(X_test.values)

        correct = 0
        x_values = X.values
        x_scaled = min_max_scaler.fit_transform(x_values)
        X = pd.DataFrame(x_scaled)
        for i in range(0, 1000):
            if original_predict[i] == y_test.values[i]: correct += 1
        print("original accuracy: ", correct/1000)

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

        print("all models created")
    # ---------------------------------------------------similar

        prepare_backdoors(2000)
        backdoorsdataset = make_backdoors()

        backdoordf = pd.DataFrame(backdoorsdataset, columns=col_names)

        X_test = backdoordf[feature_cols]
        y_test = backdoordf.malware

        x_values = X_test.values
        x_scaled = min_max_scaler.fit_transform(x_values)
        X_test = pd.DataFrame(x_scaled)

        extracted_predict = extracted_logreg.predict(X_test.values)
        
        benign_count = 0
        malware_count = 0

        for i in range(0, 1000):
            if extracted_predict[i] == 0: benign_count += 1
            else: malware_count += 1
        print("nr. of benign: ", benign_count, ", nr. of malware : ",malware_count)
        print("extracted accuracy: ", malware_count/(benign_count+malware_count))

        result = str(malware_count/(benign_count+malware_count))

        similar_predict = similar_logreg.predict(X_test.values)

        benign_count = 0
        malware_count = 0

        for i in range(0, 1000):
            if similar_predict[i] == 0: benign_count += 1
            else: malware_count += 1
        print("nr. of benign: ", benign_count, ", nr. of malware: ",malware_count)
        print("similar accuracy: ", malware_count/(benign_count+malware_count))

        result = result + "," + str(malware_count/(benign_count+malware_count))
        with open("results.txt", 'a') as f:
            f.write(result+'\n')

main()