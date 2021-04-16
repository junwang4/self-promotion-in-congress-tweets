import os, sys, time, re, json
import numpy as np
import pandas as pd
from IPython.display import display
from configparser import ConfigParser
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from bert_sklearn import BertClassifier, load_model

pd.options.display.max_colwidth = 80
pd.options.display.width = 1000
pd.options.display.precision = 3
np.set_printoptions(precision=3)

LABEL_NAME = {0:'NOT', 1:'YES'}
NUM_CLASSES = len(LABEL_NAME)

LABEL_COLUMN_NAME = 'label'
TEXT_COLUMN_NAME = 'text'

config = ConfigParser()
config.read('settings.ini')

BERT_MODEL = config.get('common', 'BERT_MODEL')
print('- BERT model:', BERT_MODEL)

K_FOLDS = config.getint('common', 'K_FOLDS')
EPOCHS = config.getint('common', 'EPOCHS')

MAX_SEQ_LENGTH = config.getint('common', 'MAX_SEQ_LENGTH')
TRAIN_BATCH_SIZE = config.getint('common', 'TRAIN_BATCH_SIZE')
LEARNING_RATE = config.getfloat('common', 'LEARNING_RATE')

RANDOM_STATE = config.getint('common', 'RANDOM_STATE')
SAVE_MODEL_FILE_FOR_EACH_TRAINING_FOLD = config.getboolean('common', 'SAVE_MODEL_FILE_FOR_EACH_TRAINING_FOLD')

HTML_FOLDER_OF_ERROR_ANALYSIS_OUTPUT = config.get('common', "HTML_FOLDER_OF_ERROR_ANALYSIS_OUTPUT")

data_version = config.get('common', 'DATA_VERSION')

ANNOTATED_FILE = config.get(data_version, "ANNOTATED_FILE")
DATA_FILE_TO_WORK = config.get(data_version, "DATA_FILE_TO_WORK")
TAG = config.get(data_version, "TAG")
print('- data version:', data_version, '   TAG:', TAG)

def get_or_create_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder


class TweetClassifier:
    def __init__(self):
        self.kfolds = K_FOLDS
        self.bert_model = BERT_MODEL

        self.data_folder = '../data'
        self.working_folder = get_or_create_dir(f'working')
        self.model_folder = get_or_create_dir(f'{self.working_folder}/model')
        self.pred_folder = get_or_create_dir(f'{self.working_folder}/pred')

        self.annotated_file = ANNOTATED_FILE
        self.data_file_to_work = DATA_FILE_TO_WORK


    def get_train_data_csv_fpath(self):
        fpath = f'{self.data_folder}/{self.annotated_file}'
        print('- annotated csv file:', fpath)
        if os.path.exists(fpath):
            return fpath
        else:
            print('- error: training csv file not exists:', fpath)
            sys.exit()

    def read_train_data(self):
        return pd.read_csv(self.get_train_data_csv_fpath(), usecols=[TEXT_COLUMN_NAME, LABEL_COLUMN_NAME], encoding = 'utf8', keep_default_na=False)

    def get_model_bin_file(self, fold=0):
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)
            print(f'\ncreate a new folder for storing BERT model: "{self.model_folder}"\n')
        if fold>=0:
            return f'{self.model_folder}/{TAG}_K{self.kfolds}_epochs{EPOCHS}_{fold}.bin'
        elif fold==-1:
            return f'{self.model_folder}/{TAG}_full_epochs{EPOCHS}.bin'
        else:
            print('- wrong value for fold:', fold)
            sys.exit()

    def get_pred_csv_file(self, mode='train'):
        if mode == 'train':
            fpath = f'{self.pred_folder}/{TAG}_{mode}_K{self.kfolds}_epochs{EPOCHS}.csv'
        elif mode == 'apply':
            fpath = f'{self.pred_folder}/{TAG}_{mode}_epochs{EPOCHS}.csv'
        else:
            print('- wrong mode:', mode, '\n')
            sys.exit()
        print('- get pred csv file:', fpath)
        return fpath

    def get_train_test_data(self, df, fold=0):
        df[TEXT_COLUMN_NAME] = df[TEXT_COLUMN_NAME].apply(lambda x: x.strip())
        kf = StratifiedKFold(n_splits=self.kfolds, shuffle=True, random_state=RANDOM_STATE)
        cv = kf.split(df[TEXT_COLUMN_NAME], df[LABEL_COLUMN_NAME])

        for i, (train_index, test_index) in enumerate(cv):
            if i == fold:
                break
        train = df.iloc[train_index]
        test = df.iloc[test_index]

        print(f"\nALL: {len(df)}   TRAIN: {len(train)}   TEST: {len(test)}")
        label_list = np.unique(train[LABEL_COLUMN_NAME])
        return train, test, label_list

    def train_model(self, df_train, model_file_to_save, val_frac=0.1):
        X_train = df_train[TEXT_COLUMN_NAME]
        y_train = df_train[LABEL_COLUMN_NAME]

        model = BertClassifier(bert_model=self.bert_model, random_state=RANDOM_STATE, \
                                max_seq_length=MAX_SEQ_LENGTH, \
                                train_batch_size=TRAIN_BATCH_SIZE, learning_rate=LEARNING_RATE, \
                                epochs=EPOCHS, validation_fraction=val_frac)
        print(model)
        model.fit(X_train, y_train)

        if model_file_to_save:
            model.save(model_file_to_save)
            print(f'\n- model saved to: {model_file_to_save}\n')
        return model

    def train_one_full_model(self):
        df_train = self.read_train_data()

        model_file_to_save = self.get_model_bin_file(fold=-1) # -1: for one full model
        val_frac = 0.0
        self.train_model(df_train, model_file_to_save, val_frac=val_frac)

    def train_KFold_model(self):
        df = self.read_train_data()
        print('- label value counts:')
        print(df[LABEL_COLUMN_NAME].value_counts())

        y_test_all, y_pred_all = [], []
        results = []
        df_out_proba = None
        for fold in range(self.kfolds):
            train_data, test_data, label_list = self.get_train_test_data(df, fold)

            if SAVE_MODEL_FILE_FOR_EACH_TRAINING_FOLD:
                model_file = self.get_model_bin_file(fold)
            else:
                model_file = ''

            val_frac = 0.05
            model = self.train_model(train_data, model_file, val_frac=val_frac)

            X_test = test_data[TEXT_COLUMN_NAME]
            y_test = test_data[LABEL_COLUMN_NAME]
            y_test_all += y_test.tolist()

            y_proba = model.predict_proba(X_test)
            del model

            tmp = pd.DataFrame(data=y_proba, columns=[f'c{i}' for i in range(NUM_CLASSES)])
            tmp['confidence'] = tmp.max(axis=1)
            tmp['winner'] = tmp.idxmax(axis=1)
            tmp[TEXT_COLUMN_NAME] = X_test.tolist()
            tmp[LABEL_COLUMN_NAME] = y_test.tolist()
            df_out_proba = tmp if df_out_proba is None else pd.concat((df_out_proba, tmp))

            y_pred = [int(x[1]) for x in tmp['winner']]
            y_pred_all += y_pred

            acc = accuracy_score(y_pred, y_test)
            res = precision_recall_fscore_support(y_test, y_pred, average='macro')
            print(f'\nAcc: {acc:.3f}      F1:{res[2]:.3f}       P: {res[0]:.3f}   R: {res[1]:.3f} \n')

            item = {'Acc': acc, 'weight': len(test_data)/len(df), 'size': len(test_data)}
            item.update({'P':res[0], 'R':res[1], 'F1':res[2]})
            for cls in np.unique(y_test):
                res = precision_recall_fscore_support(y_test, y_pred, average=None, labels=[cls])
                for i, scoring in enumerate('P R F1'.split()):
                    item['{}_{}'.format(scoring, cls)] = res[i][0]
            results.append(item)

            acc_all = np.mean(np.array(y_pred_all) == np.array(y_test_all))
            res = precision_recall_fscore_support(y_test_all, y_pred_all, average='macro')
            print( f'\nAVG of {fold+1} folds  |  Acc: {acc_all:.3f}    F1:{res[2]:.3f}       P: {res[0]:.3f}   R: {res[1]:.3f} \n')

        # show an overview of the performance
        df_2 = pd.DataFrame(list(results)).transpose()
        df_2['avg'] = df_2.mean(axis=1)
        df_2 = df_2.transpose()
        df_2['size'] = df_2['size'].astype(int)
        display(df_2)

        # put together the results of all 5-fold tests and save
        output_pred_csv_file_train = self.get_pred_csv_file(mode='train')
        df_out_proba.to_csv(output_pred_csv_file_train, index=False, float_format="%.3f")
        print(f'\noutput all {self.kfolds}-fold test results to: "{output_pred_csv_file_train}"\n')


    def apply_one_full_model_to_new_sentences(self):
        fpath_data = f'{self.data_folder}/{self.data_file_to_work}'

        nrows = None
        #nrows = 1000
        df = pd.read_csv(fpath_data, nrows=nrows, keep_default_na=False, dtype={'id':str}, usecols='id text'.split())
        #id,date,text,has_i

        # NOTE: df is replaced with new data
        print('- cnt:', len(df), '- unique ids:', len(df.id.unique()))
        print(df[:2], '\n')

        output_pred_file = self.get_pred_csv_file('apply')
        print('\n- predictions to save to:', output_pred_file)

        model_file = self.get_model_bin_file(fold=-1)  # -1: indicating this is the model trained on all data
        print(f'\n- use trained model: {model_file}\n')

        model = load_model(model_file)

        model.eval_batch_size = 32
        y_prob = model.predict_proba(df.text)

        df_out = pd.DataFrame(data=y_prob, columns=[f'c{i}' for i in range(NUM_CLASSES)])
        df_out['confidence'] = df_out.max(axis=1)
        df_out['winner'] = df_out.idxmax(axis=1)
        #for col in df.columns:
        #    df_out[col] = df[col]
        df_out['id'] = df['id']

        df_out.to_csv(output_pred_file, index=False, float_format="%.3f")
        print(f'\n- predictions saved to: {output_pred_file}\n')


    def evaluate_and_error_analysis(self):
        df = pd.read_csv(self.get_pred_csv_file(mode='train')) # -2: a flag indicating putting together the results on all folds
        df['pred'] = df['winner'].apply(lambda x:int(x[1])) # from c0->0, c1->1, c2->2, c3->3

        print('\nConfusion Matrix:\n')
        cm = confusion_matrix(df[LABEL_COLUMN_NAME], df.pred)
        print(cm)

        print('\n\nClassification Report:\n')
        print(classification_report(df[LABEL_COLUMN_NAME], df.pred, digits=3))

        out = ["""
<style>
* {font-family:arial}
body {width:900px;margin:auto}
.wrong {color:red;}
.hi1 {font-weight:bold}
</style>
<div><table cellpadding=10>
    """]

        row = f'<tr><th><th><th colspan=4>Predicted</tr>\n<tr><td><td>'
        label_name = LABEL_NAME
        for i in range(NUM_CLASSES):
            row += f"<th>{label_name[i]}"
        for i in range(NUM_CLASSES):
            row += f'''\n<tr>{'<th rowspan=4>Actual' if i==0 else ''}<th align=right>{label_name[i]}'''
            for j in range(NUM_CLASSES):
                row += f'''<td align=right><a href='#link{i}{j}'>{cm[i][j]}</a></td>'''
        out.append(row + "</table>")

        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                row = f"<div id=link{i}{j}><h2>{label_name[i]} => {label_name[j]}</h2><table cellpadding=10>"
                label_names = ' '.join([f'<th>{label_name[k]}</th>' for k in range(len(label_name))])
                row += f'<tr> <th></th> <th>Sentence</th> <th>Label</th> {label_names} <th>mark</th> </tr>'

                out.append(row)

                df_ = df[(df[LABEL_COLUMN_NAME]==i) & (df.pred==j)]
                df_ = df_.sort_values('confidence', ascending=False)

                cnt = 0
                for idx, row in df_.iterrows():
                    sentence, label, pred = row[TEXT_COLUMN_NAME], row[LABEL_COLUMN_NAME], row['pred']
                    cnt += 1
                    td_mark = "" if label == pred else "<span class=wrong>oops</span>"

                    td_confidence_list = []
                    c_max = max([row[f'c{k}'] for k in range(NUM_CLASSES)])
                    for k in range(NUM_CLASSES):
                        c = row[f'c{k}']
                        is_max = int(c >= c_max)
                        td_confidence_list.append(f'<td valign=top class=hi{is_max}>{c:.2f}</td>')

                    item = f"""<tr><th valign=top>{cnt}.
                        <td valign=top width=70%>{sentence}
                        <td valign=top>{label_name[label]}
                        {''.join(td_confidence_list)}
                        <td valign=top>{td_mark}</tr>"""
                    out.append(item)

                out.append('</table></div>')

        fpath_out = f'{get_or_create_dir(HTML_FOLDER_OF_ERROR_ANALYSIS_OUTPUT)}/error_analysis_{TAG}.html'
        with open(fpath_out, 'w') as fout:
            fout.write('\n'.join(out))
            print(f'\n- Error analysis result saved to: "{fpath_out}"\n')


def main():
    clf = TweetClassifier()

    clf.train_KFold_model()
    #clf.evaluate_and_error_analysis()

    #clf.train_one_full_model()
    #clf.apply_one_full_model_to_new_sentences()


if __name__ == "__main__":
    tic = time.time()
    main()
    print(f'\ntime used: {time.time()-tic:.1f} seconds\n')

