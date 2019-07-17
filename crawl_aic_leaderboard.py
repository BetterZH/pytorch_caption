import requests
from bs4 import BeautifulSoup
import datetime
import os
import csv
import numpy as np
import json
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

def parse(html):
    records = []
    soup = BeautifulSoup(html, "lxml")
    table = soup.find('table')
    trs = table.find_all('tr')
    head = trs[0]
    tds = head.find_all('th')
    records.append([elem.text.encode('utf-8') for elem in tds])
    for row_idx,tr in enumerate(trs[1:31]):
        tds = tr.find_all('td')
        rank = int(tds[0].find('span').text)
        tn = tds[1].find('span').text.encode('utf-8')
        rec = [rank, tn]
        for elem in tds[3:-2]:
            rec.append(float(elem.text))
        rec.append(int(tds[-2].text))
        rec.append(tds[-1].text.encode('utf-8'))
        records.append(rec)
    return records

def scrape_table():
    sess = requests.session()
    login_url = "https://challenger.ai/competition/caption/leaderboard"
    html = sess.get(login_url).text
    records = parse(html)
    sess.close()
    return records

def get_json():
    json_url = 'https://challenger.ai/c/leaderboard?from=0&size=500&cid=1&round=0&type=caption'
    sess = requests.session()
    json_str = sess.get(json_url).text
    sess.close()
    out_path = ''
    filename = datetime.datetime.now().strftime("%Y-%m-%d_%Hh%Mm")+'.json'
    with open(os.path.join(out_path, filename), "w") as f:
        f.write(json_str)
    return json_str

def write_to_csv(records):
    out_path = ''
    filename = datetime.datetime.now().strftime("%Y-%m-%d_%Hh%Mm")+'.csv'
    with open(os.path.join(out_path, filename), "w") as f:
        wr = csv.writer(f)
        wr.writerows(records)

def get_scores_from_json(json_array):
    scores = {}
    np_scores = {}
    metrics = json_array[0]['score_extra'].keys()
    for metric in metrics:
        scores[metric] = []
    scores['score'] = []
    for ja in json_array:
        for metric in metrics:
            scores[metric].append(float(ja['score_extra'][metric]))
        scores['score'].append(float(ja['score']))
    metrics.append('score')
    for metric in metrics:
        np_scores[metric] =np.asarray(scores[metric])
    np_scores['Bleu_cal'] = (np_scores['Bleu_1'] + np_scores['Bleu_2'] +np_scores['Bleu_3'] + np_scores['Bleu_4'])/4.
    metrics.append('Bleu_cal')
    return np_scores, metrics

def get_normed_score(scores, metrics):
    scores_normed = {}
    std_vars = {}
    for metric in metrics:
        scores_normed[metric] = []
        std_vars[metric] = []
    for metric in metrics:
        score =  np.asarray(scores[metric])
        std_var = np.sqrt(np.sum((score-np.mean(score))**2)/(score.shape[0]-1))
        score_normed = score/std_var
        scores_normed[metric] = score_normed
        std_vars[metric] = std_var

    return scores_normed, std_vars

def cal_final_score(scores_normed, metrics):
    sum = 0
    for metric in ['Bleu', 'CIDEr', 'ROUGE_L', 'METEOR']:
        sum += scores_normed[metric]
    mean_score = sum / 4
    # mean_score = (scores_normed['Bleu'] + scores_normed['CIDEr'] + scores_normed['ROUGE_L'] + scores_normed['METEOR'])/4
    final_score = mean_score/np.sqrt(np.sum((mean_score-np.mean(mean_score))**2)/(mean_score.shape[0]-1))
    print(np.sqrt(np.sum((mean_score-np.mean(mean_score))**2)/(mean_score.shape[0]-1)))
    return final_score, mean_score

def write_scores_normed_to_csv(scores_normed):

    records = []

    metrics = ['Bleu', 'CIDEr', 'ROUGE_L', 'METEOR', 'score']
    records.append([''] + metrics)

    len = scores_normed['CIDEr'].shape[0]
    for i in range(len):
        record = []
        record.append(str(i+1))
        for metric in metrics:
            record.append(scores_normed[metric][i])
        records.append(record)

    out_path = ''
    filename = datetime.datetime.now().strftime("%Y-%m-%d_%Hh%Mm")+'_data.csv'
    with open(os.path.join(out_path, filename), "w") as f:
        wr = csv.writer(f)
        wr.writerows(records)


def download_data():
    records = scrape_table()
    write_to_csv(records)
    json_str = get_json()
    json_array = json.loads(json_str)['data']
    scores, metrics = get_scores_from_json(json_array)
    scores_normed, std_vars = get_normed_score(scores, metrics)
    final_score, mean_score = cal_final_score(scores_normed, metrics)
    print final_score
    print scores['score']
    print mean_score
    print std_vars

    for key in std_vars:
        print(key, 1 / std_vars[key])

    write_scores_normed_to_csv(scores_normed)

def linear_data():
    data = pd.read_csv('2017-10-22_16h30m_data.csv', index_col=0)
    print(data.head())
    print(data.tail())
    print(data.shape)

    metrics = ['Bleu', 'CIDEr', 'ROUGE_L', 'METEOR']
    X = data[metrics]

    y = data['score']

    print(X.head())
    print(y.head())

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    print X_train.shape
    print y_train.shape
    print X_test.shape
    print y_test.shape

    linreg = LinearRegression()

    linreg.fit(X_train, y_train)

    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)

    print(linreg.intercept_)
    print(zip(metrics, linreg.coef_))


def predict(bleu, cider, meteor, rouge):

    score = 4.7706 * bleu + 0.3565 * cider + 2.8256 * meteor + 1.7533 * rouge

    return score

if __name__ == '__main__':
    # download_data()
    # linear_data()

    score = predict(0.71773,1.95275,0.41512,0.69962)
    print(score)


