# Problem 7 - Text Clustering

This problem uses a Jupyter Notebook to solve a problem as a part of Introduction to Data Science course.
It helps to understand concepts such as text clustering, vectorizing, removing stop words and finally using
K-means clustering to create clusters of words.



## Installation and libraries

Following libraries need to be installed:

```bash
  pip install gzip-reader
  pip install -U scikit-learn
```

Following libraries need to be imported:

```python
import pandas as pd
import numpy as np
import gzip
from collections import Counter
import re
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
```

## Step-by-step Explanation

Extract the dataset from: http://snap.stanford.edu/data/web-FineFoods.html. As it is a .gz file
we need to specify values compression and encoding.
Extract only the part that contains 'review/text' from the file.
```python
data = pd.read_csv('finefoods.txt.gz', compression='gzip',header=None, sep='\t', encoding='latin-1')

data = data[data[0].str.contains('review/text')]
data
```
#### • Identify all the unique words that appear in the “review/text” field of the reviews (L)

```python
text_data = []
for index,row in data.iterrows():
    res_old = str(row[0])
    res = re.sub(r"[^\w\s :]","", res_old)
    text_data.append(res.split(":")[1].lower().strip())
data['reviews'] = text_data
data.drop(columns=0,inplace=True)

L = []
for text in text_data:
    words = text.split(' ')
    c = Counter(words)
    for w in words:
        if c[w] == 1:
            L.append(w.lower())
len(L)
```

#### • Remove from L all stopwords referrring to http://www.ranks.nl/stopwords (W)

Displaying only first few stop words from the list. P.S.: Here, we generate a list W that includes only words that are not stop
words and also words len>3. This is because, words with len < 3 are not sensible to the decision and
do not affect the count vectorizer. No meaning can be derived from them.
```python
stop_words = ["a", "able", "about", "above", "abst", "accordance", "according", "accordingly", "across ....]
W = [word for word in L if word not in stop_words and len(word) > 3]
```

#### • Count the number of times each word in W appears among all reviews (“review/text” field) and identify the top 500 words.

Using FreqDist() to find a list of words and the occurance of each word. most_common returns
list of tuples that have each word and its occurance count as one tuple.
```python
start = datetime.now()
from nltk import FreqDist
fdist1 = FreqDist(W)
results = Counter(fdist1)
end = datetime.now()
print(end-start)
results = Counter(results)

most_common = results.most_common(501)
most_common = most_common[1:]
most_common
```
![SS1](https://raw.githubusercontent.com/ruchivaria/TextClustering/master/assets/SS1.png)

#### • Vectorize all reviews (“review/text” field) using these 500 words
```python
most_common_words = []
for k,v in results.most_common(501):
    most_common_words.append(k)
most_common_words = most_common_words[1:]

sentence_vectors = []
for sentence in text_data:
    sent_vec = []
    for token in most_common_words:
        if token in sentence:
            sent_vec.append(1)
        else:
            sent_vec.append(0)
    sentence_vectors.append(sent_vec)

sentence_vectors = np.asarray(sentence_vectors)
data_new = pd.DataFrame(sentence_vectors)
```

#### • Cluster the vectorized reviews into 10 clusters using k-means.

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=10, random_state=0).fit(sentence_vectors)
centers = kmeans.cluster_centers_
centers = pd.DataFrame(centers)
centers_columns =list(centers.columns)

col_rename_dict = {i:j for i,j in zip(centers_columns,most_common_words)}
centers.rename(columns=col_rename_dict,inplace=True)
centers
```

#### •From each centroid, select the top 5 words that represent the centroid


```python
centeroid = []
for index,row in centers.iterrows():
    centeroid.append(row)

for i in range(0,10):
    print(centeroid[i].sort_values(ascending=False)[0:5])
```

Using length greater than 3  

![SS2](https://raw.githubusercontent.com/ruchivaria/TextClustering/master/assets/SS2.png)  

Using length less than 3 (did not use these results because they do not seem to be meaninful for text clustering)    
![SS2](https://raw.githubusercontent.com/ruchivaria/TextClustering/master/assets/SS3.png)