Sentiment Analysis Report
=========================

Содержание

[toc]

# Sentiment analysis русскоязычных твитов при помощи TensorFlow.

Ссылка на [github](https://github.com/b0noI/ml-lessons/tree/master/sentiments_rus) и [видео](https://www.youtube.com/watch?v=CDpbJIbDhys&list=PLsQAG1V_t58AmRRhu-UUKvDSy-gY3Vmdh&index=4).

**Разбор кода:**  

## <center>Imports</center>

```python
import pandas as pd
import numpy as np
import tensorflow as tf
import tflearn
import re

from collections import Counter
from sklearn.model_selection import train_test_split
from tflearn.data_utils import to_categorical
from nltk.stem.snowball import RussianStemmer
from nltk.tokenize import TweetTokenizer
```

## <center>Constants</center>

```python
POSITIVE_TWEETS_CSV = 'positive.csv' # имя файла, где хранятся позитивные комментарии
NEGATIVE_TWEETS_CSV = 'negative.csv' # имя файла, где хранятся негативные комментарии

VOCAB_SIZE = 5000 # размер словаря
```

## <center>Load data</center>

Загружаем данные в оперативную память (с помощью пандас).

> :information_source: Данные можно найти по ссылке [github](https://github.com/b0noI/ml-lessons/tree/master/sentiments_rus).

```python
tweets_col_number = 3

# список всех негативных и позитивных твитов

negative_tweets = pd.read_csv(
    'negative.csv', header=None, delimiter=';')[[tweets_col_number]]
positive_tweets = pd.read_csv(
    'positive.csv', header=None, delimiter=';')[[tweets_col_number]]
print('negative_tweets: ',negative_tweets)
print('positive_tweets: ',positive_tweets)
```
**Output:**  
```
negative_tweets:                                                          3
0       на работе был полный пиддес :| и так каждое за...
1       Коллеги сидят рубятся в Urban terror, а я из-з...
2       @elina_4post как говорят обещаного три года жд...
3       Желаю хорошего полёта и удачной посадки,я буду...
4       Обновил за каким-то лешим surf, теперь не рабо...
...                                                   ...
111918  Но не каждый хочет что то исправлять:( http://...
111919  скучаю так :-( только @taaannyaaa вправляет мо...
111920          Вот и в школу, в говно это идти уже надо(
111921  RT @_Them__: @LisaBeroud Тауриэль, не грусти :...
111922  Такси везет меня на работу. Раздумываю приплат...

[111923 rows x 1 columns]
positive_tweets:                                                          3
0       @first_timee хоть я и школота, но поверь, у на...
1       Да, все-таки он немного похож на него. Но мой ...
2       RT @KatiaCheh: Ну ты идиотка) я испугалась за ...
3       RT @digger2912: "Кто то в углу сидит и погибае...
4       @irina_dyshkant Вот что значит страшилка :D\nН...
...                                                   ...
114906  Спала в родительском доме, на своей кровати......
114907  RT @jebesilofyt: Эх... Мы немного решили сокра...
114908  Что происходит со мной, когда в эфире #proacti...
114909  "Любимая,я подарю тебе эту звезду..." Имя како...
114910  @Ma_che_rie посмотри #непытайтесьпокинутьомск ...

[114911 rows x 1 columns]
```

## <center>Stemmer</center>

Стеммер - модуль, который находит для каждого слова его базовую единицу.  
Например, для собака/собакой/собаке будет одно базовое слово "собак". (Чтобы при новом падеже слова, нейронная сеть не игнорировала слово)

```python
stemer = RussianStemmer()

# отсеивание не кириллических символов
# т.к. во многих твитах есть смайлы, и мл запоминает, что если есть смайл, то позитивный, если нет, то негативный твит

regex = re.compile('[^а-яА-Я ]')
stem_cache = {}

# token это единичка
def get_stem(token):
    stem = stem_cache.get(token, None) # из кэша достает результат работы стэма
    if stem: # если в кэш есть этот стеммер, то возвращает
        return stem
    token = regex.sub('', token).lower() # если стэма нет, то отрезает не кириллические символы
    stem = stemer.stem(token) # берет стэм
    stem_cache[token] = stem # записывает в кэш
    return stem
```

## <center>Vocabulary creation</center>

Создадим словарь.

```python
stem_count = Counter()
tokenizer = TweetTokenizer()

# словарик - уникальные слова, которые встречались в твитах

def count_unique_tokens_in_tweets(tweets): # получаем серию твитов
    for _, tweet_series in tweets.iterrows(): # пробегаемся по каждому твиту
        tweet = tweet_series[3]
        tokens = tokenizer.tokenize(tweet) # токинизируем (разбитие на слова, на части)
        print('tokens: ',tokens)
        for token in tokens: # для каждого токена берем стэм
            stem = get_stem(token)
            stem_count[stem] += 1 # добавляем 1
        print('stem: ',stem)

count_unique_tokens_in_tweets(negative_tweets) # кол-во негативных
count_unique_tokens_in_tweets(positive_tweets) # кол-во позитивных твитов
```
**Output:**  
```
tokens:  ['на', 'работе', 'был', 'полный', 'пиддес', ':|', 'и', 'так', 'каждое', 'закрытие', 'месяца', ',', 'я', 'же', 'свихнусь', 'так', 'D:']
stem:  
tokens:  ['Коллеги', 'сидят', 'рубятся', 'в', 'Urban', 'terror', ',', 'а', 'я', 'из-за', 'долбанной', 'винды', 'не', 'могу', ':(']
stem:  
tokens:  ['@elina_4post', 'как', 'говорят', 'обещаного', 'три', 'года', 'ждут', '...', '(', '(']
stem:  
tokens:  ['Желаю', 'хорошего', 'полёта', 'и', 'удачной', 'посадки', ',', 'я', 'буду', 'очень', 'сильно', 'скучать', '(', 'http://t.co/jCLNzVNv3S']
stem:  
tokens:  ['Обновил', 'за', 'каким-то', 'лешим', 'surf', ',', 'теперь', 'не', 'работает', 'простоплеер', ':(']
stem:  
tokens:  ['Котёнка', 'вчера', 'носик', 'разбила', ',', 'плакала', 'и', 'расстраивалась', ':(']
stem:  
tokens:  ['@juliamayko', '@O_nika55', '@and_Possum', 'Зашли', ',', 'а', 'то', 'он', 'опять', 'затихарился', ',', 'я', 'прямо', 'физически', 'страдаю', ',', 'когда', 'он', 'долго', 'молчит', '!', '(', '(', '(']
stem:  
```

Отсортируем по полуряности (частоте) и возьмем первые 5000. По причине того, что нам не нужно нейронной сети отправлять стэмы, которые встречались очень мало (т.к. нейронная сеть не сможет определить, как эти стэмы влияют на результат). 
Нейронная связь может запомнить, что, к примеру, стэм был в отрцательном твите, то и в следующем твите будет негативный, не проанализировав полностью твит.

> :information_source: Нейронная связь, как студент, пытается "схалтурить".

```python
vocab = sorted(stem_count, key=stem_count.get, reverse=True)[:VOCAB_SIZE]
print(vocab[:100])
```
**Output:**  
```
['', 'не', 'я', 'и', 'в', 'на', 'а', 'что', 'так', 'с', 'эт', 'как', 'у', 'мен', 'мне', 'все', 'но', 'он', 'ты', 'теб', 'ну', 'мо', 'то', 'уж', 'по', 'был', 'ещ', 'за', 'да', 'вот', 'же', 'тольк', 'нет', 'сегодн', 'о', 'прост', 'бы', 'над', 'когд', 'хоч', 'очен', 'к', 'сам', 'ден', 'будет', 'мы', 'от', 'хорош', 'из', 'есл', 'тепер', 'тож', 'буд', 'сво', 'год', 'даж', 'завтр', 'нов', 'дом', 'до', 'там', 'ест', 'вообщ', 'ег', 'вс', 'дела', 'пот', 'одн', 'для', 'больш', 'хот', 'спасиб', 'мог', 'сейчас', 'е', 'себ', 'нас', 'блин', 'раз', 'кто', 'дума', 'утр', 'котор', 'любл', 'поч', 'зна', 'говор', 'лучш', 'нич', 'без', 'ил', 'вы', 'друг', 'тут', 'чтоб', 'всем', 'бол', 'люд', 'сдела', 'сказа']
```

*Пример:*
```python
idx = 2
print("stem: {}, count: {}"
      .format(vocab[idx], stem_count.get(vocab[idx])))
```
**Output:**  
```
stem: я, count: 66045
```

Создадим словарь, где каждой стемме, противопоставим число, чтобы преобразовать токен в число.

```python
token_2_idx = {vocab[i] : i for i in range(VOCAB_SIZE)} # противопоставим числа до размера словаря
len(token_2_idx)
```
**Output:**  
```
5000
```

*Пример:*
```python
token_2_idx['сказа'] # 
```
**Output:**  
```
99
```
Преобразование целого твита в вектор 0 и 1 (вектор будет передаваться на вход нейросети).

```python
def tweet_to_vector(tweet, show_unknowns=False):
    vector = np.zeros(VOCAB_SIZE, dtype=np.int_) # создается вектор из нулей, который имеет длину словаря VOCAB_SIZE
    for token in tokenizer.tokenize(tweet): # токенезирует твит
        stem = get_stem(token) # для каждого токера берет стэмму
        idx = token_2_idx.get(stem, None) # получает индекс этой стэммы, если нет в словаре, то возвращает None
        if idx is not None: # если индекс найден, ставим 1
            vector[idx] = 1
        elif show_unknowns: # для дебагинга, показывает ненайденный стэм
            print("Unknown token: {}".format(token))
    return vector
```

*Пример:*
```python
# проверим
# какой-нибудь твит по индексу 3
tweet = negative_tweets.iloc[1][3]
# выведем на экран твит
print("tweet: {}".format(tweet))
# и первые 10 элементов вектора
print("vector: {}".format(tweet_to_vector(tweet)[:10]))
print(vocab[5])
```
**Output:**  
```
tweet: Коллеги сидят рубятся в Urban terror, а я из-за долбанной винды не могу :(
vector: [1 1 1 0 1 0 1 0 0 0]
на
```
## <center>Converting Tweets to vectors</center>

Cоздаем огромную двумерную матрицу, в ширину = длина словаря, в длину = размер количества твитов.
В оперативной памяти преобразуем все твиты в векторы 0 и 1.
Т.к. для обучения, нужны все твиты сразу (несколько 100 тыс).

```python
tweet_vectors = np.zeros(
    (len(negative_tweets) + len(positive_tweets), VOCAB_SIZE), 
    dtype=np.int_)
tweets = []
for ii, (_, tweet) in enumerate(negative_tweets.iterrows()):
    tweets.append(tweet[3])
    tweet_vectors[ii] = tweet_to_vector(tweet[3])
for ii, (_, tweet) in enumerate(positive_tweets.iterrows()):
    tweets.append(tweet[3])
    tweet_vectors[ii + len(negative_tweets)] = tweet_to_vector(tweet[3])
tweet_vectors

# это входные твиты
```
**Output:**  
```
array([[1, 0, 1, ..., 0, 0, 0],
       [1, 1, 1, ..., 0, 0, 0],
       [1, 0, 0, ..., 0, 0, 0],
       ...,
       [1, 0, 0, ..., 0, 0, 0],
       [1, 0, 1, ..., 0, 0, 0],
       [1, 0, 1, ..., 0, 0, 0]])
```

## <center>Preparing labels</center> 

label - это вектор, который состоит из нулей (длиной негативных твитов) + единичек (длиной позитивных твитов).
```python
labels = np.append(
    np.zeros(len(negative_tweets), dtype=np.int_), 
    np.ones(len(positive_tweets), dtype=np.int_))

# label это одномерный вектор, в каждой ячейке или 0, или 1
```
Начало вектора
```python
labels[:10] # начинается нулями
```
**Output:**  
```
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
```
Конец вектора
```python
labels[-10:] # заканчивается единичками
```
**Output:**  
```
array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
```

## <center>Preparing the data for the training</center>

Последняя преподготовка перед обучением сети.
(положено переименовывать в Х и у)
Х = input
y = output
(т.к. функция преобразуется из х в у)

```python
X = tweet_vectors
y = to_categorical(labels, 2) # преобразование label в категориальную дату
# выход имеет либо 1 0, либо 0 1
# если в label был 0, то выдаст 0 1, если 1, то 1 0

# рандомно разбиваем Х, у на тестовые и тренинговые подмножества
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
```

Важно взять небольшую часть твитов на тестирование, которую сеть никогда не видела, и на них проверять качество обучения сети (т.к. сеть может запомнить все предыдущие твит).

```python
# первые 10 элеметов у
print(y_test[:10])
```
**Output:**  
```
[[1. 0.]
 [1. 0.]
 [1. 0.]
 [1. 0.]
 [0. 1.]
 [1. 0.]
 [1. 0.]
 [1. 0.]
 [0. 1.]
 [1. 0.]]
```

## <center>Building the NN</center> 

Начнем использовать tensorflow
и строить нейросеть.
```python
def build_model(learning_rate=0.1): # подаем скорость обучения сети
    # создаем новый граф нейросети
    tf.compat.v1.reset_default_graph() #tf.reset_default_graph()
    
    # создаем входной слой, передаем 2 значения
    # VOCAB_SIZE - кол-во нейронов (у нас он равен размеру словаря)
    # None - размер батча, в тенсорфлоу можно проводить обучение, одновременно подавая несколько твитов на вход
    # default batch имеет размер 128 элементов, менять не будем
    net = tflearn.input_data([None, VOCAB_SIZE]) 
    
    # создаем один уровень со 125 нейронами
    # fully_connected lare означает, что каждый нейрон этого уровня соединен с каждым из предыдущих
    net = tflearn.fully_connected(net, 125, activation='ReLU')
    
    # еще один связанный уровень с 25 нейронами
    net = tflearn.fully_connected(net, 25, activation='ReLU')
    
    # выходной уровень с 2 нейронами
    net = tflearn.fully_connected(net, 2, activation='softmax')
    # регрессия - алгоритм обучения нашей нейросети
    regression = tflearn.regression(
        net, 
        optimizer='sgd', 
        learning_rate=learning_rate, 
        loss='categorical_crossentropy')
    
    # создаем финальную версию нашей сети и возвращаем наружу
    model = tflearn.DNN(net)
    return model
```

```python
model = build_model(learning_rate=0.75)
```

Начало обучения. Эмпирическим методом установлено, что для нормального обучения необходимо около 30 эпох
т.е. итераций обучения (кол-во прогонов).

```python
model.fit(
    X_train, 
    y_train, 
    validation_set=0.1, 
    show_metric=True, 
    batch_size=128, 
    n_epoch=30)
```
**Output:**  
```
Training Step: 33509  | total loss: 0.50075 | time: 14.289s
| SGD | epoch: 030 | loss: 0.50075 - acc: 0.9024 -- iter: 142848/142904
Training Step: 33510  | total loss: 0.48763 | time: 15.666s
| SGD | epoch: 030 | loss: 0.48763 - acc: 0.8919 | val_loss: 0.98446 - val_acc: 0.6822 -- iter: 142904/142904
--
```

* epoch - номер эпохи
* iter - кол-во итераиий в этой эпохе (1 итерация это полный цикл обучения)
* acc - (accuracy) точность, которую нейросеть предугадывает твиты (если = 1, значит нейросеть нашла какое-то простое правило и неправильно считает)
* total loss - функция потери

> :information_source: acc = 0.93 => не бывает такой точности у человека.
Вероятнее, в этой сетке больше нейронов, чем нужно, умнее чем надо. 
Если acc близко к 1, это означает, что нейросеть нашла какое-то несложное правило.
    
    
## <center>Testing</center>

Выполним тестирование и выясним, какое будет accuracy для тестовой выборки.
```python
predictions = (np.array(model.predict(X_test))[:,0] >= 0.5).astype(np.int_)
accuracy = np.mean(predictions == y_test[:,0], axis=0)
print("Accuracy: ", accuracy)
```
**Output:**  
```
Accuracy:  0.6814154090314617
```
acc равен около 0.7, следовательно 20% (93% train acc - 70% test acc) это переобучение нашей сети.

> :warning: Цель обучения, чтобы нейросеть нашла общее правило (формулу), вместо запоминания всех вариантов ответа.

Теперь сделаем такой метод, который получает на вход твит.
```python
def test_tweet(tweet):
    tweet_vector = tweet_to_vector(tweet, True) # преобразовывает в вектор
    positive_prob = model.predict([tweet_vector])[0][1] # просит нашу сетку предсказать
    print('Original tweet: {}'.format(tweet)) # выводит оригинальный твит
    print('P(positive) = {:.5f}. Result: '.format(positive_prob), 
          'Positive' if positive_prob > 0.5 else 'Negative') # пишет позитивный/негативный результат
```

```python
def test_tweet_number(idx):
    test_tweet(tweets[idx])
```

*Пример:*
```
test_tweet_number(120705)
```
**Output:**  
```
Unknown token: обладает
Unknown token: извлечь
Unknown token: выгоду
Original tweet: Он, якобы, обладает информацией, и может извлечь из нее выгоду. ::-) #RU_FF #FF_RU
P(positive) = 0.94379. Result:  Positive
```

## <center>Real life testing</center>

```python
tweets_for_testing = [
    "Хуже этого фееричного ужастика может быть только игра главного актера",
    "Вы там все что, с ума сошли что-ли?",
    "Может хватит?",
    "мой друг поднялся на еверест и установил там флаг",
    "Можно выдыхать спокойно, новые Star Wars олдскульно отличные. Абрамс - крутой, как и всегда.",
    "меня оштрафовали по дороге домой",
    "я люблю ванильное мороженое",
    "меня оштрафовал контроллер"
]
for tweet in tweets_for_testing:
    test_tweet(tweet) 
    print("---------")
```
**Output:**  
```
Unknown token: фееричного
Original tweet: Хуже этого фееричного ужастика может быть только игра главного актера
P(positive) = 0.00223. Result:  Negative
---------
Original tweet: Вы там все что, с ума сошли что-ли?
P(positive) = 0.67757. Result:  Positive
---------
Original tweet: Может хватит?
P(positive) = 0.37129. Result:  Negative
---------
Unknown token: еверест
Original tweet: мой друг поднялся на еверест и установил там флаг
P(positive) = 0.87727. Result:  Positive
---------
Unknown token: выдыхать
Unknown token: олдскульно
Unknown token: Абрамс
Original tweet: Можно выдыхать спокойно, новые Star Wars олдскульно отличные. Абрамс - крутой, как и всегда.
P(positive) = 0.99995. Result:  Positive
---------
Unknown token: оштрафовали
Original tweet: меня оштрафовали по дороге домой
P(positive) = 0.89308. Result:  Positive
---------
Original tweet: я люблю ванильное мороженое
P(positive) = 0.82168. Result:  Positive
---------
Unknown token: оштрафовал
Unknown token: контроллер
Original tweet: меня оштрафовал контроллер
P(positive) = 0.04029. Result:  Negative
---------
```
Для того, чтобы понять, правильно ли прошло обучение, accuracy на тренировочных и тестовых выборках должны быть одинаковые.

Данная схема сети не самый лучший вариант для сентимент анализа твитов.
Можно добиться лучших результатов с помощью **word embedding**.

> :warning: Сложность заключается в том, что даже человек не может дать такую точность.
Например, восход на Еверест может быть позитивным твитом для одних, а для тех, у кого плохие ассоциации с Еверестом (умер друг), то твит перестает быть позитивным.

# Плотное векторное представление слов для определения тональности текста отзывов на фильмы из IMDb (Internet Movie Database)

Ссылка на [файл](https://colab.research.google.com/drive/19b8owNo7vALRTU8hbhAuJbFU-mHlwB7U) и [видео](https://www.youtube.com/watch?v=bpcr2uFBZog&list=PLtPJ9lKvJ4ojSWFe18CSKnhmmPHHIaTx-&index=5).

**Плотное векторное представление:**
Каждому токену текста (символу, слову или предложению) ставится в соответствии вектор, относительно небольшой длины и, в которой могут использоваться любые значения
(а не только 0 и 1).  

Будем рассматривать на примере задач определения тональности отзывов на фильмы из набора данных IMDb.

## Набор данных IMDb movie review

[Набор данных IMDb movie review](https://ai.stanford.edu/~amaas/data/sentiment/) создан для задач определения тональности текста. Набор включает отзывы на фильмы с сайта [IMDb](https://www.imdb.com). Отзывы только явно положительные (оценка >= 7) или отрицательные (оценка <= 4), нейтральные отзывы в набор данных не включались.

Размер набора данных 50 тыс. отзывов:
- Набор данных для обучения - 25 тыс. отзывов
- Набор данных для тестирования - 25 тыс. отзывов

Количество положительных и отрицательных отзывов одинаковое.

Разметка набора данных:
- 0 - отзыв отрицательный
- 1 - отзыв положительный

С точки зрения машинного обучения это задача бинарной классификации.

Набор данных описан в статье: [Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). Learning Word Vectors for Sentiment Analysis. The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011)](https://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf).

![IMDb](https://www.dropbox.com/s/grd17bkapocb92o/imdb_movie_reviews.png?dl=1)

**Разбор кода:**  

Подключаем необходимые модули, в том числе слой Embedding, который используется для создания плотных векторных представлений в keras.

```python
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten, Dropout
from tensorflow.keras import utils
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline 
```

## <center>Загружаем данные</center> 

Отзывы на фильмы из набора данных imdb, с максимальнным кол-вом слов 10000.

```python
max_words=10000
```
```python
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)
```

## <center>Просмотр данных</center> 

Посмотрим исходный формат данных, здесь отзывы разделены на отдельные слова и каждое слово представлено числом, которое соответствует частоте, с которым это слово встречается в наборе данных IMDb.

Рецензия

```python
x_train[3]
```
**Output:**  
```
[1,
 4,
 2,
 2,
 33,
 2804,
 4,
 2040,
 432,
 111,
 153,
....
....
....
 30,
 579,
 21,
 64,
 2574]
```
Правильные ответы, в бинарном формате
```python
y_train[3]
```
**Output:**  
```
1
```

## <center>Подготовка данных для обучения</center>

Наша нейронная сеть, может обрабатывать данные фиксированной длины, поэтому мы взяли длину макс отзыва была не более 200 слов (с помощью функции pad_sequences).

Если отзыв больше 200, обрезает, если меньше, то дополняется специальными символами заполнителями.
```pythonpython
maxlen = 200
```
```python
x_train = pad_sequences(x_train, maxlen=maxlen, padding='post')
x_test = pad_sequences(x_test, maxlen=maxlen, padding='post')
```
Наш отзыв в модифицированном варианте, где осталось 200 слов
```python
x_train[3]
```
**Output:**  
```
array([   4,  341,    7,   27,  846,   10,   10,   29,    9, 1906,    8,
         97,    6,  236,    2, 1311,    8,    4,    2,    7,   31,    7,
          2,   91,    2, 3987,   70,    4,  882,   30,  579,   42,    9,
         12,   32,   11,  537,   10,   10,   11,   14,   65,   44,  537,
         75,    2, 1775, 3353,    2, 1846,    4,    2,    7,  154,    5,
          4,  518,   53,    2,    2,    7, 3211,  882,   11,  399,   38,
         75,  257, 3807,   19,    2,   17,   29,  456,    4,   65,    7,
         27,  205,  113,   10,   10,    2,    4,    2,    2,    9,  242,
          4,   91, 1202,    2,    5, 2070,  307,   22,    7, 5168,  126,
         93,   40,    2,   13,  188, 1076, 3222,   19,    4,    2,    7,
       2348,  537,   23,   53,  537,   21,   82,   40,    2,   13,    2,
         14,  280,   13,  219,    4,    2,  431,  758,  859,    4,  953,
       1052,    2,    7, 5991,    5,   94,   40,   25,  238,   60,    2,
          4,    2,  804,    2,    7,    4, 9941,  132,    8,   67,    6,
         22,   15,    9,  283,    8, 5168,   14,   31,    9,  242,  955,
         48,   25,  279,    2,   23,   12, 1685,  195,   25,  238,   60,
        796,    2,    4,  671,    7, 2804,    5,    4,  559,  154,  888,
          7,  726,   50,   26,   49, 7008,   15,  566,   30,  579,   21,
         64, 2574], dtype=int32)
```
```python
y_train[3]
```
Правильный ответ остался без изменения, 1 - отзыв положительный
**Output:**  
```
1
```
## <center>Создание нейронной сети</center>

В которой будем использовать плотные векторные представления

```python
# Создадим модель

model = Sequential() # нейронная сеть последовательная
model.add(Embedding(max_words, 2, input_length=maxlen)) # первый слой, который добавляем в сеть - тип Embedding, слой плотных векторных представлений слов
# 1й параметр - кол-во слов (10000), 
# 2й - длина вектора (2), в котором будут представляться слова, для простоты и удобства в визуализации исп 2
# 3й - размер входных данных (maxlen=200 слов)

model.add(Dropout(0.25)) # слой, для снижения переобучения

# Embedding выдает массив, где 200 элементов размерности 2
model.add(Flatten()) # слой, преобразовывает массив в плоский вектор
model.add(Dense(1, activation='sigmoid')) # на выходе 1 нейрон, который соответствуеет задаче бинарной классификации

# функция активации 'sigmoid', который используется для бинарной классификации
```
Каким образом слой Embedding определяет векторы для представления слов?
Эти векторы определяются в процессе обучения нейронной сети. На первом этапе обучения, как и в других слоях нейронной сети, векторы в слое Embedding инициализируются случайными значениями, затем выполняется обычное обучение нейронной сети с помощью с помощью алгоритма обратного распространения ошибки.
```python
# Скомпилируем
model.compile(optimizer='adam', # оптимизатор адам
              loss='binary_crossentropy', # функция ошибки бинарная перекрестная энтропия, которая подходит для задачи бинарной классификации
              metrics=['accuracy']) # метрика качества обучения, доля правильных ответов
```
## <center>Обучаем нейронную сеть</center>

```python
history = model.fit(x_train, # набор данных для обучения
                    y_train, # данные правильных ответов
                    epochs=15,
                    batch_size=128, # размер мини выборки
                    validation_split=0.1) # для проверочной выборки используется 10% из набора x_train и y_train
```
**Output:**  
```
Train on 22500 samples, validate on 2500 samples
Epoch 1/15
22500/22500 [==============================] - 1s 32us/sample - loss: 0.6917 - acc: 0.5206 - val_loss: 0.6854 - val_acc: 0.6048
Epoch 2/15
22500/22500 [==============================] - 1s 28us/sample - loss: 0.6307 - acc: 0.7133 - val_loss: 0.5418 - val_acc: 0.7896
Epoch 3/15
22500/22500 [==============================] - 1s 26us/sample - loss: 0.4482 - acc: 0.8316 - val_loss: 0.3975 - val_acc: 0.8460
Epoch 4/15
22500/22500 [==============================] - 1s 26us/sample - loss: 0.3428 - acc: 0.8719 - val_loss: 0.3443 - val_acc: 0.8640
Epoch 5/15
22500/22500 [==============================] - 1s 28us/sample - loss: 0.2940 - acc: 0.8883 - val_loss: 0.3183 - val_acc: 0.8708
Epoch 6/15
22500/22500 [==============================] - 1s 27us/sample - loss: 0.2619 - acc: 0.9012 - val_loss: 0.3038 - val_acc: 0.8768
Epoch 7/15
22500/22500 [==============================] - 1s 26us/sample - loss: 0.2389 - acc: 0.9120 - val_loss: 0.2937 - val_acc: 0.8792
Epoch 8/15
22500/22500 [==============================] - 1s 27us/sample - loss: 0.2234 - acc: 0.9174 - val_loss: 0.2895 - val_acc: 0.8860
Epoch 9/15
22500/22500 [==============================] - 1s 26us/sample - loss: 0.2088 - acc: 0.9215 - val_loss: 0.2855 - val_acc: 0.8840
Epoch 10/15
22500/22500 [==============================] - 1s 27us/sample - loss: 0.1955 - acc: 0.9279 - val_loss: 0.2824 - val_acc: 0.8844
Epoch 11/15
22500/22500 [==============================] - 1s 27us/sample - loss: 0.1850 - acc: 0.9332 - val_loss: 0.2821 - val_acc: 0.8836
Epoch 12/15
22500/22500 [==============================] - 1s 26us/sample - loss: 0.1763 - acc: 0.9380 - val_loss: 0.2827 - val_acc: 0.8832
Epoch 13/15
22500/22500 [==============================] - 1s 27us/sample - loss: 0.1698 - acc: 0.9401 - val_loss: 0.2832 - val_acc: 0.8836
Epoch 14/15
22500/22500 [==============================] - 1s 27us/sample - loss: 0.1623 - acc: 0.9427 - val_loss: 0.2845 - val_acc: 0.8852
Epoch 15/15
22500/22500 [==============================] - 1s 27us/sample - loss: 0.1558 - acc: 0.9450 - val_loss: 0.2867 - val_acc: 0.8872
```

```python
plt.plot(history.history['accuracy'], 
         label='Доля верных ответов на обучающем наборе')
plt.plot(history.history['val_accuracy'], 
         label='Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.show()
```
![](https://i.imgur.com/BAr8Ukv.png)

На графике и в логах обучения видно, что мы близки к переобучению.
Доля правильных ответов на проверочном наборе перестало изменяться, и, если увеличить кол-во эпох, то вероятно, что качество распознавания начнет снижаться.

Можно сделать вывод, что плотные векторные представления слов, хорошо подходит для представления текстов в цифровом виде, чтобы его потом, можно было обрабатывать с помощью нейронной сети.

Если посмотреть на нашу нейронную сеть, то она очень простая. Содержит всего лишь 1 нейрон, который используется для классификации, и даже такая простая сеть, показывает достаточно хорошие результаты на наборе данных IMDb.

## <center>Проверяем работу сети на тестовом наборе данных</center>

Доля правильных ответов на тестовом наборе = 87%

```python
scores = model.evaluate(x_test, y_test, verbose=1)
```
**Output:**  
```
25000/25000 [==============================] - 1s 38us/sample - loss: 0.3030 - acc: 0.8733
```

## <center>Исследуем обученное плотное векторное представление слов</center>

**Получаем матрицу плотных векторных представлений слов**

Теперь посмотрим на векторное представление слов, которому научилась наша нейронная сеть.
В keras векторное представление это просто веса слоя embedding, мы можем получить отдельный слой модели, с помощью вызова layers.

layers[0] - самый первый слой, у которого тип embedding
и вызываем .get_weights()[0]

Результат записываем в embedding_matrix - **переменная плотных векторных представлений слов**
```python
embedding_matrix = model.layers[0].get_weights()[0]
```
```python
embedding_matrix[:5] # векторы из 2х элементов, каждый вектор соответствует какому-то слову
```
**Output::**  
```
array([[-0.01810081, -0.01080754],
       [ 0.08826496, -0.08724205],
       [ 0.0278171 , -0.02975242],
       [ 0.01640681, -0.03693285],
       [-0.01309797,  0.03457265]], dtype=float32)
```
Нулевая строка матрица соответствует слову с кодом 0, это специальный символ заполнитель.  
* 2й вектор - слову с кодом 1.  
* 3й - слово с кодом 2.  
* 4й - слово с кодом 3.  
* 5й - слово с кодом 4 и т.д.


**Загружаем словарь с номерами слов**

Попробуем получить вектор для каких-нибудь интересующих нас слов, для этого загрузим словарь, который использовался для кодирования текста в наборе данных IMDb.
```python
word_index_org = imdb.get_word_index()
```
Дополняем словарь служебными символами

Символ с кодом 0 используется для заполнения и т.д.
```python
word_index = dict()
for word,number in word_index_org.items():
    word_index[word] = number + 3
word_index["<Заполнитель>"] = 0
word_index["<Начало последовательности>"] = 1
word_index["<Неизвестное слово>"] = 2  
word_index["<Не используется>"] = 3
```
**Ищем векторы для слов**

В качестве примера получим вектор для слова good.
```python
word = 'good'
word_number = word_index[word]
print('Номер слова', word_number)
print('Вектор для слова', embedding_matrix[word_number])
```
**Output:**  
```
Номер слова 52
Вектор для слова [-0.16588722  0.15730934]
```
Мы можем сохранить обученные плотные векторные предствления в файл, для последующего использования.

## <center>Сохраняем обученные плотные векторные представления в файл</center>

**Составляем реверсивный словарь токенов (слов)**
```python
reverse_word_index = dict()
for key, value in word_index.items():
    reverse_word_index[value] = key
```
**Записываем плотные векторные представления в файл**
```python
filename = 'imdb_embeddings.csv' # задаем имя файла
```
```python
with open(filename, 'w', encoding = 'UTF-8') as f: 
    for word_num in range(max_words): # в цикле проходим по словарю
        word = reverse_word_index[word_num] # получаем слово
        vec = embedding_matrix[word_num] # и вектор
        f.write(word + ",") # записываем в файл слово
        f.write(','.join([str(x) for x in vec]) + "\n") # затем вектор
```
```python
import pandas as pd
df = pd.read_csv(filename) 
print(df[:20])
```
**Output:**
```
<Заполнитель>,-0.018100811,-0.010807536
<Начало последовательности>,0.08826496,-0.08724205
<Неизвестное слово>,0.027817102,-0.029752418
<Не используется>,0.016406808,-0.03693285
the,-0.013097967,0.034572646
and,-0.033376634,0.08939072
a,-0.0010535739,0.010263854
of,-0.0076864795,-0.031185996
to,0.005916276,0.010063415
is,-0.029501064,0.04398097
br,0.032622937,-0.012411324
in,-0.00853795,0.014011138
it,-0.030156253,0.084226154
i,-0.014873622,0.010922928
this,-0.010680796,-0.047810882
that,0.020756848,0.01780349
was,0.012138906,-0.05804113
as,-0.023956735,-0.0013690474
for,-0.0029819263,-0.038603384
with,-0.010972076,-0.0077637797
```
## <center>Визуализация плотных векторных представлений слов</center>

Для начала визуализируем все векторы из нашей матрицы embedding_matrix.
Здесь показано 10 000 векторов, в них сложно что-то разобрать.
```python
plt.scatter(embedding_matrix[:,0], embedding_matrix[:,1])
```
![](https://i.imgur.com/w4PyiYx.png)

Выбираем коды слов, по которым можно определить тональность отзыва (эмоциональную окраску текста)

*Пример:*
```python
review = ['brilliant', 'fantastic', 'amazing', 'good',
          'bad', 'awful','crap', 'terrible', 'trash']
enc_review = []
for word in review:
    enc_review.append(word_index[word])
enc_review
```
**Output:**
```
[530, 777, 480, 52, 78, 373, 595, 394, 1157]
```
Получаем векторное представление интересующих нас слов
```python
review_vectors = embedding_matrix[enc_review]
review_vectors
```
**Output:**
```
array([[-0.48637104,  0.47641388],
       [-0.52597755,  0.5208416 ],
       [-0.5749784 ,  0.54873264],
       [-0.16588722,  0.15730934],
       [ 0.49321252, -0.41877759],
       [ 1.0940588 , -1.0372338 ],
       [ 0.57186013, -0.42061222],
       [ 0.74706984, -0.6985454 ],
       [ 0.24538721, -0.17148855]], dtype=float32)
```
Визуализация обученного плотного векторного представления слов, по которым можно определить эмоциональную окраску текста
```python
plt.scatter(review_vectors[:,0], review_vectors[:,1])
for i, txt in enumerate(review):
    plt.annotate(txt, (review_vectors[i,0], review_vectors[i,1]))
```
![](https://i.imgur.com/k8zidwG.png)

Мы можем увидеть, что обучение плотных векторных представлений прошло достаточно успешно, мы видим, что в верхнем левом углу находятся прилагательные отвечающие за положительную эмоциональную окраску.
В противоположном конце прилагательные, которые отвечают за отрицательную эмоциональную окраску.
Чуть ближе к центру bad, crap, которые тоже имеют отрицательную окраску, но не такую сильную, как terrible и awful.
Еще ближе находится trash. И совсем недалеко от центра координат находится слово good. Потому что часто в отзывах пишут "not good".

# Дополнительные ресурсы

## Как делать сентимент-анализ рекуррентной LSTM сетью 

[Видео](https://www.youtube.com/watch?v=Y96vctxr5GU) и [github](https://github.com/selfedu-rus/neural-network/blob/master/lesson%2024.%20LSTM%20sentiment%20analysis.py).

## Представление текста вектором One Hot Encoding

[Видео](https://youtu.be/MEalcn_iD00) и [файл](https://colab.research.google.com/drive/1pw07gkY_axF5J5qtGbUwPXS36G5tW-bO).

## Анализ тональности текста рекуррентной нейросетью

[Видео](https://youtu.be/I9RMwvyzGpM) и [файл](https://colab.research.google.com/drive/19RAxAnIV0fDXAcE1T23TPkzi4ZPFQXdB).