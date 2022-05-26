Setting up processes on Apache Airflow 
===============================

Настройка процессов на Apache Airflow по передачи данных из одной базы данных в другую.

## Подключение Python к базе данных

### Подключение к базе данных **DWH** (PostgreSQL) на _**Python**_ и _**Airflow**_

Установка библиотеки Psycopg2 с помощью pip, для использования Postgres в **Python**:
```
pip install psycopg2
```
Подключение (с открытым паролем) на _**Python**_ или _**Airflow**_: 
```python
import psycopg2
try:
    # Подключение к существующей базе данных
    postgres_conn = psycopg2.connect(dbname='*****', 
                                     user='*****', 
                                     host='XXX.XXX.XXX.XXX', 
                                     password='******',
                                     connect_timeout=3)
    # Курсор для выполнения операций с базой данных
    postgres_cur = postgres_conn.cursor() 
except:
    print('Не удалось подключиться к postgres')
```
Аргументы для подключения:
* `dbname` - имя базы данных
* `user` - имя пользователя
* `host` - имя сервера или IP-адрес, на котором работает база данных. Если она запущена локально, то нужно использовать localhost или 127.0.0.0.
* `password` - пароль
* `connect_timeout` - устанавливает максимальное время ожидания соединения в секундах. Значение по умолчанию равно 0, что означает, что время ожидания равно бесконечности (не рекомендуется устанавливать значение на ожидание меньше 2 секунд).
* `cursor = connection.cursor()` (или, `postgres_cur = postgres_conn.cursor()` ) — курсор для взаимодействия с базой данных. Он поможет выполнять SQL-команды из Python.


> :information_source: Узнать подробнее о Postgres на Python [<span style="color:orange"> → postgresql.pdf ← </span>](https://inp.nsk.su/~baldin/PostgreSQL/postgresql.pdf) и 
[<span style="color:violet"> → psycopg.org ← </span>](https://www.psycopg.org/docs/).

### Подключение к базе данных **CRM** (MariaDB) на _**Python**_ и _**Airflow**_

> :information_source: MariaDB — ответвление от системы управления базами данных MySQL.  
Поэтому для MariaDB подходит библиотека MySQL.

Установка библиотеки MySQL с помощью pip, для использования MariaDB в **Python**:
```
pip install mysql-connector-python
```
Подключение (с открытым паролем) на _**Python**_ или _**Airflow**_:

```python
import mysql.connector
try:
    # Подключение к существующей базе данных
    damucrm_conn = mysql.connector.connect(user='*****',
                                           password='******!',
                                           host='XXX.XXX.XXX.XXXX',
                                           database='damucrm')
    # Курсор для выполнения операций с базой данных
    damucrm_cur = damucrm_conn.cursor() #.cursor(buffered=True)
except:
    print('Не удалось подключиться к damucrm')
```
> :information_source:
При возникновении некоторых ошибок, можно указать `.cursor(buffered=True)`, подробнее [<span style="color:orange"> → mariadb-corporation ←  </span>](https://mariadb-corporation.github.io/mariadb-connector-python/connection.html).

> :information_source: Узнать подробнее о MySQL (MariaDB) на Python [<span style="color:orange"> → dev.mysql.com ←  </span>](https://dev.mysql.com/doc/connector-python/en/connector-python-example-connecting.html).

## Apache Airflow

_Airflow_ — это библиотека (или набор библиотек) для разработки, планирования и мониторинга рабочих процессов. Используется код на языке Python.

> :bulb: Подробности по [Apache Airflow](https://github.com/apache/airflow).

### DAG
Основная сущность Airflow — это Directed Acyclic Graph (DAG).

_DAG_ — это некоторое смысловое объединение задач, которые нужно выполнить в определенной последовательности по определенному расписанию. 

![](https://databand.ai/wp-content/uploads/2021/09/airflow-UI-2048x1232.png)

Подключение DAG модуля и создание:
```python
from airflow import DAG
dag = DAG(
    dag_id=DAG_ID,
    description='',
    schedule_interval='* * * *',
    default_args = default_args
    tags=['']
    )
```

> :information_source: Сам DAG не выполняет никакой работы, просто запускает dummy задачи в определенное время, пример времени [<span style="color:orange">  
→ crontab.guru ←  </span>](https://crontab.guru/).


### Операторы

_Оператор_ — это сущность, на основании которой создаются экземпляры заданий, где описывается, что будет происходить во время исполнения экземпляра задания. Релизы Airflow с GitHub уже содержат набор операторов, готовых к использованию:

Подключение операторов:
```python
from airflow.operators.python_operator import PythonOperator
from airflow.operators.postgres_operator import PostgresOperator
```
* `PythonOperator` — оператор для вызова Python-кода.
* `PostgresOperator` — оператор для выполнения PostgreSQL-кода.

> :information_source: Документация об операторах Airflow [<span style="color:orange"> → airflow.apache.org ←  </span>](https://airflow.apache.org/docs/apache-airflow/1.10.12/_modules/airflow/operators.html).

### Исключения

```python
from airflow.exceptions import AirflowFailException
```

> :information_source: Документация об исключениях Airflow [<span style="color:orange"> → airflow.apache.org ←  </span>](https://airflow.apache.org/docs/apache-airflow/1.10.12/_modules/airflow/exceptions.html).

### Подключение к базам данных

#### DWH (PostgreSQL)

Подключение через PostgresHook (без открытых указаний данных бд) на _**Airflow**_: 

```python
import psycopg2
from psycopg2.extras import execute_values
from airflow.hooks.postgres_hook import PostgresHook
postgres_conn = PostgresHook(postgres_conn_id='*****').get_conn()  
postgres_cur = postgres_conn.cursor()
```
> :warning: Это подключение должно быть установлено в настройках на Airflow.

> :information_source: Подключение Airflow к бд с открытым паролем, указано в документации выше.

#### CRM (MariaDB)

Подключение через MySqlHook (без открытых указаний данных бд) на _**Airflow**_: 

```python
import mysql.connector
from airflow.hooks.mysql_hook import MySqlHook
damucrm_conn = MySqlHook(mysql_conn_id = '*****').get_conn()
damucrm_cur = damucrm_conn.cursor()
```
> :warning: Это подключение должно быть установлено в настройках на Airflow.

### Пример процесса

```python
import psycopg2
import mysql.connector
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.postgres_operator import PostgresOperator
from airflow.hooks.postgres_hook import PostgresHook
from airflow.hooks.mysql_hook import MySqlHook
from psycopg2.extras import execute_values
from airflow.exceptions import AirflowFailException

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2022,2,10),
    'retries': 0,
}

dag = DAG(
    dag_id='dag_name_exmple',
    description='desription_example',
    schedule_interval='*/15 9-23 * * *', #every 15 min from 9 am to 23 pm
    default_args = default_args,
    tags=['Example']
    )

def get_f_mb_example():
    damucrm_conn = mysql.connector.connect(user='*****',
                                           password='******',
                                           host='XXX.XXX.XXX.XXX',
                                           database='*****')   
    # or damucrm_conn = MySqlHook(mysql_conn_id = '*****').get_conn()
    # damucrm_cur = damucrm_conn.cursor()
    postgres_conn = PostgresHook(postgres_conn_id='*****').get_conn()
    damucrm_cur = damucrm_conn.cursor()
    postgres_cur = postgres_conn.cursor()
    #Insert into "scheme_cdwh.example" table from "scheme_crm.example"
    postgres_cur.execute('''truncate scheme_dwh.example;''')      
    damucrm_cur.execute('''select smth1 ,smth2 ,smth3
                            from scheme_crm.example;''')  
    while True:
        records = damucrm_cur.fetchmany(size=500)
        if not records:
            break
        postgres_cur.executemany("""INSERT INTO scheme_dwh.example
                                    (smth1 ,smth2 ,smth3) 
                                    VALUES (%s,%s,%s)""", records)
        postgres_conn.commit()        
    damucrm_cur.close()
    damucrm_conn.close()
    postgres_cur.close()
    postgres_conn.close()

def get_crm_example():
    damucrm_conn = mysql.connector.connect(user='',
                                           password='',
                                           host='',
                                           database='')   
    postgres_conn = PostgresHook(postgres_conn_id='').get_conn()
    damucrm_cur = damucrm_conn.cursor()
    postgres_cur = postgres_conn.cursor()
    #Insert into "scheme_crm.example" table from "scheme_dwh.example"
    damucrm_cur.execute('''truncate scheme_crm.example;''')      
    postgres_cur.execute('''select smth1 ,smth2 ,smth3
                            from scheme_dwh.example;''')  
    while True:
        records = postgres_cur.fetchmany(size=500)
        if not records:
            break
        damucrm_cur.executemany("""INSERT INTO scheme_crm.example
                                    (smth1 ,smth2 ,smth3) 
                                    VALUES (%s,%s,%s)""", records)
        damucrm_cur.commit()        
    damucrm_cur.close()
    damucrm_conn.close()
    postgres_cur.close()
    postgres_conn.close()

t1 = PythonOperator(
    task_id = 'get_f_mb_example',
    python_callable = get_f_mb_example,
    dag = dag,
    email_on_failure = True,
    email = 'example@gmail.com')

t1 = PythonOperator(
    task_id = 'get_crm_example',
    python_callable = get_crm_example,
    dag = dag,
    email_on_failure = True,
    email = 'example@gmail.com')

t1 >>t2
```




























