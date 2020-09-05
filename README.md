# SemEval2021-Reading-Comprehension-of-Abstract-Meaning
 This is the repository for SemEval 2021 Task 4: Reading Comprehension of Abstract Meaning. It includes code for baseline models and data.

## Data
**Data Format**


**Sample**


**Code**

Data can be treated as python dictionary objects. A simple script to read **ReCAM** data is as follows:

```
>def read_recam(path):
>>	with open(path, mode='r') as f:
>>>		reader = jsonlines.Reader(f)
        for instance in reader:
        print(instance)

```
