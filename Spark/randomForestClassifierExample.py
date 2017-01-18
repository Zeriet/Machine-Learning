#standalone.py

from __future__ import  print_function

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession


if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("RandomForestClassifierExample")\
        .getOrCreate()
    data = spark.read.format("libsvm").load("sample_libsvm_data.txt")

    print ("_______")
    #Loading and parsing the data file, and conervting it to a dataframe
    print ("======")


