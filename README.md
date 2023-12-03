# Wine-Quality-Prediction
# Spark Decision Tree Classification

This Java-based Spark application shows how to train and test a Decision Tree classifier with Spark MLlib. The code is intended to import a dataset, train a Decision Tree model, and test it on a validation dataset.


## Prerequisites

- **Apache Spark:** Ascertain that Apache Spark is installed and setup.

- **Java Development Kit (JDK):** Check that Java is installed on your machine.

## Project Structure

- **src/:**The Java source code files are included.
  - `DecisionTreeSpark.java`: Code for training and assessing a Decision Tree model on a test dataset.
  - `TestDecisionTreeModel.java`:Loading and testing a pre-trained Decision Tree model on a validation dataset.

- **data/:**This is a placeholder for your dataset files.
  - `TrainingDataset.csv`: Training dataset in CSV format.
  - `ValidationDataset.csv`: Validation dataset in CSV format.

- **model/:** Tis is a placeholder for storing the trained Decision Tree model.

## Running the Code

1. Make sure Spark is properly installed and that the'spark-submit' script is in your system's PATH.

2. Compile the Java code:

    ```bash
    javac -cp "path/to/spark/jars/*" src/*.java
    ```

3. Run the training and testing code:

    ```bash
    spark-submit --class DecisionTreeSpark --master local[*] --driver-class-path "path/to/spark/jars/*" src/DecisionTreeSpark.java
    ```

4. Run the pre-trained model testing code:

    ```bash
    spark-submit --class TestDecisionTreeModel --master local[*] --driver-class-path "path/to/spark/jars/*" src/TestDecisionTreeModel.java
    ```



