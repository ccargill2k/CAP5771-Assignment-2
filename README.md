# CAP5771 - Assignment 2
## Author: Chris Cargill
This README.md file will explain how to properly run the code files for this assignment as well as the purpose of each script.
## File Rundown
`problem1.py`: is the full code file for Problem 1 and runs the SVM multi-label classification for the Scene dataset

`problem2.py`: is the full code file for Problem 2 and runs the K-Means clustering implementation for the Seeds dataset

`problem3.py`: is the full code file for Problem 3 and runs the Random Forest and four balancing techniques for the German Credit Card dataset

`run1.py`: is the script for Problem 1

`run2.py`: is the script for Problem 2

`run3.py`: is the script for Problem 3

`CAP5771-Assignment-2-Report.docx`: is the 3-page report summarizing my findings

`X_test.txt`, `X_train.txt`, `y_test.txt`, and `y_train.txt`: are the Scene datasets for Problem 1

`seeds.txt`: is the dataset for Problem 2

`German_Credit_Data.txt`: is the dataset for Problem 3
## Python Libraries
Make sure the following libraries are downloaded already:

(1) numpy

(2) pandas

(3) scikit-learn

(4) scipy

(5) imblearn

To install these Python packages:
```bash
pip install numpy pandas scikit-learn scipy imblearn
```
## Run the Code
Run the following code in the terminal for Questions #1-3:
```bash
py run1.py
py run2.py
py run3.py
```
As another option depending on the system:
```bash
python run1.py
python run2.py
python run3.py
```
The output in the terminal should produce the results for all three questions.