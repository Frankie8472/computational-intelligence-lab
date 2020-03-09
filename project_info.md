## Project for Computational Intelligence Lab

### Team
**Name**: Backpropagaters

**Members**: Sarah Kamp, Silvia La, Josua Cantieni, Franz Knobel

### File descriptions

* **data_train.csv** - the training set. Each entry consists of an ID of the form r3_c6 (meaning row 3 column 6) and the value between 1-5 stars, given by the user for this user/movie combination
* **sampleSubmission.csv** - a sample submission file in the correct format. You have to predict the star ratings of the matrix entries specified in this file. In this dummy submission example, a rating of 3 is predicted for all positions in question.

### Project Option 1: Collaborative Filtering
A recommender system is concerned with presenting items 
(e.g. books on Amazon, movies at Movielens or music at lastFM) 
that are likely to interest the user. 
In collaborative filtering, we base our recommendations on 
the (known) preference of the user towards other items, 
and also take into account the preferences of other users.

#### Resources

All the necessary resources (including training data) are 
available at https://inclass.kaggle.com/c/cil-collab-filtering-2020

To participate, 
follow the link http://da.inf.ethz.ch/teaching/2020/CIL/files/project2.txt.

#### Training Data

For this problem, we have acquired ratings of 10000 users 
for 1000 different items. All ratings are integer values 
between 1 and 5 stars.

#### Evaluation Metrics

Your collaborative filtering algorithm is evaluated according 
to the following weighted criteria:
* prediction error, measured by root-mean-squared error (**RMSE**)

### Report Grading Guidelines

Your paper will be graded by two independent reviewers according to 
the following three criteria:

#### 1) Quality of paper (30%)
* 6.0: Good enough for submission to an international conference.
* 5.5: Background, method, and experiment are clear. May have minor issues in one or two sections. Language is good. Scores and baselines are well documented.
* 5.0: Explanation of work is clear, and the reader is able to identify the novelty of the work. Minor issues in one or two sections. Minor problems with language. Has all the recommended sections in the howto-paper
* 4.5: Able to identify contribution. Major problems in presentation of results and or ideas and or reproducibility/baselines.
* 4.0: Hard to identify contribution, but still there. One or two good sections should get students a pass.
* 3.5: Unable to see novelty. No comparison with any baselines.

#### 2) Creativity of solution (20%)
* 6.0: Elegant proposal, either making a useful assumption, studying a particular class of data, or using a novel mathematical fact.
* 5.5: A non-obvious combination of ideas presented in the course or published in a paper (Depending on the difficulty of that idea).
* 5.0: A novel idea or combination not explicitly presented in the course.
* 4.5: An idea mentioned in a published paper with small extensions / changes, but not so trivial to implement.
* 4.0: A trivial idea taken from a published paper.

#### 3) Quality of implementation (20%)
* 6.0: Idea is executed well. The experiments done make sense in order to answer the proposed research questions. There are no obvious experiments not done that could greatly increase clarity. The submitted code and other supplementary material is understandable, commented, complete, clean and there is a README file that explains it and describes how to reproduce your results.

#### Subtractions from this grade will be made if:
* ...the submitted code is unclear, does not run or experiments cannot be reproduced or there is no description of it
* ...experiments done are useless to gain understanding or of unclear nature or obviously useful experiments have been left undone
* ...comparison to baselines are not done
