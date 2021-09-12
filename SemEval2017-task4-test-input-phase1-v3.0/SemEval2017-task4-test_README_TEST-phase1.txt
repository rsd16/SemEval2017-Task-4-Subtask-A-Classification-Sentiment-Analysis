******************************************************
* SemEval-2017 Task 4: Sentiment Analysis on Twitter *
*                                                    *
*               TEST datasets (input)                *
*                                                    *
* http://alt.qcri.org/semeval2017/task4/             *
* semevaltweet@googlegroups.com                      *
*                                                    *
******************************************************

Version 3.0: January 16, 2017


Task organizers:

Sara Rosenthal, IBM Research
Noura Farra, Columbia University
Preslav Nakov, Qatar Computing Research Institute, HBKU


NOTE

Please note that by downloading the Twitter data you agree to abide
by the Twitter terms of service (https://twitter.com/tos).


HISTORY

v. 3.0: removed 95 duplicates for subtask A, English

v. 2.0: fixed 27 lines for English


CONTENTS OF THIS DISTRIBUTION

- SemEval2017-task4-test_README_TEST-phase1.txt -- this file

- SemEval2016-task4-test.subtask-A.arabic.txt -- test input for subtask A, Arabic
- SemEval2016-task4-test.subtask-A.english.txt -- test input for subtask A, English

- SemEval2016-task4-test.subtask-CE.arabic.txt -- test input for subtasks C and E, Arabic
- SemEval2016-task4-test.subtask-CE.english.txt -- test input for subtasks C and E, English

NOTE: The test input for subtasks B and D will be released only after the deadline for A, C, E has passed.


IMPORTANT

In order to use these test datasets, the participants need (1), 
and most likely also (2) and (3):

1. the official scorers and format checkers
2. the TRAIN, DEV and DEVTEST datasets

You can find links to them here:

	http://alt.qcri.org/semeval2017/task4/index.php?id=data-and-tools

NOTES:

1. For English, we provide a default split of the data from previous years into training, development and development-time testing datasets; participants are free to use this data in any way they find useful when training and tuning their systems, e.g., use a different split, perform cross-validation, train on all datasets, etc.

2. For English, unlike in previous years, for SemEval-2017 Task 4, there will be no progress testing, and thus all the provided data can be used for training and development.


INPUT DATA FORMAT


-----------------------SUBTASK A-----------------------------------------

--Test Data--
The format for the test input file is as follows:

	id<TAB>UNKNOWN<TAB>tweet_text

for example:

	1       UNKNOWN  amoure wins oscar
	2       UNKNOWN  who's a master brogramer now?

--System Output--
We expect the following format for the prediction file:

	id<TAB>predicted_sentiment_4_tweet

where predicted_sentiment_4_tweet can be 'positive', 'neutral' or 'negative'.

For example:
1        positive
2        neutral



-----------------------SUBTASK C-----------------------------------------
--Test Data--
The format for the test input file is as follows:

	id<TAB>topic<TAB>UNKNOWN<TAB>tweet_text

for example:

	1      aaron rodgers       UNKNOWN       I just cut a 25 second audio clip of Aaron Rodgers talking about Jordy Nelson's grandma's pies. Happy Thursday.
	2      aaron rodgers       UNKNOWN       Tough loss for the Dolphins last Sunday in Miami against Aaron Rodgers &amp; the Green Bay Packers: 27-24.

--System Output--
We expect the following format for the prediction file:

	id<TAB>topic<TAB>predicted_sentiment_4_topic

where predicted_sentiment_4_topic can be -2, -1, 0, 1, or 2.

For example:
	1      aaron rodgers       1
	2      aaron rodgers       0


-----------------------SUBTASK E-----------------------------------------
--Test Data--
The format is the same as for subtask B.

--System Output--
We expect the following format for the prediction file:

	topic<TAB>label-2<TAB>label-1<TAB>label0<TAB>label1<TAB>label2

where label-2 to label2 are floating point numbers between 0.0 and 1.0, and the five numbers sum to 1.0. label-2 corresponds to the fraction of tweets labeled as -2 in the data and so on.

For example:
	aaron rodgers       0.025 0.325   0.5    0.1 0.05
	peter pan           0.05  0.40    0.5    0.05 0.0

-------------------------------------------------------------------------


EVALUATION

There are different evaluation measures for the different subtasks. A detailed description can be found here:

	http://alt.qcri.org/semeval2016/task4/data/uploads/eval.pdf


TEAMS

We discourage multiple teams with overlapping team members.


SUBMISSION NOTES

1. Submission is done using CodaLab; see here:

	http://alt.qcri.org/semeval2017/task4/index.php?id=data-and-tools

2. Participants are free to participate for a single subtask or for any combination of subtasks.

3. We allow a single run per subtask--language pair combination.

4.  Participants are free to use any data from previous years (see above): we will not distinguish between closed (that only use the provided data) and open (that also use additional data) runs. However, they will need to describe the resources and tools they have used to train their systems in the Web form they have recieved by email.


LICENSE

The accompanying dataset is released under a Creative Commons Attribution 3.0 Unported License
(http://creativecommons.org/licenses/by/3.0/).


CITATION

You can cite the following paper when referring to the dataset:

@InProceedings{SemEval:2017:task4,
  author    = {Sara Rosenthal and Noura Farra and Preslav Nakov},
  title     = {{SemEval}-2017 Task 4: Sentiment Analysis in {T}witter},
  booktitle = {Proceedings of the 11th International Workshop on Semantic Evaluation},
  series    = {SemEval '17},
  month     = {August},
  year      = {2017},
  address   = {Vancouver, Canada},
  publisher = {Association for Computational Linguistics},
}


USEFUL LINKS:

Google group: semevaltweet@googlegroups.com
SemEval-2017 Task 4 website: http://alt.qcri.org/semeval2017/task4/
SemEval-2017 website: http://alt.qcri.org/semeval2017/


REFERENCES:

Preslav Nakov, Sara Rosenthal, Svetlana Kiritchenko, Saif M. Mohammad, Zornitsa Kozareva, Alan Ritter, Veselin Stoyanov, Xiaodan Zhu. Developing a successful SemEval task in sentiment analysis of Twitter and other social media texts. Language Resources and Evaluation 50(1): 35-65 (2016).

Preslav Nakov, Alan Ritter, Sara Rosenthal, Fabrizio Sebastiani, and Veselin Stoyanov. SemEval-2016 Task 4: Sentiment Analysis in Twitter. In Proceedings of the 10th International Workshop on Semantic Evaluation (SemEval'2016), June 16-17, 2016, San Diego, California, USA.

Sara Rosenthal, Preslav Nakov, Svetlana Kiritchenko, Saif M Mohammad, Alan Ritter, and Veselin Stoyanov. SemEval-2015 Task 10: Sentiment Analysis in Twitter. In Proceedings of the 9th International Workshop on Semantic Evaluation (SemEval'2015), pp.451-463, June 4-5, 2016, Denver, Colorado, USA.

Sara Rosenthal, Preslav Nakov, Alan Ritter, Veselin Stoyanov. SemEval-2014 Task 9: Sentiment Analysis in Twitter. In Proceedings of International Workshop on Semantic Evaluation (SemEval’14), pp.73-80, August 23-24, 2014, Dublin, Ireland.

Preslav Nakov, Sara Rosenthal, Zornitsa Kozareva, Veselin Stoyanov, Alan Ritter, Theresa Wilson. SemEval-2013 Task 2: Sentiment Analysis in Twitter. In Proceedings of the Second Joint Conference on Lexical and Computational Semantics (*SEM'13), Volume 2: Proceedings of the Seventh International Workshop on Semantic Evaluation (SemEval'2013). pp. 312-320, June 17-19, 2013, Atlanta, Georgia, USA.
