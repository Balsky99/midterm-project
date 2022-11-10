# Mobile Price Classification

Bob has started his own mobile company. He wants to give tough fight to big companies like Apple,Samsung etc.

He does not know how to estimate price of mobiles his company creates. In this competitive mobile phone market you cannot simply assume things. To solve this problem he collects sales data of mobile phones of various companies.

Bob wants to find out some relation between features of a mobile phone(eg:- RAM,Internal Memory etc) and its selling price. But he is not so good at Machine Learning. So he needs your help to solve this problem.

In this problem you do not have to predict actual price but a price range indicating how high the price is

# How to Run

First activate the env inside the clone folder in your machine.

Make sure your machine already installed pipenv.

`shell pipenv`

Then, create the docker inside your machine.

`docker build -t <tag_name> .`

After that, run the following to start the churn web service.

`docker run –it –-rm -p 9696:9696 <tag_name>`

Note: If the docker daemon is not running, run the following code.

`sudo dockerd`

# Testing the Web Service

Make sure the web service is on, then simply run the following code.

`python predict-test.py`
