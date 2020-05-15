# HeavyWater Machine Learning Problem

### Problem Statement

This dataset represents the output of the OCR stage of our data pipeline. Each word in the source is mapped to one unique value in the output. If the word appears in multiple documents then that value will appear multiple times. The word order for the dataset comes directly from our OCR layer, so it should be _roughly_ in order.

Here is a sample line:

```
CANCELLATION NOTICE,641356219cbc f95d0bea231b ... [lots more words] ... 52102c70348d b32153b8b30c
```

The first field is the document label. Everything after the comma is a space delimited set of word values.

The dataset is included as part of this repo.

### Goal

Train a document classification model. Deploy your model to a public cloud platform (AWS/Google/Azure/Heroku) as a webservice, send us an email with the URL to you github repo, the URL of your publicly deployed service so we can submit test cases 




### Solution

Implemented Random Forest Algorithmto to train a classifier and get the word or set of words from user to predict the label of the document with the efficiency of 86.7%


### AWS to deploy Webservice

### Step 1 - Create Amazon S3 Bucket

i) Click Create Bucket
Bucket Name: type a unique name for your bucket, such as awscodepipeline-text-classification-example.  

ii) Region: In the drop-down, select the region where you will create your pipeline,Click Create.

iii) Enable versioning. When versioning is enabled, Amazon S3 saves every version of every object in the bucket.

iv) Upload .zip file containing the code and make it public to fetch Public URL.

### Step 2 - Create a Deployment Environment using ElasticBeanStalk

i) Create an application

ii) Choose platform as Python3.6

iii) Choose Code deploy from S3 and provide Public URL.

iv) Click Create Application.


### Step 3 - Create Pipeline

i) Click Create pipeline.

ii) Enter the name for your pipeline. 

iii) Create Service Role to be generated automatically.

iv) Click Next step.

v) Type the name of the Amazon S3 bucket you created followed by the sample file you copied to that bucket.

vi) Choose No Build, Click Next step

vii) Deployment provider: Click AWS Elastic Beanstalk. Provide an Application Name created in EBS. Select an Environment name under EBS application.

viii) Activate pipeline to deploy the code.




