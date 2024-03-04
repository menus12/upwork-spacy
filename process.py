#!/usr/bin/env python3

# common
import os
import json
import re
import argparse
import datetime

#spacy
import spacy
from spacy.pipeline import EntityRuler
from spacy.lang.en import English
from spacy.tokens import Doc

#gensim
import gensim
from gensim import corpora

#Visualization
from spacy import displacy
import pyLDAvis.gensim_models
from wordcloud import WordCloud
import plotly.express as px
import matplotlib.pyplot as plt

#Data loading/ Data manipulation
import pandas as pd
import numpy as np
import jsonlines

#nltk
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download(['stopwords','wordnet'])

#warning
import warnings 
warnings.filterwarnings('ignore')

# Adding command line parameters
parser = argparse.ArgumentParser(description='Upwork data processor')    
parser.add_argument('--file', type=str, help='Projects dump from database')
parser.add_argument('--update', type=bool, help='Update metadata (default is False)')
args = parser.parse_args()

if args.update == None:
    args.update = False

if args.file == None:
    print (parser.print_help())
    exit(1)

nlp = spacy.load("en_core_web_lg")
skill_pattern_path = "kaggle/jz_skill_patterns.jsonl"

ruler = nlp.add_pipe("entity_ruler")
ruler.from_disk(skill_pattern_path)

def get_skills(text):
    doc = nlp(text)
    myset = []
    subset = []
    for ent in doc.ents:
        if ent.label_ == "SKILL":
            subset.append(ent.text)
    myset.append(subset)
    return subset

def unique_skills(x):
    return list(set(x))

def clear_text(text):
    review = re.sub(
        '(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?"',
        " ",
        text,
    )
    review = review.lower()
    review = review.split()
    lm = WordNetLemmatizer()
    review = [
        lm.lemmatize(word)
        for word in review
        if not word in set(stopwords.words("english"))
    ]
    review = " ".join(review)
    return review

start_time = datetime.datetime.now()

print ('--- Loading data')

f = open(args.file)                 # Opening JSON file
source_file = json.loads(f.read())  # returns JSON object as  a dictionary
f.close()                           # Closing file

f = open("dumps/example_cv.json")   # Opening JSON file
cv = json.loads(f.read())           # returns JSON object as  a dictionary
f.close()                           # Closing file


print ('File is ' + args.file)
print ('---')

total_skills = []

for project in source_file:
    # separate rate
    rate_re = '\$\d+-\$\d+'
    if "Rate not defined" in project['description']:
        project['rate_min']  = None
        project['rate_max']  = None
        re.sub('Rate not defined', " ", project['description'])
    rate = re.findall(rate_re, project['description'])
    if len(rate) > 0:
        project['rate_min']  = re.sub('\$','', rate[0].split('-')[0])
        project['rate_max']  = re.sub('\$','', rate[0].split('-')[1])
        project['description'] = re.sub(rate_re, " ", project['description'])

    # separate skills to technologies
    skills_re = 'Skills: ([a-zA-Z\d -\/;]+)'
    if "Skills:" in project['description']:        
        skills = re.findall(skills_re, project['description'])  
        project['technologies'] = skills.pop()
    else: project['technologies'] = "None"
    project['description'] = re.sub(skills_re, " ", project['description'])

    # separate category
    cat_re = 'Category: ([a-zA-Z\d -\/;]+)'
    if "Category:" in project['description']:        
        category = re.findall(cat_re, project['description'])  
        project['category'] = category.pop()
    else: project['category'] = "None"
    project['description'] = re.sub(cat_re, " ", project['description'])

    # cleaning description
    #project['clean_description'] = clear_text(project['description'])
    project['clean_description'] = project['description']
    
    # checking for other skills
    project['skills'] = get_skills(project["clean_description"].lower() + ' ' + project["technologies"].lower())
    project['skills'] = unique_skills(project['skills'])
    for skill in project['skills']:
        total_skills.append(skill)

# Plotting Project Categories
fig = px.histogram(
    source_file, x="category", title="Distribution of Project Categories"
).update_xaxes(categoryorder="total descending")
fig.show()

# Plotting Project Countries
fig = px.histogram(
    source_file, x="country", title="Distribution of Countries"
).update_xaxes(categoryorder="total descending")
fig.show()

# Plotting Project skills
fig = px.histogram(
    x=total_skills,
    labels={"x": "Skills"},
    title="Distribution of Skills",
).update_xaxes(categoryorder="total descending")
fig.show()

df = pd.read_csv("kaggle/Resume/Resume.csv")
df = df.reindex(np.random.permutation(df.index))
data = df.copy().iloc[0:20,]

print ('--- Cleaning data')
clean = []
for i in range(data.shape[0]):
    review = re.sub(
        '(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?"',
        " ",
        data["Resume_str"].iloc[i],
    )
    review = review.lower()
    review = review.split()
    lm = WordNetLemmatizer()
    review = [
        lm.lemmatize(word)
        for word in review
        if not word in set(stopwords.words("english"))
    ]
    review = " ".join(review)
    clean.append(review)

data["Clean_Resume"] = clean

print ('--- Extracting skills')
data["skills"] = data["Clean_Resume"].str.lower().apply(get_skills)
data["skills"] = data["skills"].apply(unique_skills)

# fig = px.histogram(
#     data, x="Category", title="Distribution of Jobs Categories"
# ).update_xaxes(categoryorder="total descending")
# fig.show()

Job_Cat = data["Category"].unique()
Job_Cat = np.append(Job_Cat, "ALL")

Job_Category = "ALL"
Total_skills = []
if Job_Category != "ALL":
    fltr = data[data["Category"] == Job_Category]["skills"]
    for x in fltr:
        for i in x:
            Total_skills.append(i)
else:
    fltr = data["skills"]
    for x in fltr:
        for i in x:
            Total_skills.append(i)

fig = px.histogram(
    x=Total_skills,
    labels={"x": "Skills"},
    title=f"{Job_Category} Distribution of Skills",
).update_xaxes(categoryorder="total descending")
fig.show()

# print ('--- Most used words')

# text = ""
# for i in data[data["Category"] == Job_Category]["Clean_Resume"].values:
#     text += i + " "

# plt.figure(figsize=(8, 8))

# x, y = np.ogrid[:300, :300]

# mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
# mask = 255 * mask.astype(int)

# wc = WordCloud(
#     width=800,
#     height=800,
#     background_color="white",
#     min_font_size=6,
#     repeat=True,
#     mask=mask,
# )
# wc.generate(text)

# plt.axis("off")
# plt.imshow(wc, interpolation="bilinear")
# plt.title(f"Most Used Words in {Job_Category} Resume", fontsize=20)
# plt.show(wc)

# print ('--- Entity Recognition')

# sent = nlp(data["Resume_str"].iloc[0])
# displacy.render(sent, style="ent", jupyter=True)

# displacy.render(sent[0:10], style="dep", jupyter=True, options={"distance": 90})

print ('--- Match Score')

input_skills = "Data Science,Data Analysis,Database,SQL,Machine Learning,tableau"

req_skills = input_skills.lower().split(",")
for i in data["Clean_Resume"].values:
    resume_skills = unique_skills(get_skills(i.lower()))
    score = 0
    for x in req_skills:
        if x in resume_skills:
            score += 1
    req_skills_len = len(req_skills)
    match = round(score / req_skills_len * 100, 1)
    
    #print("Resume text: " + i)
    print(f"The current Resume is {match}% matched to your requirements")
    print(resume_skills)

print ('--- Topic Modeling - LDA')

docs = data["Clean_Resume"].values
dictionary = corpora.Dictionary(d.split() for d in docs)
bow = [dictionary.doc2bow(d.split()) for d in docs]
lda = gensim.models.ldamodel.LdaModel
num_topics = 4
ldamodel = lda(
    bow, 
    num_topics=num_topics, 
    id2word=dictionary, 
    passes=50, 
    minimum_probability=0
)
#ldamodel.print_topics(num_topics=num_topics)
for i in range(0, ldamodel.num_topics):
    print(ldamodel.print_topic(i))

#pyLDAvis.enable_notebook()
visualisation = pyLDAvis.gensim_models.prepare(ldamodel, bow, dictionary)
pyLDAvis.save_html(visualisation, 'LDA_Visualization.html') 

print ('---')
end_time = datetime.datetime.now()
print ('Script has been running for ' + str((end_time - start_time).seconds // 60) + ' minutes ' + str((end_time - start_time).seconds % 60) + ' seconds')
