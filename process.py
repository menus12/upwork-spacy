#!/usr/bin/env python3

# common
import os
import json
import re
import argparse
import datetime
import html
import csv
import string 
import random
import mysql.connector

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
parser.add_argument('--jobs', type=str, help='Projects dump from database')
parser.add_argument('--cv', type=str, help='Structured CVs file')
parser.add_argument('--last', type=int, help='Number of days for database query')
parser.add_argument('--sample', type=int, help='Randomly pick N jobs from dump')
parser.add_argument('--draw_cv_skills', type=str, help='Filename to CV skills distribution')
parser.add_argument('--draw_jobs_skills', type=str, help='Filename to CV skills distribution')
parser.add_argument('--draw_categories', type=str, help='Filename to draw job categories distribution')
parser.add_argument('--draw_countries', type=str, help='Filename to draw job countries distribution')
parser.add_argument('--draw_topics', type=str, help='Filename to draw topic modeling distribution')
parser.add_argument('--num_topics', type=int, help='Number of topics to model')
parser.add_argument('--skills_relevance', type=int, help='Threshold for skills relevance')
parser.add_argument('--csv', type=str, help='Filename to save relevance CSV table')
args = parser.parse_args()

sample = 0
num_topics = 5
skills_relevance = 0

mysql_host = os.environ.get('MYSQL_HOST')
mysql_port = os.environ.get('MYSQL_PORT')
mysql_user = os.environ.get('MYSQL_USER')
mysql_password = os.environ.get('MYSQL_PASSWORD')
days = 0
db_query = "SELECT * FROM bidding_machine.approved_projects"

if args.cv is None:
    print ("CV file argument is missing")
    exit(1)

if mysql_host is None or mysql_port is None or mysql_user is None or mysql_password is None:
    print("Check environment variables for MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD")
    exit(1)

if args.last is not None:
    days = args.last
    db_query = db_query + " WHERE bidding_machine.approved_projects.date  > NOW() - INTERVAL " + str(days) + " DAY"

if args.sample is not None:
    sample = args.sample

if args.num_topics is not None:
    num_topics = args.num_topics

if args.skills_relevance is not None:
    skills_relevance = args.skills_relevance
    print("Skill relevance threshold: " + str(skills_relevance))
    

nlp = spacy.load("en_use_lg")
skill_pattern_path = "jz_skill_patterns.jsonl"

ruler = nlp.add_pipe("entity_ruler")
ruler.from_disk(skill_pattern_path)

def fetch_jobs():
    db = mysql.connector.connect(
        host=mysql_host,
        port=mysql_port,
        user=mysql_user,
        password=mysql_password
        )
    cursor = db.cursor()
    cursor.execute(db_query)
    row_headers=[x[0] for x in cursor.description] #this will extract row headers
    rv = cursor.fetchall()
    json_data=[]
    for result in rv:
        json_data.append(dict(zip(row_headers,result)))
    return json_data

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
    letters = ["a", "b", "c", "d", "e", "f", "g", 
               "h", "i", "j", "k", "l", "m", "n", 
               "o", "p", "q", "r", "s", "t", "u", 
               "v", "w", "x", "y", "z"]
    review = [
        lm.lemmatize(word)        
        for word in review
        if not word in set(stopwords.words("english") + letters)
    ]
    review = " ".join(review)
    return review

def match_score_skills(required_skills, position_skills):
    score = 0
    for x in required_skills:
        if x in position_skills:
            score += 1
    req_skills_len = len(required_skills)
    if req_skills_len > 0:
        match = round(score / req_skills_len * 100, 1)
        return match
    else: return 0

start_time = datetime.datetime.now()

print ('--- Loading data')

# f = open(args.jobs)                 # Opening JSON file
# source_file = json.loads(f.read())  # returns JSON object as  a dictionary
# print ('Jobs file is ' + args.jobs)
# f.close()                           # Closing file

source_file = fetch_jobs()
print ('Fetched', len(source_file), 'jobs from bidder database')

f = open(args.cv)                   # Opening JSON file
cv = json.loads(f.read())           # returns JSON object as  a dictionary
print ('CVs file is ' + args.cv)
f.close()                           # Closing file

print ('--- Parsing CV data')

cv_total_skills = []
for person in cv:
    person['total_skills'] = []
    for exp in person['experience']:
        # cleaning description
        exp['clean_description'] = clear_text(exp['description'])
        
        # extracting any other skills
        exp['skills'] = get_skills(exp["description"].lower() + ' ' +  exp["technologies"].lower())
        exp['skills'] = unique_skills(exp['skills'])
        for skill in exp['skills']:
            person['total_skills'].append(skill)
    cv_total_skills += person['total_skills']
    
    

print ('--- Parsing projects data')

seq = [x['id'] for x in source_file]
joblist = seq

if sample > 0:
    joblist = []
    for i in range(0, sample):
        n = random.randint(min(seq), max(seq))
        if n not in joblist:
            joblist.append(n)
        else: joblist.append(n + 1)
    print("  |--- Picking " + str(sample) + " IDs from jobs file")
    source_file = list(filter(lambda source_file: source_file['id'] in joblist, source_file))
    #print(", ".join(str(x) for x in joblist))
    
total_skills = []
docs = []

for project in source_file:
    # if sample > 0 and project['id'] not in joblist:
    #     continue
    
    # fix title
    project['title'] = re.sub(' - Upwork', "", project['title'])
    
    # fix encoding
    project['description'] = html.unescape(project['description'])
    
    # separate rate
    rate_re = '\$\d+-\$\d+'
    if "Rate not defined" in project['description']:
        project['rate_min']  = None
        project['rate_max']  = None
        project['description'] = re.sub('Rate not defined', " ", project['description'])
    rate = re.findall(rate_re, project['description'])
    if len(rate) > 0:
        project['rate_min']  = re.sub('\$','', rate[0].split('-')[0])
        project['rate_max']  = re.sub('\$','', rate[0].split('-')[1])
        project['description'] = re.sub(rate_re, " ", project['description'])

    # separate skills to technologies
    skills_re = 'Skills: ([a-zA-Z\d -\/;]+)'
    if "Skills:" in project['description']:        
        skills = re.findall(skills_re, project['description'])  
        if len(skills) > 0: project['technologies'] = skills.pop()
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
    project['clean_description'] = clear_text(project['description'])
    docs.append(project['clean_description'])
    #project['clean_description'] = project['description']
    
    # extracting any other skills
    project['skills'] = get_skills(project["description"].lower() + ' ' + project["technologies"].lower())
    project['skills'] = unique_skills(project['skills'])
    for skill in project['skills']:
        total_skills.append(skill)

    # print("ID:" + str(project['id']) + " | " + 
    #       project['title'] + 
    #       " | Skills: " + ", ".join(project['skills']) + 
    #       " | Category: " + project['category'])

print ('--- Computing skills matrix')

table = {}

skills = unique_skills(total_skills + cv_total_skills)
upwork_counters = []
sharpdev_counters = []
matrix = []

for skill in skills:
    upwork_counters.append(total_skills.count(skill))
    sharpdev_counters.append(cv_total_skills.count(skill))

table['skills'] = skills
table['upwork'] = upwork_counters
table['sharpdev'] = sharpdev_counters

for person in cv:
    table[person['name']] = []
    for skill in skills:
        table[person['name']].append(person['total_skills'].count(skill))
    
df = pd.DataFrame(table)
df = df.sort_values(by=['upwork'], ascending=[False])
df.to_csv('matrix.csv', sep='\t', index=False)
print(df)
    

print ('--- Computing skills and description relevance')

project_position_relevance = []

for project in source_file:
    # if project['id'] not in joblist:
    #     continue
    for person in cv:
        for exp in person['experience']:
            proj_skills = nlp(" ".join(project['skills']))
            pos_skills = nlp(" ".join(exp['skills']))
            skills_sim = proj_skills.similarity(pos_skills)
            if round(skills_sim * 100, 2) > skills_relevance:
                proj_desc = nlp(project['clean_description'])                
                pos_desc = nlp(exp['clean_description'])
                desc_sim = proj_desc.similarity(pos_desc)
                entry = [project['id'], project['url'],
                    person['id'], # cv_id
                    exp['id'], # cv position id
                    skills_sim,
                    #match_score_skills(project['skills'], cv['experience'][i]['skills']),
                    desc_sim]
                project_position_relevance.append(entry)
                print("ID: " + str(project['id']) + " | " + 
                    project['title'] + 
                    " | Skills: " + ", ".join(project['skills']) + 
                    " | Category: " + project['category'])
                print("  |--> " + 
                    person['name'] + 
                    " | " + exp['title'] + 
                    " | Skills match: " + str(round(skills_sim * 100, 2)) + 
                    "% | Desc match: " + str(round(desc_sim * 100, 2)) + "%")

if args.csv is not None:
    print ('--- Saving relevance table in ' + args.csv)
    with open(args.csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['project_id', 'upwork_url,' 'cv_id', 'cv_position_id', 'skills_match', 'description_match'])
        for entry in project_position_relevance:
            writer.writerow(entry)


if args.draw_cv_skills is not None:
    print("--- Plotting CV skills to " + args.draw_cv_skills)
    fig = px.histogram(
        x=cv_total_skills,
        labels={"x": "Skills"},
        title="Distribution of Skills for CVs",
    ).update_xaxes(categoryorder="total descending")
    #fig.show()
    fig.write_image(args.draw_cv_skills)

if args.draw_categories is not None:
    print("--- Plotting project categories to " + args.draw_categories)
    fig = px.histogram(
        source_file, x="category", title="Distribution of Project Categories"
    ).update_xaxes(categoryorder="total descending")
    #fig.show()
    fig.write_image(args.draw_categories)

if args.draw_countries is not None:
    print("--- Plotting Project Countries to " + args.draw_countries)
    fig = px.histogram(
        source_file, x="country", title="Distribution of Countries"
    ).update_xaxes(categoryorder="total descending")
    #fig.show()
    fig.write_image(args.draw_countries)

if args.draw_jobs_skills is not None:
    print("--- Plotting jobs skills to " + args.draw_jobs_skills)
    fig = px.histogram(
        x=total_skills,
        labels={"x": "Skills"},
        title="Distribution of Skills",
    ).update_xaxes(categoryorder="total descending")
    #fig.show()
    fig.write_image(args.draw_jobs_skills)



if args.draw_topics is not None:
    print ('--- Topic Modeling - LDA with', args.num_topics, 'topics')

    removal= ['CCONJ','PUNCT','PART','DET','ADP','SPACE', 'NUM', 'SYM']
    tokens = []
    
    # for summary in nlp.pipe(docs):
    #     proj_tok = [token.lemma_.lower() for token in summary if token.pos_ not in removal and not token.is_stop and token.is_alpha]
    #     tokens.append(proj_tok)
    # dictionary = corpora.Dictionary(tokens)
    # dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=1000)
    # corpus = [dictionary.doc2bow(doc) for doc in tokens]
    # lda = gensim.models.ldamodel.LdaModel
    # lda_model = lda(corpus=corpus, id2word=dictionary, iterations=50, num_topics=5, workers = 4, passes=10)
    # lda_model.print_topics(-1)
    # visualisation = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    # pyLDAvis.save_html(visualisation, 'LDA_Visualization.html') 

    dictionary = corpora.Dictionary(d.split() for d in docs)
    
    bow = [dictionary.doc2bow(d.split()) for d in docs]
    lda = gensim.models.ldamodel.LdaModel
    num_topics = num_topics
    ldamodel = lda(
        bow, 
        num_topics=num_topics, 
        id2word=dictionary, 
        passes=50, 
        minimum_probability=0
    )
    
    #ldamodel.print_topics(num_topics=num_topics)
    # for i in range(0, ldamodel.num_topics):
    #     print(ldamodel.print_topic(i))

    #pyLDAvis.enable_notebook()
    visualisation = pyLDAvis.gensim_models.prepare(ldamodel, bow, dictionary)
    print ('   ---> Saving LDA to', args.draw_topics)
    pyLDAvis.save_html(visualisation, 'LDA_Visualization.html') 
    




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




        
print ('---')
end_time = datetime.datetime.now()
print ('Script has been running for ' + str((end_time - start_time).seconds // 60) + ' minutes ' + str((end_time - start_time).seconds % 60) + ' seconds')
