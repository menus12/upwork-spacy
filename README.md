# Upwork analysis tool

This tool is helping to get strategic insights into Upwork market requirements, such as skills demand and CV relevance.

## Initialization

* Create python virtual environment: ```python -m venv venv-upwork-tool```
* Source into created vitual environment: ```source venv-upwork-tool/bin/activate```
* Install PyPI requirements: ```pip install -f requirements.txt```
* Export connection details to MYSQL database, which contains parsed Upwork data:

    ```
    MYSQL_HOST=1.2.3.4
    MYSQL_PORT=1488
    MYSQL_USER=login
    MYSQL_PASSWORD=password
    ```

## Named entity recognition

### Structure

Tool is using model to recognize specific keywords both in job postings and CVs. The structure is maintained in [masterlist.yaml](masterlist.yaml).

### Generation
Base YAML masterlist is created from example [excel skills table](https://docs.google.com/spreadsheets/u/1/d/12ALZam4SnZk9FL88GNBr2C-R90IDCuG9FhZnZFos5Ac/edit#gid=0) using ChatGPT.

Example prompt for ChatGPT to create such YAML structure from list of entities:

```
YOU:
consider following example of entity structure in YAML:

- id: Amazon
  products:
    - id: AWS
      aliases: 
        - Amazon Web Services
      features:
        - id: EC2
          aliases:
            - Elastic Compute Cloud
          scopes: 
            - id: VM
              aliases: 
                - Virtual machine
                - Virtual machines
                - Virtual machine instances

Based on this example provide analog for AWS ECR service

ChatGPT:
Based on the provided structure, an analogous YAML structure for the AWS ECR (Amazon Elastic Container Registry) service would look like this:

yaml
Copy code
- id: Amazon
  products:
    - id: AWS
      aliases: 
        - Amazon Web Services
      features:
        - id: ECR
          aliases:
            - Elastic Container Registry
          scopes: 
            - id: Container Registry
              aliases: 
                - Docker container registry
                - Private Docker registry

YOU:
good job. provide analog for following AWS features:
- ACM
- Amazon Connect
- API Gateway
- App Mesh
- AppSync
- Athena
- Auto Scaling
- Batch
- CDK

```

### Compilation

To build a model from YAML master list, [ner_builder.py](ner_builder.py) script is used to produce JSON list of entity rules. 

```bash 
./ner_builder.py --yaml masterlist.yaml --jsonl masterlist.jsonl
```

## Processing script

### Execution parameters

- ``--parse_jobs``: Parse jobs from the database
- ``--jobs_directory``: Parse jobs from JSONs in given directory
- ``--cv``: Structured CVs file in JSON format or excel skills tables
- ``--last``: Filter projects by number of days for database query
- ``--category``: Filter projects by category
- ``--sample``: Random sample of N jobs
- ``--draw_cv_skills``: Filename to draw CV skills distribution
- ``--draw_jobs_skills``: Filename to draw Upwork skills distribution
- ``--draw_categories``: Filename to draw job categories distribution
- ``--draw_countries``: Filename to draw job countries distribution
- ``--skills_relevance``: Threshold % for skills relevance
- ``--csv``: Filename to save relevance CSV table
- ``--matrix``: Filename to save skill matrix table
- ``--labels``: Space separated labels for extraction via entity ruler

### Execution examples

#### Example 1. Parse pre-processed jobs and skills tables to create skills matrix

Following command will:
 - parse job postings in given ``processed`` folder  
 - parse skills tables in ``skills_sheets`` folder
 - extract labels ``PRODUCT``, ``FEATURE`` and ``SCOPE`` labels from jobs and CVs
 - create skills matrix ``matrix.xls``
 - create YAML file ``new_skills.yaml`` with skills that absent in master list 

```bash
./process.py \
    --jobs_directory processed/ \
    --cv skills_sheets/ \
    --labels PRODUCT\ FEATURE\ SCOPE \
    --matrix matrix.xls \
    --new_skills_file new_skills.yaml
```

#### Example 2. Computing job relevance and skill matrix

Following command will process 500 random postings from the crawler database for 'devops' category, check for job relevance for each position in each CV with 75% threshold, will save relevance table in 'relevance.xls' and will compute skill matrix and save it to 'matrix.xls'

```bash
./process.py \
    --parse_jobs True \
    --sample 500 \
    --category devops \
    --labels PRODUCT\ FEATURE\ SCOPE \
    --cv example_cv.json \
    --skills_relevance 75
    --csv relevance.xls
    --matrix matrix.xls
```


## Tool maintanance

* aleksandr.gorbachev@sharp-dev.net
