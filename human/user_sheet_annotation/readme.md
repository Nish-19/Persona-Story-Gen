# Human Annotation 

Annotation File Path: ```annotation_data/```

There are three durectories each named after a user on Reddit.
Each directory contains the following files:
1. ```*.txt``` corresponding to particular stories written by the user
2. ```annotation_sheet.csv``` - containing the claims for annotation

The ```annotation_sheet.csv``` consists of the following fields:
**Input Fields**
1. source - the source story identifier
2. category - the category of the claim
3. claim - the claim made about the user story writing given the story
4. example - example from the story used to provide evidence for the claim

**Annotation Fields**
1. coherence - whether the claim is meaningful by itself
2. groundedness - whether the example is grounded in the story text
3. evidence - whether the example supports the claim about the user story writing
4. comments - comments justifying annotation/ providing reasoning (optional)


Annotation for ```coherence```, ```groundedness```, and ```evidence``` is done on a Likert scale of 1-5 where 1 represents the least score and 5 the highest for each field.