# Restricted Boltzmann Machines for recommending Steam games

## Project Layout
- `output` folder contains RMSE plots for the train set (and sometimes also the test set) for all our experiments
in selecting different hyperparamaters
- `scripts` folder contains script versions of different ways of training the RBM, purely experimental do not use
- `len-scripts` contains (slightly modified) tutorial scripts provided by Len
- `rbm_notebook.ipynb` contains all code needed to read the Australian Steam dataset, train an RBM and evaluate is using HR@k, Recall@k and nDCG@k.
- other files can be ignored for now

## Overview
Pipelines:
- Aussies: Train on reviews, test on reviews (train test random split)
- Big: Train on reviews, test on reviews (train test random split)
- Big: Train on reviews, test on reviews (> 2 reviews, split in user on time, user has at least one train and test review)
- Train on games ownership


## Data
### Aussies reviews
steam_reviews:
- products: # games in user account
- compensation: Product received for free / NaN

### Aussies games
user => games, hours played
- \approx 5.1 million total interactions

### Reviews all
user => game review
- 7.8 million interactions
- lots of users with no, or just one review
- 626 794 unique users: 442 055 with just one review
- We removed that cannot be linked to an user (contain no user id field)
- => 1 485 611 unique user left, 3 176 223 unique reviews left
- Group on users ids
- Remove users that contain only one review (model cannot learn from a single review)
- => 581 343 unique users, 2 271 955 unique reviews left
- Split user's reviews in history and future. History and future both contain at least one review
- Train on history (1 404 885 reviews), test on future (708 285 reviews)

### Games all
- user => game, hours played, title etc
- 2.8 million total interactions


## Trials
### 