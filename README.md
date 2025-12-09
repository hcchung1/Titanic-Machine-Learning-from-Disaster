# Titanic from Kaggle using machine learning techiniche
### Feature Engineering
1. **Title Extraction**
    - Extract titles from Name (e.g., Mr, Mrs, Miss, Master)
    - Combine rare titles:
      - Mlle, Ms, Mme → Miss/Mrs
      - Lady, Countess, Sir, Don, Dona → Royalty
      - Col, Major, Capt, Dr, Rev → Officer
      - Very rare titles → Rare

2. **Missing Value Imputation**
    - Embarked: Impute with mode
    - Fare: Impute with median
    - Age: Impute with median grouped by Title and Pclass

3. **Feature Creation**
    - FamilySize: SibSp + Parch + 1 (total family members)
    - IsAlone: Binary indicator for solo travelers (FamilySize == 1)
    - CabinDeck: Extract cabin letter (A, B, C, etc.), fill missing with 'N'
    - HasCabin: Binary indicator for cabin information availability
    - TicketGroup: Count of passengers sharing the same ticket (capped at 4)

4. **Binning**
    - AgeBin: Age binned into 5 groups: [0-12], [12-20], [20-40], [40-60], [60-80]
    - FareBin: Fare binned into 5 groups: [-1, 8], [8, 15], [15, 31], [31, 100], [100+]

5. **Encoding**
    - Sex: Convert to binary (female=1, male=0)
    - Categorical Features: One-hot encoding with drop_first=True to avoid multicollinearity
      - Embarked, Title, CabinDeck, Pclass

6. **Feature Removal**
    - Remove PassengerId, Name, Ticket, Cabin (information already extracted)

7. **Scaling**
    - Apply StandardScaler to all features
    - Fit and transform separately on train and test sets to prevent data leakage