# Titanic from Kaggle using machine learning techiniche
## Feature Engineering
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

---

## 三種特徵組合說明（RF / XGB / MLP）
- **RF 特徵**：
    - 缺失補值：Embarked→'S'，Fare→均值。
    - Title：從姓名切分出 Title1，再映射 Title2（Mlle/Ms→Miss，Mme→Mrs，醫官/軍階/貴族→Mr/Mrs）。
    - Ticket/Cabin：Ticket 去掉符號並取前綴；Cabin 取首字母，缺失→NoCabin。
    - Family_Size：Parch + SibSp。
    - 分類欄位（Sex/Embarked/Pclass/Title1/Title2/Cabin/Ticket_info）做 cat.codes 整數化。
    - Age 缺失用 RandomForestRegressor（2000 棵樹）根據 Embarked, Fare, Pclass, Sex, Family_Size, Title1, Title2, Cabin, Ticket_info 補值。
    - 最終特徵欄：Age, Embarked, Fare, Pclass, Sex, Family_Size, Title2, Ticket_info, Cabin。

- **XGB 特徵**：
    - 缺失補值：Embarked 先用同 Pclass 眾數，否則全域眾數 'S'；Fare→均值；Pclass→眾數。
    - Title2：從姓名取 Title1，Mlle/Ms→Miss，Mme→Mrs，罕見頭銜→Rare。
    - Ticket/Cabin：Ticket_info 前綴；Cabin 首字母，缺失→NoCabin；並計算 Ticket_info_freq、Cabin_freq 出現次數。
    - Family_Size：Parch + SibSp + 1；IsAlone：Family_Size==1。
    - Sex/Embarked/Title2 以 cat.codes 轉整數。
    - Age 缺失用 RandomForestRegressor（2000 棵樹）根據 Embarked, Fare, Pclass, Sex, Family_Size, IsAlone, Title2, Ticket_info_freq, Cabin_freq 補值。
    - 分箱與交互：FareBin（四分位數）、AgeBin（5 等分），Sex_Pclass＝Sex*Pclass，Pclass_AgeBin＝Pclass*AgeBin。
    - 最終特徵欄：Age, AgeBin, Embarked, Fare, FareBin, Pclass, Sex, Family_Size, IsAlone, Title2, Ticket_info_freq, Cabin_freq, Sex_Pclass, Pclass_AgeBin。

- **MLP 特徵**：
    - Title：從姓名抽取並合併（Mlle/Ms/Mme→Miss/Mrs；貴族→Royalty；醫師/軍階/牧師→Officer；其他罕見→Rare）。
    - 缺失補值：Embarked→眾數（缺省 'S'）；Fare→中位數；Age 以 Title+Pclass 分組中位數補，仍缺失再用全域中位數。
    - 家庭與艙房：FamilySize＝SibSp+Parch+1，IsAlone＝FamilySize==1，CabinDeck＝Cabin 首字母缺失→N，HasCabin 指示是否有 Cabin。
    - 票號群組：TicketGroup＝同票號人數，最大截斷為 4。
    - 分箱：AgeBin＝[0-12], [12-20], [20-40], [40-60], [60-80]；FareBin＝[-1,8], [8,15], [15,31], [31,100], [100+].
    - 刪除欄：PassengerId, Name, Ticket, Cabin；類別欄（Embarked, Title, CabinDeck, Pclass）做 one-hot，drop_first=True。
    - 標準化：對所有特徵使用 StandardScaler（獨立 fit train，再 transform test），並輸出 float32 供 MLP 使用。

## Result Accuracy
### Current Best Model: **Random Forest**
1. **MLP**: `0.77751`

2. **RandomForest**: `0.7967` (RF)

3. **GradientBoosting**: `0.7751`(RF)

4. **XGBoost**: `0.7895` (RF)

5. **LogisticRegression**: `0.7464`(RF)

6. **svm**: `0.7727` (RF)

7. **knn**: `0.7895` (RF)

8. **Emsembling Models**:
    - MLP
    - RandomForest...