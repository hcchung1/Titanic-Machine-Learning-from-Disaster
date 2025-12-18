import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Load the training data
    import pandas as pd
    train_df = pd.read_csv('./data/train.csv')

    # Set the style for seaborn
    sns.set(style="whitegrid")

    # 1. Survival by Sex
    plt.figure(figsize=(8, 5))
    sns.barplot(data=train_df, x='Sex', y='Survived')
    plt.title('Survival Rate by Gender')
    plt.ylabel('Survival Rate')
    plt.savefig('survival_by_sex.png', dpi=300, bbox_inches='tight')

    # 2. Survival by Pclass
    plt.figure(figsize=(8, 5))
    sns.barplot(data=train_df, x='Pclass', y='Survived')
    plt.title('Survival Rate by Passenger Class')
    plt.ylabel('Survival Rate')
    plt.savefig('survival_by_pclass.png', dpi=300, bbox_inches='tight')

    # 3. Age distribution
    plt.figure(figsize=(10, 6))
    train_df[train_df['Survived']==1]['Age'].hist(alpha=0.5, bins=30, label='Survived')
    train_df[train_df['Survived']==0]['Age'].hist(alpha=0.5, bins=30, label='Did not survive')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.title('Age Distribution by Survival')
    plt.legend()
    plt.savefig('age_distribution.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    main()