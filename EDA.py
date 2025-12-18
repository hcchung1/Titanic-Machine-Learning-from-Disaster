import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import missingno as msno

def main():
    # Set style for better-looking plots
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 10
    plt.rcParams['font.family'] = 'serif'

    # Create output directory
    output_dir = 'figures'
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    data_dir = 'data'
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))

    print(f"Loaded {len(train_df)} training samples")
    print(f"Survival rate: {train_df['Survived'].mean():.2%}")

    # ============================================================
    # Figure 1: Survival Rate by Gender
    # ============================================================
    plt.figure(figsize=(8, 5))
    survival_by_sex = train_df.groupby('Sex')['Survived'].agg(['mean', 'count'])
    survival_by_sex['percentage'] = survival_by_sex['mean'] * 100

    ax = sns.barplot(data=train_df, x='Sex', y='Survived', 
                    palette=['#E74C3C', '#3498DB'], ci=None)
    plt.title('Survival Rate by Gender', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Gender', fontsize=12)
    plt.ylabel('Survival Rate', fontsize=12)
    plt.ylim(0, 1)

    # Add percentage labels on bars
    for i, (idx, row) in enumerate(survival_by_sex.iterrows()):
        ax.text(i, row['mean'] + 0.02, f"{row['percentage']:.1f}%\n(n={int(row['count'])})", 
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'survival_by_sex.png'), 
                dpi=300, bbox_inches='tight')
    print("✓ Generated: survival_by_sex.png")
    plt.close()

    # ============================================================
    # Figure 2: Survival Rate by Passenger Class
    # ============================================================
    plt.figure(figsize=(8, 5))
    survival_by_class = train_df.groupby('Pclass')['Survived'].agg(['mean', 'count'])
    survival_by_class['percentage'] = survival_by_class['mean'] * 100

    ax = sns.barplot(data=train_df, x='Pclass', y='Survived', 
                    palette=['#2ECC71', '#F39C12', '#E74C3C'], ci=None)
    plt.title('Survival Rate by Passenger Class', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Passenger Class', fontsize=12)
    plt.ylabel('Survival Rate', fontsize=12)
    plt.ylim(0, 1)
    plt.xticks([0, 1, 2], ['1st Class', '2nd Class', '3rd Class'])

    # Add percentage labels on bars
    for i, (idx, row) in enumerate(survival_by_class.iterrows()):
        ax.text(i, row['mean'] + 0.02, f"{row['percentage']:.1f}%\n(n={int(row['count'])})", 
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'survival_by_pclass.png'), 
                dpi=300, bbox_inches='tight')
    print("✓ Generated: survival_by_pclass.png")
    plt.close()

    # ============================================================
    # Figure 3: Age Distribution by Survival
    # ============================================================
    plt.figure(figsize=(10, 6))
    survived = train_df[train_df['Survived'] == 1]['Age'].dropna()
    died = train_df[train_df['Survived'] == 0]['Age'].dropna()

    plt.hist(survived, bins=30, alpha=0.6, label='Survived', color='#3498DB', edgecolor='black')
    plt.hist(died, bins=30, alpha=0.6, label='Did Not Survive', color='#E74C3C', edgecolor='black')

    plt.xlabel('Age (years)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Age Distribution by Survival Status', fontsize=14, fontweight='bold', pad=15)
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(axis='y', alpha=0.3)

    # Add statistics
    textstr = f'Survived:\n  Mean: {survived.mean():.1f}\n  Median: {survived.median():.1f}\n\n'
    textstr += f'Did Not Survive:\n  Mean: {died.mean():.1f}\n  Median: {died.median():.1f}'
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'age_distribution.png'), 
                dpi=300, bbox_inches='tight')
    print("✓ Generated: age_distribution.png")
    plt.close()

    # ============================================================
    # Figure 4: Missing Values Heatmap
    # ============================================================
    plt.figure(figsize=(10, 8))
    msno.matrix(train_df, figsize=(10, 8), fontsize=10, sparkline=False)
    plt.title('Missing Value Pattern in Training Data', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'missing_values_heatmap.png'), 
                dpi=300, bbox_inches='tight')
    print("✓ Generated: missing_values_heatmap.png")
    plt.close()

    # ============================================================
    # Figure 5: Combined Survival Analysis (Sex + Class)
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left plot: Survival by Sex and Class
    survival_sex_class = train_df.groupby(['Sex', 'Pclass'])['Survived'].mean().unstack()
    survival_sex_class.plot(kind='bar', ax=axes[0], 
                            color=['#2ECC71', '#F39C12', '#E74C3C'], 
                            width=0.8, edgecolor='black')
    axes[0].set_title('Survival Rate by Gender and Class', fontsize=12, fontweight='bold', pad=10)
    axes[0].set_xlabel('Gender', fontsize=11)
    axes[0].set_ylabel('Survival Rate', fontsize=11)
    axes[0].set_xticklabels(['Female', 'Male'], rotation=0)
    axes[0].legend(['1st Class', '2nd Class', '3rd Class'], title='Passenger Class')
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis='y', alpha=0.3)

    # Right plot: Fare distribution by Survival
    axes[1].hist(train_df[train_df['Survived'] == 1]['Fare'].dropna(), 
                bins=30, alpha=0.6, label='Survived', color='#3498DB', edgecolor='black')
    axes[1].hist(train_df[train_df['Survived'] == 0]['Fare'].dropna(), 
                bins=30, alpha=0.6, label='Did Not Survive', color='#E74C3C', edgecolor='black')
    axes[1].set_xlabel('Fare (£)', fontsize=11)
    axes[1].set_ylabel('Count', fontsize=11)
    axes[1].set_title('Fare Distribution by Survival Status', fontsize=12, fontweight='bold', pad=10)
    axes[1].legend(fontsize=10)
    axes[1].set_xlim(0, 300)
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_survival_analysis.png'), 
                dpi=300, bbox_inches='tight')
    print("✓ Generated: combined_survival_analysis.png")
    plt.close()

    # ============================================================
    # Figure 6: Correlation Heatmap
    # ============================================================
    plt.figure(figsize=(10, 8))

    # Select numerical features for correlation
    numeric_features = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    correlation_data = train_df[numeric_features].corr()

    sns.heatmap(correlation_data, annot=True, fmt='.2f', cmap='RdYlGn', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), 
                dpi=300, bbox_inches='tight')
    print("✓ Generated: correlation_heatmap.png")
    plt.close()

    # ============================================================
    # Summary Statistics
    # ============================================================
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"\nTotal samples: {len(train_df)}")
    print(f"Survival rate: {train_df['Survived'].mean():.2%}")
    print(f"\nMissing values:")
    print(train_df.isnull().sum()[train_df.isnull().sum() > 0])
    print(f"\nSurvival by Gender:")
    print(train_df.groupby('Sex')['Survived'].agg(['mean', 'count']))
    print(f"\nSurvival by Class:")
    print(train_df.groupby('Pclass')['Survived'].agg(['mean', 'count']))
    print("\n✓ All EDA plots generated successfully!")
    print(f"✓ Saved to '{output_dir}/' directory")


if __name__ == "__main__":
    main()