# Install required libraries
!pip install pandas matplotlib seaborn textblob wordcloud plotly --quiet
!pip install squarify --quiet

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
import squarify
import requests
from io import StringIO

# Set professional color scheme
PROFESSIONAL_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", 
                        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

# Load the dataset directly from GitHub
github_raw_url = 'https://raw.githubusercontent.com/krish8070/FUTURE_DS_03/main/Student_Satisfaction_Survey.csv'

try:
    response = requests.get(github_raw_url)
    response.raise_for_status()
    csv_data = StringIO(response.text)
    df = pd.read_csv(csv_data)
    print(f"✅ Dataset loaded successfully with {len(df)} records")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    from google.colab import files
    uploaded = files.upload()
    file_name = list(uploaded.keys())[0]
    df = pd.read_csv(file_name)
    print(f"✅ Local file '{file_name}' loaded successfully")

# Data Cleaning
df['Average_Rating'] = df['Average/ Percentage'].str.split('/').str[0].str.strip().astype(float)
df['Course_Level'] = df['Basic Course'].apply(lambda x: 'PG' if ('MA' in x or 'MSC' in x) else 'UG')

# 1. Enhanced Overall Satisfaction Analysis
plt.figure(figsize=(12, 7))
sns.set_style("whitegrid")
ax = sns.histplot(df['Average_Rating'], bins=12, kde=True, 
                 color=PROFESSIONAL_PALETTE[0], alpha=0.8)
plt.title('Distribution of Average Ratings', fontsize=16, pad=15)
plt.xlabel('Average Rating (1-5 scale)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.axvline(df['Average_Rating'].mean(), color=PROFESSIONAL_PALETTE[1], 
            linestyle='--', linewidth=2, label=f'Mean: {df["Average_Rating"].mean():.2f}')
plt.legend()
sns.despine(left=True)
plt.tight_layout()
plt.show()

# 2. Enhanced Course-wise Performance with Annotations
plt.figure(figsize=(14, 10))
course_ratings = df.groupby('Basic Course')['Average_Rating'].mean().sort_values(ascending=False)
ax = sns.barplot(x=course_ratings.values, y=course_ratings.index, 
                palette=sns.color_palette("viridis", len(course_ratings)))
plt.title('Program Performance Comparison', fontsize=18, pad=15)
plt.xlabel('Average Rating', fontsize=12)
plt.ylabel('')
plt.xlim(0, 5)
plt.grid(axis='x', alpha=0.2)

# Add data labels
for i, v in enumerate(course_ratings.values):
    ax.text(v + 0.05, i, f"{v:.2f}", color='black', ha='left', va='center')

sns.despine(left=True)
plt.tight_layout()
plt.show()

# 3. Enhanced Question Analysis with Heatmap
question_ratings = df.groupby('Questions')['Average_Rating'].mean().sort_values(ascending=False)

plt.figure(figsize=(16, 12))
sns.heatmap(pd.DataFrame(question_ratings).T, 
            annot=True, fmt=".2f", 
            cmap="YlGnBu", 
            cbar_kws={'label': 'Average Rating'},
            linewidths=0.5)
plt.title('Question Performance Heatmap', fontsize=16, pad=15)
plt.xlabel('Survey Questions', fontsize=12)
plt.ylabel('')
plt.xticks(rotation=90)
plt.yticks([])
plt.tight_layout()
plt.show()

# 4. Enhanced Sentiment Analysis with Donut Chart
def get_sentiment(text):
    analysis = TextBlob(str(text))
    pol = analysis.sentiment.polarity
    if pol > 0.2:
        return 'Positive'
    elif pol < -0.2:
        return 'Negative'
    else:
        return 'Neutral'

df['Sentiment'] = df['Questions'].apply(get_sentiment)

plt.figure(figsize=(10, 8))
sentiment_counts = df['Sentiment'].value_counts()
colors = [PROFESSIONAL_PALETTE[2], PROFESSIONAL_PALETTE[8], PROFESSIONAL_PALETTE[3]]
plt.pie(sentiment_counts, labels=sentiment_counts.index, 
        autopct='%1.1f%%', startangle=90, 
        colors=colors, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
centre_circle = plt.Circle((0,0), 0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.title('Sentiment Distribution in Survey Questions', fontsize=16, pad=20)
plt.axis('equal')
plt.tight_layout()
plt.show()

# 5. Enhanced Weightage Analysis with Stacked Bar Chart
weightage_cols = ['Weightage 1', 'Weightage 2', 'Weightage 3', 'Weightage 4', 'Weightage 5']
weightage_totals = df[weightage_cols].sum()

plt.figure(figsize=(14, 8))
sns.set_palette("RdYlGn")
ax = weightage_totals.plot(kind='bar', width=0.7, edgecolor='white')
plt.title('Response Distribution Across Rating Scale', fontsize=16, pad=15)
plt.xlabel('Rating Scale', fontsize=12)
plt.ylabel('Total Responses', fontsize=12)
plt.xticks(ticks=range(5), labels=['1 (Very Dissatisfied)', '2', '3 (Neutral)', '4', '5 (Very Satisfied)'], 
           rotation=0, fontsize=10)

# Add data labels
for i, v in enumerate(weightage_totals):
    ax.text(i, v + 100, str(v), ha='center', fontsize=10)

plt.grid(axis='y', alpha=0.2)
sns.despine()
plt.tight_layout()
plt.show()

# 6. Enhanced Correlation Analysis with Pair Plot
sns.pairplot(df[weightage_cols], diag_kind='kde', 
             plot_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'k'},
             diag_kws={'color': PROFESSIONAL_PALETTE[0]})
plt.suptitle('Correlation Between Rating Categories', y=1.02, fontsize=16)
plt.tight_layout()
plt.show()

# 7. Top Recommendations with Treemap
# Prepare data
lowest_rated = df.groupby('Questions')['Average_Rating'].mean().nsmallest(5)
highest_rated = df.groupby('Questions')['Average_Rating'].mean().nlargest(5)

# Create treemap for improvement areas
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
squarify.plot(sizes=lowest_rated.values, 
             label=[f"{q[:40]}...\nRating: {r:.2f}" for q, r in lowest_rated.items()],
             color=sns.color_palette("Reds", len(lowest_rated)),
             alpha=0.8, text_kwargs={'fontsize':10})
plt.title('Top 5 Areas for Improvement', fontsize=14)
plt.axis('off')

# Create treemap for strengths
plt.subplot(1, 2, 2)
squarify.plot(sizes=highest_rated.values, 
             label=[f"{q[:40]}...\nRating: {r:.2f}" for q, r in highest_rated.items()],
             color=sns.color_palette("Greens", len(highest_rated)),
             alpha=0.8, text_kwargs={'fontsize':10})
plt.title('Top 5 Strongest Aspects', fontsize=14)
plt.axis('off')

plt.suptitle('Key Recommendations', fontsize=18, y=0.98)
plt.tight_layout()
plt.show()

# 8. Comparative Analysis with Radar Chart
def create_radar_chart(program_name):
    program_df = df[df['Basic Course'] == program_name]
    categories = program_df['Questions'].tolist()
    ratings = program_df['Average_Rating'].tolist()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=ratings,
        theta=categories,
        fill='toself',
        name=program_name,
        line=dict(color=PROFESSIONAL_PALETTE[0])
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5]
            )),
        showlegend=False,
        title=f'Performance Radar Chart: {program_name}',
        title_font_size=16
    )
    
    fig.show()

# Create radar charts for top and bottom programs
print("\nGenerating Radar Charts for Top and Bottom Programs...")
create_radar_chart(course_ratings.idxmax())
create_radar_chart(course_ratings.idxmin())

# 9. UG vs PG Comparison
plt.figure(figsize=(10, 6))
sns.boxplot(x='Course_Level', y='Average_Rating', data=df, 
            palette=[PROFESSIONAL_PALETTE[0], PROFESSIONAL_PALETTE[4]], 
            showmeans=True, meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black"})
plt.title('Satisfaction Comparison: UG vs PG Programs', fontsize=16, pad=15)
plt.xlabel('Program Level', fontsize=12)
plt.ylabel('Average Rating', fontsize=12)
plt.grid(axis='y', alpha=0.2)
sns.despine()
plt.tight_layout()
plt.show()

# 10. Enhanced Word Cloud
all_text = " ".join(df['Questions'].dropna())
wordcloud = WordCloud(width=1200, height=600, 
                      background_color='white', 
                      colormap='viridis', 
                      max_words=150, 
                      contour_width=3, 
                      contour_color='steelblue').generate(all_text)

plt.figure(figsize=(18, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Survey Questions', fontsize=20, pad=20)
plt.tight_layout()
plt.show()

# 11. Trend Analysis Over Time (Simulated)
# Since we don't have time data, we'll simulate by program
plt.figure(figsize=(14, 8))
sns.lineplot(data=df, x='Basic Course', y='Average_Rating', 
             err_style=None, marker='o', 
             color=PROFESSIONAL_PALETTE[0], linewidth=2.5)
plt.title('Satisfaction Trends Across Programs', fontsize=16, pad=15)
plt.xlabel('Program', fontsize=12)
plt.ylabel('Average Rating', fontsize=12)
plt.xticks(rotation=90)
plt.grid(axis='y', alpha=0.3)
plt.axhline(y=df['Average_Rating'].mean(), color=PROFESSIONAL_PALETTE[1], linestyle='--', 
            label=f'Overall Mean: {df["Average_Rating"].mean():.2f}')
plt.legend()
sns.despine()
plt.tight_layout()
plt.show()

# Final Enhanced Summary
print("\n" + "="*80)
print("📊 COMPREHENSIVE ANALYSIS SUMMARY".center(80))
print("="*80)
print(f"• Overall average rating: {df['Average_Rating'].mean():.2f}/5.00")
print(f"• Most satisfied program: {course_ratings.idxmax()} ({course_ratings.max():.2f})")
print(f"• Least satisfied program: {course_ratings.idxmin()} ({course_ratings.min():.2f})")
print(f"• UG vs PG Satisfaction: UG: {df[df['Course_Level']=='UG']['Average_Rating'].mean():.2f} | PG: {df[df['Course_Level']=='PG']['Average_Rating'].mean():.2f}")
print(f"• Most positive sentiment: {sentiment_counts.idxmax()} ({sentiment_counts.max()/len(df):.1%})")
print(f"• Most common rating: {weightage_totals.idxmax()} responses")
print("="*80)
print("💡 KEY RECOMMENDATIONS".center(80))
print("="*80)
print("1. Focus on enhancing mentoring processes for cognitive/social/emotional growth")
print("2. Improve ICT infrastructure and faculty training for better tool utilization")
print("3. Develop personalized strength-based development programs for students")
print("4. Implement best practices from top-performing programs across all departments")
print("5. Create targeted interventions for PG programs to improve satisfaction")
print("6. Establish continuous feedback mechanisms for quality improvement")
print("="*80)"now convert this in to html code like before"