import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Fundamentals_of_Data_Mining_Milestone1_Raw_Dataset.csv")

print("First 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())

print("\nNumber of rows and columns:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nNumber of duplicate records:")
print(df.duplicated().sum())

print("\nStatistical summary:")
print(df.describe())

print("\nRows containing missing values:")
print(df[df.isnull().any(axis=1)])

df.loc[
    (df["Attendance"] < 0) | (df["Attendance"] > 100),
    "Attendance"
] = pd.NA

numeric_columns = [
    "Age",
    "Attendance",
    "Quiz_Score",
    "Assignment_Score"
]

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

df = df.drop_duplicates()

df["Gender"] = df["Gender"].str.strip().str.title()

print("\nAge range:")
print(df["Age"].min(), "to", df["Age"].max())

print("\nAttendance range:")
print(df["Attendance"].min(), "to", df["Attendance"].max())

print("\nQuiz Score range:")
print(df["Quiz_Score"].min(), "to", df["Quiz_Score"].max())

print("\nAssignment Score range:")
print(df["Assignment_Score"].min(), "to", df["Assignment_Score"].max())

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nDuplicates after cleaning:")
print(df.duplicated().sum())

print("\nStandardized Gender values:")
print(df["Gender"].value_counts())

df["Average_Score"] = (
    df["Quiz_Score"] + df["Assignment_Score"]
) / 2

def categorize_score(score):
    if score >= 90:
        return "Excellent"
    elif score >= 80:
        return "Good"
    else:
        return "Needs Improvement"

df["Score_Category"] = df["Average_Score"].apply(categorize_score)

print("\nTransformed dataset:")
print(df.head())

overall_average = df["Average_Score"].mean()

print("\nOverall average score:")
print(round(overall_average, 2))

highest_average = df["Average_Score"].max()

print("\nHighest average score:")
print(highest_average)

lowest_average = df["Average_Score"].min()

print("\nLowest average score:")
print(lowest_average)

category_counts = df["Score_Category"].value_counts()

print("\nStudents per Score Category:")
print(category_counts)

course_average = df.groupby("Course")["Average_Score"].mean().sort_values(
    ascending=False
)

print("\nAverage score per course:")
print(course_average)

highest_course = course_average.idxmax()
highest_course_score = course_average.max()

print("\nCourse with the highest average score:")
print(highest_course)

print("Course average score:")
print(round(highest_course_score, 2))

plt.figure(figsize=(8, 5))

category_counts.plot(kind="bar")

plt.title("Number of Students per Score Category")
plt.xlabel("Score Category")
plt.ylabel("Number of Students")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("score_category_distribution.png")

plt.show()

df.to_csv("cleaned_dataset.csv", index=False)

print("\nCleaned dataset saved as:")
print("cleaned_dataset.csv")


"""
Data preprocessing is important because it improves the quality and reliability
of data before data mining. It helps identify missing values, duplicate records,
inconsistent text, and invalid numerical values. Cleaning the dataset prevents
these problems from affecting the accuracy of analysis and results. Properly
preprocessed data also makes it easier to identify useful patterns and trends.
"""
