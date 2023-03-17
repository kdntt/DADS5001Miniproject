## DADS5001_Mini-Project
## School Bully
## Dataset Information
ค้นหา Data จาก Kaggle https://www.kaggle.com/datasets/leomartinelli/bullying-in-schools
File name: school_bully1.csv
This file has the following attributes
* จำนวนเด็กทั้งหมด (56,981 row)
* 17 column
  - Bullied on school property in past 12 months,
  - Bullied not on school property in past 12_months
  - Cyber bullied in past 12 months
  - Custom Age
  - Sex
  - Physically attacked
  - Physical fighting
  - Felt lonely
  - Close friends
  - Miss school no permission
  - Other students kind and helpful
  - Parents understand problems
  - Most of the time or always felt lonely
  - Missed classes or school without permission
  - Were underweight
  - Were overweight
  - Were obese
## Import Library & Import Data
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
df = pd.read_csv("school_bully1.csv")
df.info()
```
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 56981 entries, 0 to 56980
Data columns (total 18 columns):
 #   Column                                            Non-Null Count  Dtype 
---  ------                                            --------------  ----- 
 0   record                                            56981 non-null  int64 
 1   Bullied_on_school_property_in_past_12_months      56981 non-null  object
 2   Bullied_not_on_school_property_in_past_12_months  56981 non-null  object
 3   Cyber_bullied_in_past_12_months                   56981 non-null  object
 4   Custom_Age                                        56981 non-null  object
 5   Sex                                               56981 non-null  object
 6   Physically_attacked                               56981 non-null  object
 7   Physical_fighting                                 56981 non-null  object
 8   Felt_lonely                                       56981 non-null  object
 9   Close_friends                                     56981 non-null  object
 10  Miss_school_no_permission                         56981 non-null  object
 11  Other_students_kind_and_helpful                   56981 non-null  object
 12  Parents_understand_problems                       56981 non-null  object
 13  Most_of_the_time_or_always_felt_lonely            56981 non-null  object
 14  Missed_classes_or_school_without_permission       56981 non-null  object
 15  Were_underweight                                  56981 non-null  object
 16  Were_overweight                                   56981 non-null  object
 17  Were_obese                                        56981 non-null  object
dtypes: int64(1), object(17)
memory usage: 7.8+ MB
```
* import file ซึ่งนำมาจาก kaggle
* ตรวจสอบข้อมูลที่ได้มาว่า column ไหนสามารถใช้ได้บ้าง เป็น type อะไรบ้าง มี column ใดบ้างที่เป็น N/A

## Cleansing Data

```
df = df.drop( columns=['Were_obese'] )
df = df.drop( columns=['Missed_classes_or_school_without_permission'] )
df.info()
```
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 56981 entries, 0 to 56980
Data columns (total 16 columns):
 #   Column                                            Non-Null Count  Dtype 
---  ------                                            --------------  ----- 
 0   record                                            56981 non-null  int64 
 1   Bullied_on_school_property_in_past_12_months      56981 non-null  object
 2   Bullied_not_on_school_property_in_past_12_months  56981 non-null  object
 3   Cyber_bullied_in_past_12_months                   56981 non-null  object
 4   Custom_Age                                        56981 non-null  object
 5   Sex                                               56981 non-null  object
 6   Physically_attacked                               56981 non-null  object
 7   Physical_fighting                                 56981 non-null  object
 8   Felt_lonely                                       56981 non-null  object
 9   Close_friends                                     56981 non-null  object
 10  Miss_school_no_permission                         56981 non-null  object
 11  Other_students_kind_and_helpful                   56981 non-null  object
 12  Parents_understand_problems                       56981 non-null  object
 13  Most_of_the_time_or_always_felt_lonely            56981 non-null  object
 14  Were_underweight                                  56981 non-null  object
 15  Were_overweight                                   56981 non-null  object
dtypes: int64(1), object(15)
memory usage: 7.0+ MB
```
* ทำการ Drop data ที่คิดว่าไม่ได้ใช้ทิ้งไป
```
df.rename(columns = {'Bullied_on_school_property_in_past_12_months':'Bullied_on_school','Bullied_not_on_school_property_in_past_12_months':'Bullied_not_on_school', 
   'Cyber_bullied_in_past_12_months':'Cyber_bullied','Miss_school_no_permission':'Miss_school'}, inplace = True)
print(df.columns)
```
```
Index(['record', 'Bullied_on_school', 'Bullied_not_on_school', 'Cyber_bullied',
       'Custom_Age', 'Sex', 'Physically_attacked', 'Physical_fighting',
       'Felt_lonely', 'Close_friends', 'Miss_school',
       'Other_students_kind_and_helpful', 'Parents_understand_problems',
       'Most_of_the_time_or_always_felt_lonely', 'Were_underweight',
       'Were_overweight'],
      dtype='object')
```
* เปลี่ยนชื่อ column ที่ยาวๆให้สั้นลงและเข้าใจง่ายสะดวกกต่อการจัดการ
```
df['Custom_Age'] = df['Custom_Age'].str.replace('years old', '').str.replace('or younger', '').str.replace('or older', '')
df['Close_friends'] = df['Close_friends'].str.replace(' or more', '')
df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
```

* ทำการตัดข้อมูลใน column ทิ้ง และใส่ค่า Nanใน cell ที่ว่างเพื่อนำมาวิเคราะห์ต่อ

## Data analysis and Visualization
* เริ่มจากทำการหาคนที่โดนบูลลี่ และคนที่ไม่โดนบูลลี่จากทั้งในโรงเรียน นอกโรงเรียน และ cyberบูลลี่
```
on_school_counts = df['Bullied_on_school'].value_counts()
not_on_school_counts = df['Bullied_not_on_school'].value_counts()
cyber_bullied_counts = df['Cyber_bullied'].value_counts()

sns.set_palette("ch:.25")

fig, axes = plt.subplots(ncols=3, figsize=(12, 4))

sns.barplot(x=on_school_counts.index, y=on_school_counts.values, ax=axes[0])
axes[0].set_title('Bullied on School')
axes[0].set_xlabel('Bullied')
axes[0].set_ylabel('Count')

sns.barplot(x=not_on_school_counts.index, y=not_on_school_counts.values, ax=axes[1])
axes[1].set_title('Bullied not on School')
axes[1].set_xlabel('Bullied')
axes[1].set_ylabel('Count')

sns.barplot(x=cyber_bullied_counts.index, y=cyber_bullied_counts.values, ax=axes[2])
axes[2].set_title('Cyber Bullied')
axes[2].set_xlabel('Bullied')
axes[2].set_ylabel('Count')

plt.subplots_adjust(wspace=0.4)

plt.show()
```
![output1](https://user-images.githubusercontent.com/71549398/225819437-85d40b7c-26f1-4f1f-826c-abc2641d2cb0.png)

## Q1 เด็กที่โดนบูลลี่ส่วนใหญ่เป็นเพศอะไร และอายุเท่าไหร่

```
filtered_df3 = df[(df['Bullied_on_school'] == 'Yes')][['Sex']].value_counts().reset_index(name='count')
display(pd.DataFrame(filtered_df3))
plt.bar(filtered_df3['Sex'], filtered_df3['count'])

plt.xlabel('Sex')
plt.ylabel('Count')
plt.title('Counts of Bullied Students by Sex')

plt.show()
```
```
Sex	
Female	6761
Male	5007
```
![output2](https://user-images.githubusercontent.com/71549398/225819770-59c60921-f1de-4242-92f6-1f301218e5e7.png)

```
df2=df.dropna(subset='Custom_Age')
df2['Custom_Age']=df2['Custom_Age'].astype('Int64')

bully_age = df2.groupby(['Sex', 'Bullied_on_school']).agg({'Custom_Age': 'mean'})
display(bully_age)

bully_age.unstack('Sex').plot(kind='barh')

plt.xlabel('Bullied on school')
plt.ylabel('Mean age')
plt.title('Mean age of Females and Males by Bullied_on_school')

plt.show()
```
```
Custom_Age
Sex	    Bullied_on_school	Custom_Age
Female	      No	          14.928097
              Yes	          14.794166
Male	        No	          14.955129
              Yes	          14.772491
```
![output3](https://user-images.githubusercontent.com/71549398/225820204-02bbd895-7c79-4688-9f16-d8a75647239b.png)

* เด็กที่โดนกลั่นแกล้งในโรงเรียน เป็นเพศหญิง 6761 คน และ เป็นเพศชาย 5007 คน 
อายุเฉลี่ยนของเพศหญิงที่โดนกลั่นแกล้งคือ 14.794166 และที่ไม่โดนกลั่นแกล้งคือ 14.928097
อายุเฉลี่ยนของเพศชายที่โดนกลั่นแกล้งคือ 14.772491 และที่ไม่โดนกลั่นแกล้งคือ 14.955129

## Q2 เด็กที่โดนแกล้งในโรงเรียน ส่วนใหญ่เป็นเด็กที่มีรูปร่างอ้วนหรือผอม

```
filtered_df2 = df[(df['Bullied_on_school'] == 'Yes')] [['Were_underweight', 'Were_overweight']]
underweight2_counts = filtered_df2['Were_underweight'].value_counts()
display(pd.DataFrame(underweight2_counts))
overweight2_counts = filtered_df2['Were_overweight'].value_counts()
display(pd.DataFrame(overweight2_counts))
```
```
Were_underweight
No	7301
Yes	171

Were_overweight
No	5113
Yes	2359
```
* จากข้อมูลพบว่า เด็กที่มีรูปร่างอ้วนโดนบูลลี่เยอะกว่าเด็กที่มีรูปร่างผอม

## Q3 ถ้าในโรงเรียนเด็กอ้วนโดนบูลลีเยอะสุดแล้ว แล้วนอกโรงเรียน กับcyberbully จะยังเป็นเด็กที่อ้วนอยู่ไหม
```
filtered_df = df[(df['Bullied_on_school'] == 'Yes') & (df['Bullied_not_on_school'] == 'Yes') & (df['Cyber_bullied'] == 'Yes')][['Were_underweight', 'Were_overweight']]
underweight_counts = filtered_df['Were_underweight'].value_counts()
display(pd.DataFrame(underweight_counts))
overweight_counts = filtered_df['Were_overweight'].value_counts()
display(pd.DataFrame(overweight_counts))
```
```
Were_underweight
No	2346
Yes	50

Were_overweight
No	1651
Yes	745
```
* จากข้อมูลข้างต้นทำให้อยากรู้ต่อว่าส่วนใหญ่เป็นเพศอะไร แล้วอายุเท่าไหร่
```
allbully_sex = df[(df['Bullied_on_school'] == 'Yes') & (df['Bullied_not_on_school'] == 'Yes') & (df['Cyber_bullied'] == 'Yes')& (df['Were_overweight'] == 'Yes')][['Sex']].value_counts()
display(pd.DataFrame(allbully_sex))
```
```
Sex	
Female	449
Male	296
```
![output4](https://user-images.githubusercontent.com/71549398/225821603-27f3c7e2-cedb-4854-8022-02434adf3d35.png)

```
allbully_sex_f = df[(df['Bullied_on_school'] == 'Yes') & 
                    (df['Bullied_not_on_school'] == 'Yes') & 
                    (df['Cyber_bullied'] == 'Yes') & 
                    (df['Were_overweight'] == 'Yes') & 
                    (df['Sex'] == 'Female')][['Custom_Age']].value_counts().reset_index(name='count')

allbully_sex_f = allbully_sex_f.sort_values('Custom_Age')

sns.barplot(x='Custom_Age', y='count', data=allbully_sex_f, palette="ch:.25")

plt.xlabel('Custom Age')
plt.ylabel('Count')
plt.title('Counts of Female Students Who Experienced All Types of Bullying by Age')

plt.show()
```
![output5](https://user-images.githubusercontent.com/71549398/225821720-d1d48619-188e-4e5f-87b7-6909f0dcdbc6.png)

```
allbully_sex_m = df[(df['Bullied_on_school'] == 'Yes') & 
                    (df['Bullied_not_on_school'] == 'Yes') & 
                    (df['Cyber_bullied'] == 'Yes') & 
                    (df['Were_overweight'] == 'Yes') & 
                    (df['Sex'] == 'Male')][['Custom_Age']].value_counts().reset_index(name='count')

allbully_sex_m = allbully_sex_m.sort_values('Custom_Age')

sns.barplot(x='Custom_Age', y='count', data=allbully_sex_m, palette="ch:.25")

plt.xlabel('Custom Age')
plt.ylabel('Count')
plt.title('Counts of Male Students Who Experienced All Types of Bullying by Age')

plt.show()
```
![output6](https://user-images.githubusercontent.com/71549398/225821861-ac03514d-b2cf-4a23-bf56-d5d17e79c16e.png)

* ถ้านับจากการโดนบูลลี่ทั้งหมดแล้ว ไม่ว่าจะในโรงเรียน นอกโรงเรียน หรือ ทาง cyber พบว่า เด็กที่อ้วนจะโดนบูลลี่เยอะกว่าเด็กที่ผอม
หลังจากนั้นได้ทำมาศึกษาต่อว่าเด็กที่โดนบูลลี่นั้นเป็นเพศอะไร พบว่าเป็นเพศหญิง 60.3% และเป็นเพศชาย 39.7%
และได้ศึกษาต่อเพราะต้องการรู้ลักษณะทั้งหมดของเด็กที่โดนบูลลี่ พบว่าถ้าเป็นเพศหญิงจะอายุที่โดนกลั่นแกล้งมากที่สุดคือ 15 ปี และน้อยสุดคือ 18ปี
แต่ถ้าเป็นเพศชายอายุที่โดนบูลลี่มากที่สุดคือ 16 ปี และน้อยสุดคือ 12 ปี

## Q4 การโดนบูลลี่ในโรงเรียนทำให้เด็กไม่ยอมไปโรงเรียนจริงไหม

```
filtered_df4 = df[(df['Bullied_on_school'] == 'Yes')][['Miss_school']].value_counts()
display(pd.DataFrame(filtered_df4 ,columns=['Days Missed']))
```
```
Miss_school		Days Missed
0 days	          7642
1 or 2 days	      2172
3 to 5 days	      955
10 or more days	  361
6 to 9 days	      345
```
* จากนั้นก็เลยนำค่าเฉลี่ยมาเทียบ ระหว่างเด็กที่โดนบูลลี่ในโรงเรียน กับเด็กที่ไม่โดนบูลลี่ในโรงเรียน เพื่อมาหาความแตกต่าง

```
for index, row in df.iterrows():
    days = row['Miss_school']
    
    # Check if the value is missing and skip the row if true
    if pd.isnull(days):
        continue
    
    if days == '10 or more days':
        df.at[index, 'Miss_school'] = 10
    elif days == '6 to 9 days': 
        df.at[index, 'Miss_school'] = 7.5 #จำนวนวันตั้งแต่ 6-9หาร2 (6+7+8+9)/2
    elif days == '3 to 5 days':
        df.at[index, 'Miss_school'] = 4#จำนวนวันตั้งแต่ 3-5หาร2 (3+4+5)/2
    elif days == '1 or 2 days':
        df.at[index, 'Miss_school'] = 1.5#จำนวนวันตั้งแต่่ 1-2หาร2 (1+2)/2
    elif days == '0 days':
        df.at[index, 'Miss_school'] = 0
df['Miss_school'] = pd.to_numeric(df['Miss_school'], errors='coerce')
bully_and_miss_school = df[(df['Bullied_on_school'] == 'Yes')].agg({'Miss_school': 'mean'})
display(pd.DataFrame(bully_and_miss_school ,columns=['AVG get bullied on school Days Missed']))
not_bully_and_miss_school = df[(df['Bullied_on_school'] == 'No')].agg({'Miss_school': 'mean'})
display(pd.DataFrame(not_bully_and_miss_school,columns=['AVG not get bullied on school Days Missed'] ))
```
```

AVG get bullied on school Days Missed
Miss_school         	1.156906
AVG not get bullied on school Days Missed
Miss_school	          0.937998
```
* เมื่อเทียบระหว่าง เด็กที่โดนบูลลี่ในโรงเรียน กับเด็กที่ไม่โดนบูลลี่ในโรงเรียน เด็กที่โดนบูลลี่มีจำนวนวันเฉลี่ยที่จะขาดเรียนมากกว่า เด็กที่ไม่โดนบูลลี่ในโรงเรียน

## Q5 ยิ่งเด็กมีเพื่อนน้อย จะยิ่งโดนบูลลี่จริงไหม

```
getbully_count_closefriend = df[(df['Bullied_on_school'] == 'Yes') & (df['Bullied_not_on_school'] == 'Yes') & (df['Cyber_bullied'] == 'Yes')][['Close_friends']].value_counts()

plt.bar(getbully_count_closefriend.index.get_level_values(0), getbully_count_closefriend)

plt.title("Number of Close Friends for People who Experience Bullying")
plt.xlabel("Number of Close Friends")
plt.ylabel("Count")

plt.show()
```
![output7](https://user-images.githubusercontent.com/71549398/225822910-5dbb1dc7-af72-45d6-8aa1-053775b5ab79.png)

* กลายเป็นว่า เด็กที่มีเพื่อนเยอะยิ่งโดนบูลลี่ จึงทำมาเทียบกับเด็กที่ไม่โดนบูลลี่ ว่ามีเพื่อนเยอะหรือน้อย

```
not_getbully_count_closefriend = df[(df['Bullied_on_school'] == 'No') & (df['Bullied_not_on_school'] == 'No') & (df['Cyber_bullied'] == 'No')][['Close_friends']].value_counts()
plt.bar(not_getbully_count_closefriend.index.get_level_values(0), not_getbully_count_closefriend)

plt.title("Number of Close Friends for People who Experience Bullying")
plt.xlabel("Number of Close Friends")
plt.ylabel("Count")

plt.show()
```
![output8](https://user-images.githubusercontent.com/71549398/225823135-378e9fd9-d32b-4d87-bad5-d61f8ae98904.png)

* หลังจากได้ทำการ plot กราฟทั้ง2กลุ่มดูแล้วไม่เห็นความแตกต่างกัน จึงลองมาทดสอบสมมติฐานทางสถิติโดนใช้ z-test

```

from statsmodels.stats.proportion import proportions_ztest
get_bully_have_closefriend_ratio = df[(df['Bullied_on_school'] == 'Yes')][['Close_friends']].value_counts(normalize=True)
not_get_bully_have_closefriend_ratio = df[(df['Bullied_on_school'] == 'No')][['Close_friends']].value_counts(normalize=True)

# Define the sample sizes and successes for each group
n1 = sum(get_bully_have_closefriend.values)
x1 = np.array([get_bully_have_closefriend[0], get_bully_have_closefriend[1], get_bully_have_closefriend[2],get_bully_have_closefriend[3]])
n2 = sum(not_get_bully_have_closefriend_ratio.values)
x2 = np.array([not_get_bully_have_closefriend_ratio[0], not_get_bully_have_closefriend_ratio[1], not_get_bully_have_closefriend_ratio[2], not_get_bully_have_closefriend_ratio[3]])

# Perform the z-test for the difference in proportions
count = np.array([x1[3], x2[3]])  # number of successes in each group
nobs = np.array([n1, n2])  # sample sizes for each group
stat, pval = proportions_ztest(count, nobs)

# Print the results
print(f"Test statistic: {stat:.3f}")
print(f"P-value: {pval:.3f}")
if pval < 0.05:
    print("There is a significant difference in the proportion of Close_friends between the two groups.")
else:
    print("There is no significant difference in the proportion of Close_friends between the two groups.")
```
* ผลที่ออกมาคือ
```
Test statistic: 0.088
P-value: 0.930
There is no significant difference in the proportion of Close_friends between the two groups.
```
* จากนั้นเพื่อความชัวร์ จึงได้ลองทำการทดสอบกับเด็กที่โดนบูลลี่นอกโรงเรียน และทาง cyber bully
```
filtered_df6 = df[(df['Bullied_not_on_school'] == 'Yes')][['Close_friends']].value_counts(normalize=True)
filtered_df66 = df[(df['Bullied_not_on_school'] == 'No')][['Close_friends']].value_counts(normalize=True)
n1 = sum(filtered_df6.values)
x1 = np.array([filtered_df6[0], filtered_df6[1], filtered_df6[2], filtered_df6[3]])
n2 = sum(filtered_df66.values)
x2 = np.array([filtered_df66[0], filtered_df66[1], filtered_df66[2], filtered_df66[3]])

# Perform the z-test for the difference in proportions
count = np.array([x1[3], x2[3]])  # number of successes in each group
nobs = np.array([n1, n2])  # sample sizes for each group
stat, pval = proportions_ztest(count, nobs)

# Print the results
print(f"Test statistic: {stat:.3f}")
print(f"P-value: {pval:.3f}")
if pval < 0.05:
    print("There is a significant difference in the proportion of Close_friends between the two groups.")
else:
    print("There is no significant difference in the proportion of Close_friends between the two groups.")
```
```
Test statistic: 0.038
P-value: 0.970
There is no significant difference in the proportion of Close_friends between the two groups.
```
* ผลการทดสอบของคนที่โดนบูลลี่นอกโรงเรียน และ คนที่ไม่โดนบูลลี่นอกโรงเรียน

```
filtered_df7 = df[(df['Cyber_bullied'] == 'Yes')][['Close_friends']].value_counts(normalize=True)
filtered_df77 = df[(df['Cyber_bullied'] == 'No')][['Close_friends']].value_counts(normalize=True)
n1 = sum(filtered_df7.values)
x1 = np.array([filtered_df7[0], filtered_df7[1], filtered_df7[2], filtered_df7[3]])
n2 = sum(filtered_df77.values)
x2 = np.array([filtered_df77[0], filtered_df77[1], filtered_df77[2], filtered_df77[3]])

# Perform the z-test for the difference in proportions
count = np.array([x1[3], x2[3]])  # number of successes in each group
nobs = np.array([n1, n2])  # sample sizes for each group
stat, pval = proportions_ztest(count, nobs)

# Print the results
print(f"Test statistic: {stat:.3f}")
print(f"P-value: {pval:.3f}")
if pval < 0.05:
    print("There is a significant difference in the proportion of Close_friends between the two groups.")
else:
    print("There is no significant difference in the proportion of Close_friends between the two groups.")
```

```
Test statistic: 0.033
P-value: 0.974
There is no significant difference in the proportion of Close_friends between the two groups.
```
* ผลการทดสอบของคนที่โดน cyberbully และ ไม่โดน cyberbully


* จากข้อมูลพบว่า เด็กที่มีเพื่อนน้อย ไม่ได้ส่งผลต่อการโดนบูลลี่ การที่เด็กมีเพื่่อนเยอะกลับถูกบูลลี่เยอะกว่า
จึงได้ทำการเทียบระหว่าง 2 กลุ่ม ระหว่างเด็กที่โดนบูลลี่  กับเด็กที่ไม่โดนบูลลี่  ว่าจำนวนเพื่อนสนิทที่มี ทำให้โดนบูลลี่ แตกต่างกันแค่ไหน
หลังจากนั้นจึงพบว่าทั้งสองกลุ่ม ไม่ว่าจะมีเพื่อนกี่คนถ้าจะโดนบูลลี่ ก็โดนบูลลี่อยู่ดี
