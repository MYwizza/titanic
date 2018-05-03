# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 13:26:51 2018

@author: Wizza
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_data=pd.read_csv(r'C:\Users\Wizza\Documents\Python Scripts\titanic\train.csv')
test_data=pd.read_csv(r'C:\Users\Wizza\Documents\Python Scripts\titanic\test.csv')

sns.set_style('whitegrid')
print(train_data.info())
train_data.Survived.value_counts().plot.pie(autopct='%.2f%%')

#缺失值处理 用众数'Embarked'
train_data.Embarked[train_data.Embarked.isnull()]=train_data.Embarked.dropna().mode().values

#缺失值较多的，赋予代表缺失的值,Cabin
train_data.Cabin=train_data.Cabin.fillna('U0')

#使用回归 随机森林模型预测Age缺失值
from sklearn.ensemble import RandomForestRegressor

age_df=train_data[['Age','Survived','Fare','Parch','SibSp','Pclass']]
age_df_notnull=age_df.loc[age_df['Age'].notnull()]
age_df_isnull=age_df.loc[age_df['Age'].isnull()]
x=age_df_notnull.values[:,1:]
y=age_df_notnull.values[:,0]
RFR=RandomForestRegressor(n_estimators=1000,n_jobs=-1)
RFR.fit(x,y)
predictAges=RFR.predict(age_df_isnull.values[:,1:])
train_data.Age[train_data.Age.isnull()]=predictAges
print(train_data.info())
#分析数据关系
#1性别与生存的关系(sex-survived)
train_data.groupby(['Sex','Survived'])['Survived'].count()
train_data[['Sex','Survived']].groupby('Sex').mean().plot.bar()

#船舱等级与生存的关系（Pclass-survived)
train_data.groupby(['Pclass','Survived'])['Survived'].count()
train_data[['Pclass','Survived']].groupby('Pclass').mean().plot.bar()
#不同等级船舱的男女生存率
train_data.groupby(['Pclass','Sex','Survived'])['Survived'].count()
train_data[['Pclass','Sex','Survived']].groupby(['Pclass','Sex']).mean().plot.bar()

#年龄与生存的关系
fix,ax=plt.subplots(1,2,figsize=(18,8))
sns.violinplot('Pclass','Age',hue='Survived',data=train_data,split=True,ax=ax[0])
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(range(0,100,10))
sns.violinplot('Sex','Age',hue='Survived',data=train_data,split=True,ax=ax[1])
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0,100,10))

#分析总体年龄分布
plt.figure(figsize=(12,5))
plt.subplot(121)
train_data['Age'].hist(bins=70)
plt.xlabel('Age')
plt.ylabel('sum')

plt.subplot(122)
train_data.boxplot(column='Age',showfliers=False)

#不同年龄下生存与非生存分别
facet=sns.FacetGrid(train_data,hue='Survived',aspect=4)
facet.map(sns.kdeplot,'Age',shade=True)
facet.set(xlim=(0,train_data['Age'].max()))
facet.add_legend()
#不同年龄下的平均生存率
fix,axis1=plt.subplots(1,1,figsize=(18,4))
train_data['Age_int']=train_data['Age'].astype(int)
average_age=train_data[['Age_int','Survived']].groupby('Age_int',as_index=False).mean()
sns.barplot(x='Age_int',y='Survived',data=average_age)
#划分不同年龄段，儿童，少年，成年和老年，分析生还情况
bins=[0,12,18,65,100]
train_data['Age_group']=pd.cut(train_data['Age'],bins)
by_age=train_data[['Age_group','Survived']].groupby('Age_group').mean()
by_age.plot(kind='bar')

#称呼与生存的关系
train_data['Title']=train_data['Name'].str.extract('([A-Za-z]+)\.',expand=False)
pd.crosstab(train_data['Title'],train_data['Sex'])
train_data[['Title','Survived']].groupby('Title').mean().plot.bar()
#名字长度与生存率关系
fig,axis1=plt.subplots(1,1,figsize=(18,4))
train_data['Name_length']=train_data['Name'].apply(len)
name_length=train_data[['Name_length','Survived']].groupby('Name_length',as_index=False).mean()
sns.barplot(x='Name_length',y='Survived',data=name_length)

#有无兄弟姐妹和生存的关系
sibsp_df=train_data[train_data['SibSp']!=0]
no_sibsp_df=train_data[train_data['SibSp']==0]
plt.figure(figsize=(10,5))
plt.subplot(121)
sibsp_df['Survived'].value_counts().plot.pie(labels=['No Survived','Survived'],autopct='%.2f%%')
plt.xlabel('sibsp')

plt.subplot(122)
no_sibsp_df['Survived'].value_counts().plot.pie(labels=['No Survived','Survived'],autopct='%.2f%%')
plt.xlabel('no_sibsp')

#有无父母子女和生存的关系
parch_df=train_data[train_data['Parch']!=0]
no_parch_df=train_data[train_data['Parch']==0]
plt.figure(figsize=(10,5))
plt.subplot(121)
parch_df['Survived'].value_counts().plot.pie(labels=['No Survived','Survived'],autopct='%.2f%%')
plt.xlabel('parch')

plt.subplot(122)
no_parch_df['Survived'].value_counts().plot.pie(labels=['No Survived','Survived'],autopct='%.2f%%')
plt.xlabel('no_parch')

#亲友的人数与生存的关系
fig,ax=plt.subplots(1,2,figsize=(18,8))
train_data[['Parch','Survived']].groupby('Parch').mean().plot.bar(ax=ax[0])
ax[0].set_title('Parch and Survived')
train_data[['SibSp','Survived']].groupby('SibSp').mean().plot.bar(ax=ax[1])
ax[1].set_title('SibSp and Survived')

train_data['family_size']=train_data['Parch']+train_data['SibSp']+1
train_data[['family_size','Survived']].groupby('family_size').mean().plot.bar()

#票价分布与生存的关系
plt.figure(figsize=(10,5))
train_data['Fare'].hist(bins=70)
train_data.boxplot(column='Fare',by='Survived',showfliers=False)
print(train_data['Fare'].describe())
#票价均值与方差的关系
fare_not_survived=train_data['Fare'][train_data['Survived']==0]
fare_survived=train_data['Fare'][train_data['Survived']==1]
average_fare=pd.DataFrame([fare_not_survived.mean(),fare_survived.mean()])
std_fare=pd.DataFrame([fare_not_survived.std(),fare_survived.std()])
average_fare.plot(yerr=std_fare,kind='bar',legend=False)

#船舱类型与生存
train_data['Has_Cabin']=train_data['Cabin'].apply(lambda x:0 if x=='U0' else 1)
train_data[['Has_Cabin','Survived']].groupby('Has_Cabin').mean().plot.bar()
#对不同类型的船舱进行分析
#使用Factorize处理多变量
train_data['Cabinlevel']=train_data['Cabin'].map(lambda x:re.compile('([a-zA-Z]+)').search(x).group())
train_data['Cabinlevel']=pd.factorize(train_data['Cabinlevel'])[0]
train_data[['Cabinlevel','Survived']].groupby('Cabinlevel').mean().plot.bar()

#港口与生存
sns.countplot('Embarked',hue='Survived',data=train_data)
plt.title('Embarked and Survived')
sns.factorplot('Embarked','Survived',data=train_data,size=3,aspect=2)
plt.title('Embarked and Survived rate')

address='https://blog.csdn.net/Koala_Tree/article/details/78725881'

#定性转换船舱,使用get_dummies方法
embark_dummies=pd.get_dummies(train_data['Embarked'])
train_data=train_data.join(embark_dummies)
train_data.drop(['Embarked'],axis=1,inplace=True)
#定量转换，将很大范围的数值映射到小范围内
from sklearn import preprocessing
#将Age转到（-1，1）之间
scaler=preprocessing.StandardScaler()
train_data['Age_scaled']=scaler.fit_transform(train_data['Age'].values.reshape(-1,1),)

#将连续数据离散化
#将Fare离散为5部分
train_data['Fare_bin']=pd.qcut(train_data['Fare'],5)
#将离散化后的数据factorize化
train_data['Fare_bin_id']=pd.factorize(train_data['Fare_bin'])[0]
train_data['Fare_bin_id'].head()
