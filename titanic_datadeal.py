# -*- coding: utf-8 -*-
"""
Created on Thu May  3 14:32:12 2018

@author: Wizza
"""

import pandas as pd
import re
import numpy as np

train_df_org=pd.read_csv(r'C:\Users\Wizza\Documents\Python Scripts\titanic\train.csv')
test_df_org=pd.read_csv(r'C:\Users\Wizza\Documents\Python Scripts\titanic\test.csv')
test_df_org['Survived']=0
combined_train_test=train_df_org.append(test_df_org)
PassengerId=test_df_org['PassengerId']

#处理Embarked
#使用factorize
combined_train_test['Embarked'].fillna(combined_train_test['Embarked'].mode().iloc[0],inplace=True)
combined_train_test['Embarked']=pd.factorize(combined_train_test['Embarked'])[0]
#使用独热编码one-hot get_dummies
emb_dummies_df=pd.get_dummies(combined_train_test['Embarked'],prefix=combined_train_test[['Embarked']].columns[0])
combined_train_test=pd.concat([combined_train_test,emb_dummies_df],axis=1)

#处理Sex
#使用one-hot编码
combined_train_test['Sex']=pd.factorize(combined_train_test['Sex'])[0]
sex_dummies_df=pd.get_dummies(combined_train_test['Sex'],prefix=combined_train_test[['Sex']].columns[0])
combined_train_test=pd.concat([combined_train_test,sex_dummies_df],axis=1)

#处理Name
combined_train_test['Title']=combined_train_test['Name'].map(lambda x: re.compile(", (.*?)\.").findall(x)[0])

#对称呼进行统一化处理
title_Dict={}
title_Dict.update(dict.fromkeys(['Capt','Col','Major','Dr','Rev'],'Officer'))
title_Dict.update(dict.fromkeys(['Don','Sir','the Countess','Dona','Lady'],'Royalty'))
title_Dict.update(dict.fromkeys(['Mme','Ms','Mrs'],'Mrs'))
title_Dict.update(dict.fromkeys(['Mlle','Miss'],'Miss'))
title_Dict.update(dict.fromkeys(['Mr'],'Mr'))
title_Dict.update(dict.fromkeys(['Master','Jonkheer'],'Master'))

combined_train_test['Title']=combined_train_test['Title'].map(title_Dict)
#将称呼进行编码
combined_train_test['Title']=pd.factorize(combined_train_test['Title'])[0]
title_dummies_df=pd.get_dummies(combined_train_test['Title'],prefix=combined_train_test[['Title']].columns[0])
combined_train_test=pd.concat([combined_train_test,title_dummies_df],axis=1)
combined_train_test['name_length']=combined_train_test['Name'].apply(len)

#处理Fare
#填写空值
combined_train_test['Fare']=combined_train_test[['Fare']].fillna(combined_train_test.groupby('Pclass').transform(np.mean))
#团体票转为单人费用
combined_train_test['Group_Ticket']=combined_train_test['Fare'].groupby(by=combined_train_test['Ticket']).transform('count')
combined_train_test['Fare']=combined_train_test['Fare']/combined_train_test['Group_Ticket']
combined_train_test.drop(['Group_Ticket'],axis=1,inplace=True)
#print(combined_train_test[['Ticket','Fare']])
#使用pd.qcut给票价分等级
combined_train_test['Fare_bin']=pd.qcut(combined_train_test['Fare'],5)
#使用factorize和get_dummies对Fare进行编码
combined_train_test['Fare_bin_id']=pd.factorize(combined_train_test['Fare_bin'])[0]
fare_bin_dummies_df=pd.get_dummies(combined_train_test['Fare_bin_id']).rename(columns=lambda x:'Fare_'+str(x))
combined_train_test=pd.concat([combined_train_test,fare_bin_dummies_df],axis=1)
combined_train_test.drop(['Fare_bin'],axis=1,inplace=True)


#等舱中价位不同与逃生顺序有关

from sklearn.preprocessing import LabelEncoder

def pclass_fare_category(df,pclass1_mean_fare,pclass2_mean_fare,pclass3_mean_fare):
    if df['Pclass']==1:
        if df['Fare']<=pclass1_mean_fare:
            return 'Pclass1_Low'
        else:
            return 'Pclass1_High'
    elif df['Pclass'==2]:
        if df['Fare']<=pclass2_mean_fare:
            return 'Pclass2_Low'
        else:
            return 'Pclass2_High'
    elif df['Pclass'==3]:
        if df['Fare']<=pclass3_mean_fare:
            return 'Pclass3_Low'
        else:
            return 'Pclass3_High'

Pclass1_mean_fare=combined_train_test['Fare'].groupby(by=combined_train_test['Pclass']).mean().iloc[0]
Pclass2_mean_fare=combined_train_test['Fare'].groupby(by=combined_train_test['Pclass']).mean().iloc[1]
Pclass3_mean_fare=combined_train_test['Fare'].groupby(by=combined_train_test['Pclass']).mean().iloc[2]
combined_train_test['Pclass_Fare_Category']=combined_train_test.apply(pclass_fare_category,args=(
        Pclass1_mean_fare,Pclass2_mean_fare,Pclass3_mean_fare),axis=1)

pclass_level=LabelEncoder()
pclass_level.fit(np.array(['Pclass1_Low','Pclass1_High','Pclass2_Low','Pclass2_High',
                           'Pclass3_Low','Pclass3_High']))
combined_train_test['Pclass_Fare_Category']=pclass_level.transform(combined_train_test['Pclass_Fare_Category'])
#dummy转换
pclass_dummies_df=pd.get_dummies(combined_train_test['Pclass_Fare_Category']).rename(columns=lambda x:'Pclass_'+str(x))
combined_train_test=pd.concat([combined_train_test,pclass_dummies_df],axis=1)
#将Pclass factorize化
combined_train_test['Pclass']['Pclass']=pd.factorize(combined_train_test['Pclass'])[0]

#处理Parch 和 Sibsp
#将亲戚和父母进行合并为FamilySize
def family_size_category(family_size):
    if family_size<=1:
        return 'Single'
    elif family_size<=4:
        return 'Small_family'
    else:
        return 'Large_family'
    
combined_train_test['Family_Size']=combined_train_test['Parch']+combined_train_test['SibSp']
combined_train_test['family_size_category']=combined_train_test['Family_Size'].map(family_size_category)

le_family=LabelEncoder()
le_family.fit(['Single','Small_family','Large_family'])
combined_train_test['family_size_category']=le_family.transform(combined_train_test['family_size_category'])
family_size_dummies_df=pd.get_dummies(combined_train_test['family_size_category'],
                                      prefix=combined_train_test[['family_size_category']].columns[0])
combined_train_test=pd.concat([combined_train_test,family_size_dummies_df],axis=1)

#处理Age
#通过机器学习填补Age缺失值
missing_age_df=pd.DataFrame(combined_train_test[['Age','Embarked','Sex','Title','name_length','Family_Size',
                                                 'family_size_category','Fare','Fare_bin_id','Pclass']])
missing_age_train=missing_age_df[missing_age_df['Age'].notnull()]
missing_age_test=missing_age_df[missing_age_df['Age'].isnull()]
missing_age_test.head()

#建立Age的预测模型
from sklearn import ensemble
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

def fill_missing_age(missing_age_train,missing_age_test):
    missing_age_x_train=missing_age_train.drop(['Age'],axis=1)
    missing_age_y_train=missing_age_train['Age']
    missing_age_x_test=missing_age_test.drop(['Age'],axis=1)
    print(missing_age_x_train.columns)
    print(missing_age_x_test.columns)
    #模型1 GradienBoostingRegressor
    gbm_reg=GradientBoostingRegressor(random_state=42)
    gbm_reg_param_grid={'n_estimators':[2000],'max_depth':[4],'learning_rate':[0.01],
                        'max_features':[3]}
    gbm_reg_grid=model_selection.GridSearchCV(gbm_reg,gbm_reg_param_grid,cv=10,
                                              n_jobs=25,verbose=1,scoring='neg_mean_squared_error')
    gbm_reg_grid.fit(missing_age_x_train,missing_age_y_train)
    print('Age feature Best GB Params:'+str(gbm_reg_grid.best_params_))
    print('Age feature Best GB Score:'+str(gbm_reg_grid.best_score_))
    print('GB Train Error for age feature regressor:'+str(gbm_reg_grid.score(missing_age_x_train,missing_age_y_train)))
    missing_age_test.loc[:,'Age_gb']=gbm_reg_grid.predict(missing_age_x_test)
    print(missing_age_test['Age_gb'][:4])

    #模型2 使用随机森林回归 RandomForestRegressor 
    rf_reg=RandomForestRegressor()
    rf_reg_param_grid={'n_estimators':[200],'max_depth':[5],'random_state':[0]}
    rf_reg_grid=model_selection.GridSearchCV(rf_reg,rf_reg_param_grid,cv=10,n_jobs=25,
                                             verbose=1,scoring='neg_mean_squared_error')
    rf_reg_grid.fit(missing_age_x_train,missing_age_y_train)
    print('Age feature Best RF Params:'+str(rf_reg_grid.best_params_))
    print('Age feature Best RF Score:'+str(rf_reg_grid.best_score_))
    print('RF Train Error for age feature regressor:'+str(rf_reg_grid.score(missing_age_x_train,missing_age_y_train)))
    missing_age_test.loc[:,'Age_rf']=rf_reg_grid.predict(missing_age_x_test)
    print(missing_age_test['Age_rf'][:4])
    
    #模型1 和模型2合并
    #missing_age_test.loc[:,'Age']=(missing_age_test.loc[:,'Age_gb']+missing_age_test.loc[:,'Age_rf'])/2
    missing_age_test.loc[:,'Age']=np.mean([missing_age_test.loc[:,'Age_gb'],missing_age_test.loc[:,'Age_rf']])
    missing_age_test.drop(['Age_gb','Age_rf'],axis=1,inplace=True)
    print(missing_age_test['Age'][:4])
    return missing_age_test

combined_train_test.loc[combined_train_test.Age.isnull(),'Age']=fill_missing_age(missing_age_train,missing_age_test)

#将Ticket字母和数字分开，各为一类
combined_train_test['Ticket_Letter']=combined_train_test['Ticket'].str.split().str[0]
combined_train_test['Ticket_Letter']=combined_train_test['Ticket_Letter'].apply(lambda x:'u0' if x.isnumeric() else x)
combined_train_test['Ticket_Letter']=pd.factorize(combined_train_test['Ticket_Letter'])[0]

#Cabin分为有值和空值
combined_train_test.loc[combined_train_test.Cabin.isnull(),'Cabin']='u0'
combined_train_test['Cabin']=combined_train_test['Cabin'].apply(lambda x:0 if x=='u0' else 1)
Cabin_grouped=combined_train_test.groupby('Cabin').count()
Cabin_grouped['Survived'].plot.bar()

#特征值间相关性分析,Pearson相关性
import matplotlib.pyplot as plt
import seaborn as sns

Correlation=pd.DataFrame(combined_train_test[['Embarked', 'Sex', 'Title', 'name_length', 'Family_Size', 
                                              'family_size_category','Fare', 'Fare_bin_id', 'Pclass', 
                                              'Pclass_Fare_Category', 'Age', 'Ticket_Letter', 'Cabin']])
colormap=plt.cm.viridis
plt.figure(figsize=(14,12))
plt.title('Pearson correlation of features',y=1.05,size=15)
sns.heatmap(Correlation.astype(float).corr(),linewidths=0.1,vmax=1.0,square=True,cmap=colormap,
            linecolor='white',annot=True)

#特征之间的数据分布图
g = sns.pairplot(combined_train_test[[u'Survived',u'Pclass',u'Sex',u'Age',u'Fare',u'Embarked',
                                      u'Family_Size',u'Title',u'Ticket_Letter']],hue='Survived',palette='seismic',size=1.2,diag_kind='kde',diag_kws=dict(shade=True),plot_kws=dict(s=10))
g.set(xticklabels=[])

#对一些数据正则化处理
from sklearn import preprocessing
scale_age_fare=preprocessing.StandardScaler().fit(combined_train_test[['Age','Fare','name_length']])
combined_train_test[['Age','Fare','name_length']]=scale_age_fare.transform(combined_train_test[['Age','Fare','name_length']])
#print(combined_train_test[['Age','Fare','name_length']])

#去掉无用特征
#combined_data_back=combined_train_test
print(combined_train_test.columns)
combined_train_test.drop(['PassengerId','Embarked','Sex','Name','Title','Fare_bin_id','Pclass_Fare_Category',
                           'Parch','SibSp','family_size_category','Ticket'],axis=1,inplace=True)
#将训练集和测试集分开
train_data=combined_train_test[:891]
test_data=combined_train_test[891:]

titanic_train_data_x=train_data.drop('Survived',axis=1)
titanic_train_data_y=train_data['Survived']
titanic_test_data_x=test_data.drop('Survived',axis=1)

#模型融合
#利用不同的模型来对特征进行筛选，选出较为重要的特征
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

def get_top_n_features(titanic_train_data_x,titanic_train_data_y,top_n_features):
    
    #random forest(随机森林)
    rf_est=RandomForestClassifier(random_state=0)
    rf_param_grid={'n_estimators':[500],'min_samples_split':[2,3],'max_depth':[20]}
    rf_grid=model_selection.GridSearchCV(rf_est,rf_param_grid,n_jobs=25,cv=10,verbose=1)
    rf_grid.fit(titanic_train_data_x,titanic_train_data_y)
    print('Top N features best RF Params:'+str(rf_grid.best_params_))
    print('Top N features best RF Score:'+str(rf_grid.best_score_))
    print('Top N features train RF Params:'+str(rf_grid.score(titanic_train_data_x,titanic_train_data_y)))
    feature_imp_sorted_rf=pd.DataFrame({'feature':list(titanic_train_data_x),
                                        'importance':rf_grid.best_estimator_.feature_importances_}).sort_values('importance',ascending=False)
    features_top_n_rf=feature_imp_sorted_rf.head(top_n_features)['feature']
    print('sample 10 features from RF classifier')
    print(str(features_top_n_rf[:10]))
    
    #AdoBoost
    ada_est=AdaBoostClassifier()
    ada_param_grid={'n_estimators':[500],'learning_rate':[0.01,0.1]}
    ada_grid=model_selection.GridSearchCV(ada_est,ada_param_grid,n_jobs=25,cv=10,verbose=1)
    ada_grid.fit(titanic_train_data_x,titanic_train_data_y)
    print('Top N features best ADB Params:'+str(ada_grid.best_params_))
    print('Top N features best ADB Score:'+str(ada_grid.best_score_))
    print('Top N features train ADB Params:'+str(ada_grid.score(titanic_train_data_x,titanic_train_data_y)))
    feature_imp_sorted_ada=pd.DataFrame({'feature':list(titanic_train_data_x),
                                        'importance':ada_grid.best_estimator_.feature_importances_}).sort_values('importance',ascending=False)
    features_top_n_ada=feature_imp_sorted_ada.head(top_n_features)['feature']
    print('sample 10 features from ada classifier')
    print(str(features_top_n_ada[:10]))
    
    #ExtraTree
    et_est=ExtraTreesClassifier()
    et_param_grid={'n_estimators':[500],'min_samples_split':[3,4],'max_depth':[20]}
    et_grid=model_selection.GridSearchCV(et_est,et_param_grid,n_jobs=25,cv=10,verbose=1)
    et_grid.fit(titanic_train_data_x,titanic_train_data_y)
    print('Top N features best ET Params:'+str(et_grid.best_params_))
    print('Top N features best ET Score:'+str(et_grid.best_score_))
    print('Top N features train ET Params:'+str(et_grid.score(titanic_train_data_x,titanic_train_data_y)))
    feature_imp_sorted_et=pd.DataFrame({'feature':list(titanic_train_data_x),
                                        'importance':et_grid.best_estimator_.feature_importances_}).sort_values('importance',ascending=False)
    features_top_n_et=feature_imp_sorted_et.head(top_n_features)['feature']
    print('sample 10 features from ET classifier')
    print(str(features_top_n_et[:10]))
    
    #GradientBoosting
    gb_est=GradientBoostingClassifier()
    gb_param_grid={'n_estimators':[500],'learning_rate':[0.01,0.1],'max_depth':[20]}
    gb_grid=model_selection.GridSearchCV(gb_est,gb_param_grid,n_jobs=25,cv=10,verbose=1)
    gb_grid.fit(titanic_train_data_x,titanic_train_data_y)
    print('Top N features best GB Params:'+str(gb_grid.best_params_))
    print('Top N features best GB Score:'+str(gb_grid.best_score_))
    print('Top N features train GB Params:'+str(gb_grid.score(titanic_train_data_x,titanic_train_data_y)))
    feature_imp_sorted_gb=pd.DataFrame({'feature':list(titanic_train_data_x),
                                        'importance':gb_grid.best_estimator_.feature_importances_}).sort_values('importance',ascending=False)
    features_top_n_gb=feature_imp_sorted_gb.head(top_n_features)['feature']
    print('sample 10 features from GB classifier')
    print(str(features_top_n_gb[:10]))
    
    #DicisionTree
    dt_est=DecisionTreeClassifier()
    dt_param_grid={'min_samples_split':[2,4],'max_depth':[20]}
    dt_grid=model_selection.GridSearchCV(dt_est,dt_param_grid,n_jobs=25,cv=10,verbose=1)
    dt_grid.fit(titanic_train_data_x,titanic_train_data_y)
    print('Top N features best DT Params:'+str(dt_grid.best_params_))
    print('Top N features best DT Score:'+str(dt_grid.best_score_))
    print('Top N features train DT Params:'+str(dt_grid.score(titanic_train_data_x,titanic_train_data_y)))
    feature_imp_sorted_dt=pd.DataFrame({'feature':list(titanic_train_data_x),
                                        'importance':dt_grid.best_estimator_.feature_importances_}).sort_values('importance',ascending=False)
    features_top_n_dt=feature_imp_sorted_dt.head(top_n_features)['feature']
    print('sample 10 features from DT classifier')
    print(str(features_top_n_dt[:10]))   
    
    #模型结果合并
    features_top_n=pd.concat([features_top_n_rf,features_top_n_ada,features_top_n_et,
                              features_top_n_gb,features_top_n_dt],ignore_index=True).drop_duplicates()
    features_importance=pd.concat([feature_imp_sorted_rf,feature_imp_sorted_ada,feature_imp_sorted_et,
                                   feature_imp_sorted_gb,feature_imp_sorted_dt],ignore_index=True)
    
    return features_top_n, features_importance

feature_top_n,features_importance=get_top_n_features(titanic_train_data_x,titanic_train_data_y,30)
titanic_train_data_x=pd.DataFrame(titanic_train_data_x[feature_top_n])
titanic_test_data_x=titanic_test_data_x[feature_top_n]

#对筛选的特征进行视图可视化
rf_feature_imp=features_importance[:10]
ada_feature_imp=features_importance[30:30+10].reset_index(drop=True)

#除以最大值来衡量特征的相对重要性
rf_feature_importance=100*(rf_feature_imp['importance']/rf_feature_imp['importance'].max())
ada_feature_importance=100*(ada_feature_imp['importance']/ada_feature_imp['importance'].max())
rf_importance_idx=np.where(rf_feature_importance)[0]
ada_importance_idx=np.where(ada_feature_importance)[0]
pos=np.arange(rf_importance_idx.shape[0])+0.5
plt.figure(1,figsize=(18,8))
plt.subplot(121)
plt.barh(pos,rf_feature_importance[rf_importance_idx][::-1])
plt.yticks(pos,rf_feature_imp['feature'][::-1])
plt.xlabel('Relative importance')
plt.title('RandomForest feature importance')

plt.subplot(122)
plt.barh(pos,ada_feature_importance[::-1])
plt.yticks(pos,ada_feature_imp['feature'][::-1])
plt.xlabel('Relative importance')
plt.title('AdaBoost feature importance')
plt.show()

#采用stacking方法，第一层模型使用KFold方法
from sklearn.model_selection import KFold

ntrain=titanic_train_data_x.shape[0]
ntest=titanic_test_data_x.shape[0]
seed=0
nfolds=7
kf=KFold(n_splits=nfolds,random_state=seed,shuffle=False)

def get_out_fold(clf,x_train,y_train,x_test):
    oof_train=np.zeros(ntrain)
    oof_test=np.zeros(ntest)
    oof_test_skf=np.empty((nfolds,ntest))
    
    for i, (train_index,test_index) in enumerate(kf.split(x_train)):
        x_tr=x_train[train_index]
        y_tr=y_train[train_index]
        x_te=x_train[test_index]
        clf.fit(x_tr,y_tr)
        oof_train[test_index]=clf.predict(x_te)
        oof_test_skf[i,:]=clf.predict(x_test)
    
    oof_test[:]=oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1,1),oof_test.reshape(-1,1)

#构建不同的基学习器，使用RandomForest、AdaBoost、ExtraTrees、GBDT、DecisionTree、KNN、SVM七种基学习器
    
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

rf=RandomForestClassifier(n_estimators=500,warm_start=True,max_features='sqrt',max_depth=6,min_samples_split=3,
                          min_samples_leaf=2,n_jobs=1,verbose=0)
ada=AdaBoostClassifier(n_estimators=500,learning_rate=0.1)

et=ExtraTreesClassifier(n_estimators=500,n_jobs=-1,max_depth=8,min_samples_leaf=2,verbose=0)

gb=GradientBoostingClassifier(n_estimators=500,learning_rate=0.008,min_samples_split=3,min_samples_leaf=2,
                              max_depth=5,verbose=0)

dt=DecisionTreeClassifier(max_depth=8)

knn=KNeighborsClassifier(n_neighbors=2)

svm=SVC(kernel='linear',C=0.025)

#将训练数据从Pandas转为arrays
x_train=titanic_train_data_x.values
y_train=titanic_train_data_y.values
x_test=titanic_test_data_x.values

#对每一个基学习器进行构建和预测
rf_oof_train,rf_oof_test=get_out_fold(rf,x_train,y_train,x_test)
ada_oof_train,ada_oof_test=get_out_fold(ada,x_train,y_train,x_test)
et_oof_train,et_oof_test=get_out_fold(et,x_train,y_train,x_test)
gb_oof_train,gb_oof_test=get_out_fold(gb,x_train,y_train,x_test)
dt_oof_train,dt_oof_test=get_out_fold(dt,x_train,y_train,x_test)
knn_oof_train,knn_oof_test=get_out_fold(knn,x_train,y_train,x_test)
svm_oof_train,svm_oof_test=get_out_fold(svm,x_train,y_train,x_test)
print('training is complete')

x_train=np.concatenate((rf_oof_train,ada_oof_train,et_oof_train,gb_oof_train,
                        dt_oof_train,knn_oof_train,svm_oof_train),axis=1)
x_test=np.concatenate((rf_oof_test,ada_oof_test,et_oof_test,gb_oof_test,dt_oof_test,
                        knn_oof_test,svm_oof_test),axis=1)

#第二层学习，利用XGBoost,使用第一层预测的结果作为特征对最终的结果进行预测
from xgboost import XGBClassifier

gbm=XGBClassifier(n_estimators=2000,max_depth=4,min_child_weight=2,gamma=0.9,subsample=0.8,colsample_bytree=0.8,
                  object='binary:logistic',nthread=-1,scale_pos_weigth=1).fit(x_train,y_train)
predictions=gbm.predict(x_test)

stackingsubmission=pd.DataFrame({'PassengersId':PassengerId,'Survived':predictions})
stackingsubmission.to_csv('stacksubmission.csv',index=False,sep=',')


#构建学习曲线
from sklearn.learning_curve import learning_curve

def plot_learning_curve(estimator,title,x,y,ylim=None,cv=None,n_jobs=1,train_sizes=np.linspace(0.1,1,5),verbose=0):
    plt.figure()
    plt.title('title')
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel('training examples')
    plt.ylabel('score')
    train_sizes,train_scores,test_scores=learning_curve(estimator,x,y,cv=cv,n_jobs=n_jobs,train_sizes=train_sizes)
    train_scores_mean=np.mean(train_scores,axis=1)
    train_scores_std=np.std(train_scores,axis=1)
    test_scores_mean=np.mean(test_scores,axis=1)
    test_scores_std=np.std(test_scores,axis=1)
    plt.grid()
    
    plt.fill_between(train_sizes,train_scores_mean-train_scores_std,
                     train_scores_mean+train_scores_std,alpha=0.1,color='r')
    plt.fill_between(train_sizes,test_scores_mean-test_scores_std,
                     test_scores_mean+test_scores_std,alpha=0.1,color='r')
    plt.plot(train_sizes,train_scores_mean,'o-',color='r',label='training score')
    plt.plot(train_sizes,test_scores_mean,'o-',color='g',label='cross-validation score')
    plt.legend(loc='best')
    return plt

#逐一观察不同模型的学习曲线
rf_parameters = {'n_jobs': -1, 'n_estimators': 500, 'warm_start': True, 'max_depth': 6, 'min_samples_leaf': 2, 
                 'max_features' : 'sqrt','verbose': 0}
# AdaBoost
ada_parameters = {'n_estimators':500, 'learning_rate':0.1}
# ExtraTrees
et_parameters = {'n_jobs': -1, 'n_estimators':500, 'max_depth': 8, 'min_samples_leaf': 2, 'verbose': 0}
# GradientBoosting
gb_parameters = {'n_estimators': 500, 'max_depth': 5, 'min_samples_leaf': 2, 'verbose': 0}
# DecisionTree
dt_parameters = {'max_depth':8}
# KNeighbors
knn_parameters = {'n_neighbors':2}
# SVM
svm_parameters = {'kernel':'linear', 'C':0.025}
# XGB
gbm_parameters = {'n_estimators': 2000, 'max_depth': 4, 'min_child_weight': 2, 'gamma':0.9, 'subsample':0.8, 
                  'colsample_bytree':0.8, 'objective': 'binary:logistic', 'nthread':-1, 'scale_pos_weight':1}

title='learning curve'
plot_learning_curve(RandomForestClassifier(**rf_parameters),title,x_train,y_train,cv=None,n_jobs=4,
                    train_sizes=np.linspace(50,500,10).astype(int))
plt.show()


