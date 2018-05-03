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
combined_train_test['Fare'].fillena(combined_train_test.groupby('Pclass').transform(np.mean))
print(pclass_grouped)