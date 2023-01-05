import numpy as np
import time
import pandas as pd

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import fpgrowth
# 讀檔
pd.set_option('display.max_columns',10)
x = pd.read_csv('invoice-sorted.csv')
invoice = x.drop(columns=['ITEM_ID','PRODUCT_TYPE','CUST_ID','TRX_DATE','QUANTITY'])
invoice_ary = np.array(invoice)
#y = [[],[]]
tmp = []
count = 0
#path = 'frequentitemset.txt'
#f = open(path, 'w', encoding='UTF-8')
for i in range(len(invoice_ary)):
    if i+1 != len(invoice_ary):
        if invoice_ary[i,1] != invoice_ary[i+1,1] and invoice_ary[i,1] == invoice_ary[i-1,1]:
            count += 1
        if invoice_ary[i,1] != invoice_ary[i+1,1] and invoice_ary[i,1] != invoice_ary[i-1,1]:
            count += 1
    if i+1 == len(invoice_ary):
        if invoice_ary[i,1] != invoice_ary[i-1,1]:
            count += 1
y = [[]for i in range(count)]
count = 0
for i in range(len(invoice_ary)):
    if i+1 != len(invoice_ary):
        if invoice_ary[i,1] == invoice_ary[i+1,1]:
            tmp = np.append(tmp,invoice_ary[i,0])
            #print(count)
            y[count].append(invoice_ary[i,0])
            #print(tmp)
        if invoice_ary[i,1] != invoice_ary[i+1,1] and invoice_ary[i,1] == invoice_ary[i-1,1]:
            tmp = np.append(tmp,invoice_ary[i,0])
            y[count].append(invoice_ary[i,0])
            #print(count)
            #print(tmp)
            #print(tmp, file=f)
            count += 1
            tmp = []
        if invoice_ary[i,1] != invoice_ary[i+1,1] and invoice_ary[i,1] != invoice_ary[i-1,1]:
            tmp = np.append(tmp,invoice_ary[i,0])
            y[count].append(invoice_ary[i,0])
            #print(count)
            #print(tmp)
            #print(tmp,file=f)
            count += 1
            tmp = []
    if i+1 == len(invoice_ary):
        if invoice_ary[i,1] != invoice_ary[i-1,1]:
            tmp = np.append(tmp,invoice_ary[i,0])
            y[count].append(invoice_ary[i,0])
            #print(count)
            #print(tmp)
            #print(tmp,file=f)
            count += 1
            tmp = []
te = TransactionEncoder()
freq_array = te.fit(y).transform(y)
df = pd.DataFrame(freq_array,columns=te.columns_)

start = time.time()
frequent_items = apriori(df,min_support=0.002,use_colnames=True)
end = time.time()
rules = association_rules(frequent_items,metric="confidence",min_threshold=0.9)
print("Apriori:")
print(frequent_items)
print("執行時間：%f 豪秒" % ((end - start)/0.002))
print("------------------------------------------------")
print(rules)
p = np.array(rules)
#print(p,file=f)
start = time.time()
FP = fpgrowth(df,min_support=0.002,use_colnames=True)
print("FP-Growth")
print(FP)
end = time.time()
print("執行時間：%f 豪秒" % ((end - start)/0.002))