//（1）导入文件并将所有特征转换为float形式
#加载数据  
def loadCSV(filename):  
    dataSet=[]  
    with open(filename,'r') as file:  
        csvReader=csv.reader(file)  
        for line in csvReader:  
            dataSet.append(line)  
    return dataSet  
  
#除了判别列，其他列都转换为float类型  
def column_to_float(dataSet):  
    featLen=len(dataSet[0])-1  
    for data in dataSet:  
        for column in range(featLen):  
            data[column]=float(data[column].strip())  
//（2）    将数据集分成n份，方便交叉验证

#将数据集分成N块，方便交叉验证  
def spiltDataSet(dataSet,n_folds):  
    fold_size=int(len(dataSet)/n_folds)  
    dataSet_copy=list(dataSet)  
    dataSet_spilt=[]  
    for i in range(n_folds):  
        fold=[]  
        while len(fold) < fold_size:   #这里不能用if，if只是在第一次判断时起作用，while执行循环，直到条件不成立  
            index=randrange(len(dataSet_copy))  
            fold.append(dataSet_copy.pop(index))  #pop() 函数用于移除列表中的一个元素（默认最后一个元素），并且返回该元素的值。  
        dataSet_spilt.append(fold)  
    return dataSet_spilt  

//（3）    构造数据子集（随机采样），并在指定特征个数（假设m个，手动调参）下选取最优特征

#构造数据子集  
def get_subsample(dataSet,ratio):  
    subdataSet=[]  
    lenSubdata=round(len(dataSet)*ratio)  
    while len(subdataSet) < lenSubdata:  
        index=randrange(len(dataSet)-1)  
        subdataSet.append(dataSet[index])  
    #print len(subdataSet)  
    return subdataSet  
  
#选取任意的n个特征，在这n个特征中，选取分割时的最优特征  
def get_best_spilt(dataSet,n_features):  
    features=[]  
    class_values=list(set(row[-1] for row in dataSet))  
    b_index,b_value,b_loss,b_left,b_right=999,999,999,None,None  
    while len(features) < n_features:  
        index=randrange(len(dataSet[0])-1)  
        if index not in features:  
            features.append(index)  
    #print 'features:',features  
    for index in features:  
        for row in dataSet:  
            left,right=data_spilt(dataSet,index,row[index])  
            loss=spilt_loss(left,right,class_values)  
            if loss < b_loss:  
                b_index,b_value,b_loss,b_left,b_right=index,row[index],loss,left,right  
    #print b_loss  
    #print type(b_index)  
    return {'index':b_index,'value':b_value,'left':b_left,'right':b_right}  

//（4）    构造决策树

#构造决策树  
def build_tree(dataSet,n_features,max_depth,min_size):  
    root=get_best_spilt(dataSet,n_features)  
    sub_spilt(root,n_features,max_depth,min_size,1)   
    return root  
//（5）    创建随机森林（多个决策树的结合）

#创建随机森林  
def random_forest(train,test,ratio,n_feature,max_depth,min_size,n_trees):  
    trees=[]  
    for i in range(n_trees):  
        train=get_subsample(train,ratio)  
        tree=build_tree(train,n_features,max_depth,min_size)  
        #print 'tree %d: '%i,tree  
        trees.append(tree)  
    #predict_values = [predict(trees,row) for row in test]  
    predict_values = [bagging_predict(trees, row) for row in test]  
    return predict_values  

//（6）    输入测试集并进行测试，输出预测结果

#预测测试集结果  
def predict(tree,row):  
    predictions=[]  
    if row[tree['index']] < tree['value']:  
        if isinstance(tree['left'],dict):  
            return predict(tree['left'],row)  
        else:  
            return tree['left']  
    else:  
        if isinstance(tree['right'],dict):  
            return predict(tree['right'],row)  
        else:  
            return tree['right']  
   # predictions=set(predictions)  
