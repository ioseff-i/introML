import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint
import random

df = pd.read_csv("Iris.csv")
df = df.drop("Id",axis=1)
df = df.rename(columns={"Species":"label"})

random.seed(0)
def train_test_split(df,test_size):
    if(isinstance(test_size,float)):
        test_size = round(test_size*len(df))
    indices = df.index.tolist()
    test_indices = random.sample(population=indices,k=test_size)
    test_df = df.loc[test_indices]
    training_df = df.drop(test_indices)

    return training_df,test_df
train , test = train_test_split(df,test_size = 0.2)

data = train.values

def check_purity(data):
    labels = np.unique(data[:,-1])
    if(len(labels)==1):
        return True
    return False

def classify(data):
    labels = data[:,-1]
    labels,counts = np.unique(labels,return_counts=True)
    index=counts.argmax()
    return labels[index]

def get_potential_splits(data):
    potential_splits = {}
    _,n_columns = data.shape
    for column_index in range(n_columns - 1):
        potential_splits[column_index] = []
        values = data[:,column_index]
        unique_values = np.unique(values)

        for index in range(len(unique_values)):
            if index!= 0:
                current_value = unique_values[index] 
                previous_value = unique_values[index-1]
                potential_split = (current_value+previous_value)/2

                potential_splits[column_index].append(potential_split)

    return potential_splits

def split_data(data,split_column,split_value):
    split_column_values = data[:,split_column]
    data_below=data[split_column_values <= split_value]
    data_above=data[split_column_values > split_value]

    return data_below,data_above

def calculate_entropy(data):
    label_column = data[:,-1]
    _,counts = np.unique(label_column,return_counts = True)
    probabilites = counts/counts.sum()
    entropy = sum(probabilites*-np.log2(probabilites))

    return entropy

def calculate_overall_entropy(data_below,data_above):
    n = len(data_below) + len(data_above)
    p_below = len(data_below)/n
    p_above = len(data_above)/n

    overall_entropy = (p_below*calculate_entropy(data_below)+p_above*calculate_entropy(data_above))
    return overall_entropy

def determine_best_split(data,potential_splits):
    overall_entropy = 999
    for index in potential_splits:
        for value in potential_splits[index]:
            data_below,data_above = split_data(data,split_column = index,split_value = value)
            current_overall_entropy = calculate_overall_entropy(data_below,data_above)

            if current_overall_entropy <= overall_entropy:
                overall_entropy = current_overall_entropy
                best_split_column = index
                best_split_value = value
    return best_split_column, best_split_value

def decision_tree_algorithm(df,counter=0,min_samples=2,max_depth = 5):
    if counter == 0:
        global COLUMN_HEADERS
        COLUMN_HEADERS = df.columns
        data = df.values
    else:
        data = df

    # induction case

    if(check_purity(data)) or (len(data) < min_samples) or (counter == max_depth) :
        classification = classify(data)
        return classification
    else:
        counter += 1

        potential_splits = get_potential_splits(data)
        split_column,split_value = determine_best_split(data,potential_splits)
        data_below,data_above=split_data(data, split_column,split_value)

        #instantiate sub-tree
        feature_name = COLUMN_HEADERS[split_column]
        
        question = "{} <= {}".format(feature_name,split_value)
        sub_tree = {question:[]}

        #Find answers
        yes_answer = decision_tree_algorithm(data_below,counter)
        no_answer = decision_tree_algorithm(data_above,counter)

        if(yes_answer == no_answer):
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)
            

        

        return sub_tree

# tree = decision_tree_algorithm(df)
# pprint(tree)

def classify_example(example,tree):
    question = list(tree.keys())[0]
    feature_name,comparison_operator,value = question.split()

    if example[feature_name] <= float(value):
        answer = tree[question][0]

    else:
        answer = tree[question][1]

    if not isinstance(answer,dict):
        return answer
    else:
        residual_tree = answer
        return classify_example(example,residual_tree)


def calculate_accuracy(df,tree):
    df["classified"] = df.apply(classify_example,axis=1,args=(tree,))
    df["bet"] = df["classified"] == df["label"]

    accuracy = df["bet"].mean()

    return accuracy
    
        
            
     
     
    
    
    



# Testing of get_potential_splits method 
# potential_splits = get_potential_splits(data)
# sns.lmplot(data=train,x="PetalWidthCm",y="PetalLengthCm",hue="label",fit_reg=False,height=6,aspect=1.5)
# plt.vlinesvalues(data,3,0.6)
# plotting_df=pd.DataFrame(data_above,columns=df.columns)
# sns.lmplot(data=plotting_df,x="PetalWidthCm",y="PetalLengthCm",hue="label",fit_reg=False,height=6,aspect=1.5)
# plt.vlines(x=0.8,ymin=1,ymax=7)
# plt.show()


    

    
    
