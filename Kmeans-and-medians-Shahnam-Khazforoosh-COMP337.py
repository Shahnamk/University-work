"Shahnam Khazforoosh"

"Importing allowed libaries"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(122)


def kmedians(X, k):
    "Setting Centres for clusters"
    centriods = X[np.random.choice(X.shape[0], k)]    
    clusters = np.zeros(X.shape[0])
    
    while True:
        closest_cluster = clusters.copy()
        
        "Finding distances between points and clusters"
        distances = np.zeros((X.shape[0],k))
        for i in range(k):
            distances[:, i] = (abs(X - centriods[i]).sum(axis=1))
            clusters = np.argmin(distances, axis=1)
        
        "Updating centers"
        for i in range(k):
            centriods[i, :] = np.median(X[clusters == i], axis=0)

        "Breaking the loop once cloest cluster was found"
        if all(clusters == closest_cluster):
            break

    "Returning cluster array"
    return clusters

        
def kmeans(X, k):
    "Setting Centers for clusters"
    centriods = X[np.random.choice(X.shape[0], k)]    
    clusters = np.zeros(X.shape[0])
    
    while True:
        closest_cluster = clusters.copy()
        
        "Finding distances between points and clusters"
        distances = np.zeros((X.shape[0],k))
        for i in range(k):
            distances[:, i] = ((X-centriods[i])**2).sum(axis=1)**0.5
            clusters = np.argmin(distances, axis=1)
        
        "Updating centers"
        for i in range(k):
            centriods[i, :] = np.mean(X[clusters == i], axis=0)
            
        "Breaking the loop once cloest cluster was found aka converged"
        if all(clusters == closest_cluster):
            break
    "Returning the cluster array"        
    return clusters


"Importing each data set"

"DATA SECTION START"

"Animals data set goes here"
data =  (pd.read_csv(r"C:\\Users\\shahn\\OneDrive\\Desktop\\animals", sep = " ", header = None))

"Countries data set goes here"
data2 = (pd.read_csv(r"C:\\Users\\shahn\\OneDrive\\Desktop\\countries", sep = " ", header = None))

"Veggies data set goes here"
data3 = (pd.read_csv(r"C:\\Users\\shahn\\OneDrive\\Desktop\\veggies", sep = " ", header = None))

"Fruits data set goes here"
data4 = (pd.read_csv(r"C:\\Users\\shahn\\OneDrive\\Desktop\\fruits", sep = " ", header = None))

"DATA SECTION END"

a = data.iloc[:, 1:]

b = data2.iloc[:, 1:]

c = data3.iloc[:, 1:]

d = data4.iloc[:, 1:]

array_a = np.array(a)
array_b = np.array(b)
array_c = np.array(c)
array_d = np.array(d)

"Combining all data sets"
combined = np.concatenate((array_a, array_b, array_c, array_d), axis = 0)

"Running Kmeans and Kmedians for 4 clusters"
x = kmedians(combined, 4)
y = kmeans(combined, 4)

"Getting the most common cloest cluster set for each data sheet"
animals_Kmeans = np.bincount(y[:50]).argmax()
countries_Kmeans = np.bincount(y[50:212]).argmax()
veggies_Kmeans = np.bincount(y[212:270]).argmax()
fruits_Kmeans = np.bincount(y[270:327]).argmax()

print("The cluster array for Kmeans is as follows", y)
print("The cluster for animals was found to be", animals_Kmeans)
print("The cluster for countries was found to be", countries_Kmeans)
print("The cluster for veggies was found to be", veggies_Kmeans)
print("The cluster for fruits was found to be", fruits_Kmeans)

"Getting the most common cloest cluster set for each data sheet"
animals_Kmedians = np.bincount(x[:50]).argmax()
countries_Kmedians = np.bincount(x[50:211]).argmax()
veggies_Kmedians = np.bincount(x[211:270]).argmax()
fruits_Kmedians = np.bincount(x[270:327]).argmax()

print("The cluster array for Kmedians is as follows", x)
print("The cluster for animals was found to be", animals_Kmedians)
print("The cluster for countries was found to be", countries_Kmedians)
print("The cluster for veggies was found to be", veggies_Kmedians)
print("The cluster for fruits was found to be", fruits_Kmedians)

"Getting normalized combined data set"
combined_normal = combined/np.linalg.norm(combined)


"Creating arrays to store values"
precision = np.zeros(9,)
recall = np.zeros(9,)
fscore = np.zeros(9,)
precision_norm = np.zeros(9,)
recall_norm = np.zeros(9)
fscore_norm = np.zeros(9,)
k = [1,2,3,4,5,6,7,8,9]

"Determing Precision, Recall and Fscore for Unnormalized dataset" 
"for Kmeans for k clusters of 1 to 9"
for i in range(1,10):
        
    swap = kmeans(combined, i)
    
    "Calculating Animal precision and recall"
    total_animals = swap[:50]
        
    animals = np.bincount(swap[:50]).argmax()
        
    truePos_animals = np.sum(total_animals == animals)
    falsePos_animals = np.sum(swap == animals)
    falseNeg_animals = (50 - truePos_animals)
        
    animals_precision = truePos_animals/(truePos_animals+falsePos_animals)
        
    animals_recall = truePos_animals/(truePos_animals+falseNeg_animals)
    
    "Calculating Countries precision and recall"    
    total_countries = swap[50:211]
        
    countries = np.bincount(swap[50:211]).argmax()
        
    truePos_countries = np.sum(total_countries == countries)
    falsePos_countries = np.sum(swap == countries)
    falseNeg_countries = (161 - truePos_countries)
        
    countries_precision = truePos_countries/(truePos_countries+falsePos_countries)
        
    countries_recall = truePos_countries/(truePos_countries+falseNeg_countries)
    
    "Calculating Veggies precision and recall"    
    total_veggies = swap[211:270]
        
    veggies = np.bincount(swap[211:270]).argmax()
        
    truePos_veggies = np.sum(total_veggies == veggies)
    falsePos_veggies = np.sum(swap == veggies)
    falseNeg_veggies = (58 - truePos_veggies)
        
    veggies_precision = truePos_veggies/(truePos_veggies+falsePos_veggies)
        
    veggies_recall = truePos_veggies/(truePos_veggies + falseNeg_veggies)
    
    "Calculating Fruits precision and recall"    
    total_fruits = swap[270:327]
                
    fruits = np.bincount(swap[270:327]).argmax()
        
    truePos_fruits = np.sum(total_fruits == fruits)
    falsePos_fruits = np.sum(swap == fruits)
    falseNeg_fruits = (58 - truePos_fruits)
        
    fruits_precision = truePos_fruits/(truePos_fruits+falsePos_fruits)
        
    fruits_recall = truePos_fruits/(truePos_fruits+falseNeg_fruits)
    
    "Calculating overall precision, recall and its corrosponding fscore"    
    total_precision = (animals_precision + countries_precision + veggies_precision + fruits_precision)/4
    total_recall = (animals_recall + countries_recall + veggies_recall + fruits_recall)/4
    total_fscore = (2*(total_precision*total_recall))/(total_precision + total_recall)
    
    "Storing precision, recall and fscore for this cluster iteration"
    precision[i-1] = total_precision
    recall[i-1] = total_recall
    fscore[i-1] = total_fscore

"Plotting graph for Kmeans Unnormalized datatset"
plt.plot(k, precision, color = 'r', label= "precision")
plt.plot(k, recall, color = 'b', label= "recall")
plt.plot(k, fscore, color = 'g', label= "fscore")

plt.title("Not normalized graph - kmeans")
plt.xlabel("K clusters")
plt.ylabel("Percentage as a decimal")
plt.legend(loc='best')

plt.show()

"Determing Precision, Recall and Fscore for Normalized dataset" 
"for Kmeans for k clusters of 1 to 9"
for i in range(1,10):
        
    swap = kmeans(combined_normal, i)
    
    "Calculating Animal precision and recall"    
    total_animals = swap[:50]
        
    animals = np.bincount(total_animals).argmax()
    truePos_animals = np.sum(total_animals == animals)
    falsePos_animals = np.sum(swap == animals)
    falseNeg_animals = (50 - truePos_animals)
        
    animals_precision = truePos_animals/(truePos_animals+falsePos_animals)
        
    animals_recall = truePos_animals/(truePos_animals + falseNeg_animals)
    
    "Calculating Countries precision and recall"    
    total_countries = swap[50:211]
        
    countries = np.bincount(total_countries).argmax()
        
    truePos_countries = np.sum(total_countries == countries)
    falsePos_countries = np.sum(swap == countries)
    falseNeg_countries = (161 - truePos_countries)
    
    countries_precision = truePos_countries/(truePos_countries+falsePos_countries)
        
    countries_recall = truePos_countries/(truePos_countries + falseNeg_countries)

    "Calculating Veggies precision and recall"      
    total_veggies = swap[211:270]
        
    veggies = np.bincount(total_veggies).argmax()
        
    truePos_veggies = np.sum(total_veggies == veggies)
    falsePos_veggies = np.sum(swap == veggies)
    falseNeg_veggies = (58 - truePos_veggies)
        
    veggies_precision = truePos_veggies/(truePos_veggies+falsePos_veggies)
        
    veggies_recall = truePos_veggies/(truePos_veggies + falseNeg_veggies)
        
    "Calculating Fruits precision and recall"  
    total_fruits = swap[270:327]
                
    fruits = np.bincount(total_fruits).argmax()
        
    truePos_fruits = np.sum(total_fruits == fruits)
    falsePos_fruits = np.sum(swap == fruits)
    falseNeg_fruits = (58 - truePos_fruits)
        
    fruits_precision = truePos_fruits/(truePos_fruits+falsePos_fruits)
        
    fruits_recall = truePos_fruits/(truePos_fruits + falseNeg_fruits)
        
    "Calculating overall precision, recall and its corrosponding fscore"
    total_precision = (animals_precision + countries_precision + veggies_precision + fruits_precision)/4
    total_recall = (animals_recall + countries_recall + veggies_recall + fruits_recall)/4
    total_fscore = (2*(total_precision * total_recall))/(total_precision + total_recall)
    
    "Storing precision, recall and fscore for this cluster iteration"
    precision_norm[i-1] = total_precision
    recall_norm[i-1] = total_recall
    fscore_norm[i-1] = total_fscore


"Plotting graph for Kmeans Normalized datatset"
plt.plot(k, precision_norm, color = 'r', label= "precision")
plt.plot(k, recall_norm, color = 'b', label= "recall")
plt.plot(k, fscore_norm, color = 'g', label= "fscore")

plt.title("Normalized graph - kmeans")
plt.xlabel("K clusters")
plt.ylabel("Percentage as a decimal")
plt.legend(loc='best')

plt.show()

"Creating arrays to store values"
precision_medians = np.zeros(9,)
recall_medians = np.zeros(9,)
fscore_medians = np.zeros(9,)
precisionmed_norm = np.zeros(9,)
recallmed_norm = np.zeros(9)
fscoremed_norm = np.zeros(9,)

"Determing Precision, Recall and Fscore for Unnormalized dataset" 
"for Kmeadians for k clusters of 1 to 9"
for i in range(1,10):
        
    swap = kmedians(combined, i)

    "Calculating Animal precision and recall" 
    total_animals = swap[:50]
        
    animals = np.bincount(swap[:50]).argmax()
        
    truePos_animals = np.sum(total_animals == animals)
    falsePos_animals = np.sum(swap == animals)
    falseNeg_animals = (50 - truePos_animals)
    
    animals_precision = truePos_animals/(truePos_animals+falsePos_animals)
        
    animals_recall = truePos_animals/(truePos_animals+falseNeg_animals)
      
    "Calculating Countries precision and recall" 
    total_countries = swap[50:211]
        
    countries = np.bincount(swap[50:211]).argmax()
        
    truePos_countries = np.sum(total_countries == countries)
    falsePos_countries = np.sum(swap == countries)
    falseNeg_countries = (161 - truePos_countries)
        
    countries_precision = truePos_countries/(truePos_countries+falsePos_countries)
    
    countries_recall = truePos_countries/(truePos_countries+falseNeg_countries)
      
    "Calculating Veggies precision and recall" 
    total_veggies = swap[211:270]
        
    veggies = np.bincount(swap[211:270]).argmax()
        
    truePos_veggies = np.sum(total_veggies == veggies)
    falsePos_veggies = np.sum(swap == veggies)
    falseNeg_veggies = (58 - truePos_veggies)
        
    veggies_precision = truePos_veggies/(truePos_veggies+falsePos_veggies)
        
    veggies_recall = truePos_veggies/(truePos_veggies + falseNeg_veggies)
        
    "Calculating Fruits precision and recall" 
    total_fruits = swap[270:327]
                
    fruits = np.bincount(swap[270:327]).argmax()
        
    truePos_fruits = np.sum(total_fruits == fruits)
    falsePos_fruits = np.sum(swap == fruits)
    falseNeg_fruits = (58 - truePos_fruits)
        
    fruits_precision = truePos_fruits/(truePos_fruits+falsePos_fruits)
        
    fruits_recall = truePos_fruits/(truePos_fruits+falseNeg_fruits)
        
    "Calculating overall precision, recall and its corrosponding fscore"
    total_precision = (animals_precision + countries_precision + veggies_precision + fruits_precision)/4
    total_recall = (animals_recall + countries_recall + veggies_recall + fruits_recall)/4
    total_fscore = (2*(total_precision*total_recall))/(total_precision + total_recall)
    
    "Storing precision, recall and fscore for this cluster iteration"
    precision_medians[i-1] = total_precision
    recall_medians[i-1] = total_recall
    fscore_medians[i-1] = total_fscore


"Plotting graph for Kmedians Unnormalized datatset"
plt.plot(k, precision_medians, color = 'r', label= "precision")
plt.plot(k, recall_medians, color = 'b', label= "recall")
plt.plot(k, fscore_medians, color = 'g', label= "fscore")

plt.title("Not normalized graph - kmedians")
plt.xlabel("K clusters")
plt.ylabel("Percentage as a decimal")
plt.legend(loc='best')

plt.show()

"Determing Precision, Recall and Fscore for Normalized dataset" 
"for Kmedians for k clusters of 1 to 9"
for i in range(1,10):
        
    swap = kmedians(combined_normal, i)
    
    "Calculating Animals precision and recall"       
    total_animals = swap[:50]
        
    animals = np.bincount(total_animals).argmax()
    truePos_animals = np.sum(total_animals == animals)
    falsePos_animals = np.sum(swap == animals)
    falseNeg_animals = (50 - truePos_animals)
        
    animals_precision = truePos_animals/(truePos_animals+falsePos_animals)
    
    animals_recall = truePos_animals/(truePos_animals + falseNeg_animals)

    "Calculating Countries precision and recall"
    total_countries = swap[50:211]
    
    countries = np.bincount(total_countries).argmax()
        
    truePos_countries = np.sum(total_countries == countries)
    falsePos_countries = np.sum(swap == countries)
    falseNeg_countries = (161 - truePos_countries)
        
    countries_precision = truePos_countries/(truePos_countries+falsePos_countries)
        
    countries_recall = truePos_countries/(truePos_countries + falseNeg_countries)
        
    "Calculating Veggies precision and recall"
    total_veggies = swap[211:270]
        
    veggies = np.bincount(total_veggies).argmax()
        
    truePos_veggies = np.sum(total_veggies == veggies)
    falsePos_veggies = np.sum(swap == veggies)
    falseNeg_veggies = (58 - truePos_veggies)
        
    veggies_precision = truePos_veggies/(truePos_veggies+falsePos_veggies)
        
    veggies_recall = truePos_veggies/(truePos_veggies + falseNeg_veggies)

    "Calculating Fruits precision and recall"
    total_fruits = swap[270:327]
                
    fruits = np.bincount(total_fruits).argmax()
        
    truePos_fruits = np.sum(total_fruits == fruits)
    falsePos_fruits = np.sum(swap == fruits)
    falseNeg_fruits = (58 - truePos_fruits)
        
    fruits_precision = truePos_fruits/(truePos_fruits+falsePos_fruits)
        
    fruits_recall = truePos_fruits/(truePos_fruits + falseNeg_fruits)
        
    "Calculating overall precision, recall and its corrosponding fscore"
    total_precision = (animals_precision + countries_precision + veggies_precision + fruits_precision)/4
    total_recall = (animals_recall + countries_recall + veggies_recall + fruits_recall)/4
    total_fscore = (2*(total_precision * total_recall))/(total_precision + total_recall)
    
    "Storing precision, recall and fscore for this cluster iteration"
    precisionmed_norm[i-1] = total_precision
    recallmed_norm[i-1] = total_recall
    fscoremed_norm[i-1] = total_fscore


"Plotting graph for Kmedians Normalized datatset"
plt.plot(k, precisionmed_norm, color = 'r', label= "precision")
plt.plot(k, recallmed_norm, color = 'b', label= "recall")
plt.plot(k, fscoremed_norm, color = 'g', label= "fscore")

plt.title("Normalized graph - kmedians")
plt.xlabel("K clusters")
plt.ylabel("Percentage as a decimal")
plt.legend(loc='best')

plt.show()

print("Kmeans")

print("The precision for Unnormalized K means data was for each K is", precision)
print("The recall for Unnormalized K means data was for each K is", recall)
print("The Fscore for Unnormalized K means data was for each K is", fscore)
print("The precision for normalized K means data was for each K is", precision_norm)
print("The recall for normalized K means data was for each K is", recall_norm)
print("The Fscore for Unnormalized K means data was for each K is", fscore_norm)

print("Kmedian")

print("The precision for Unnormalized K medians data was for each K is", precision_medians)
print("The recall for Unnormalized K medians data was for each K is", recall_medians)
print("The Fscore for Unnormalized K medians data was for each K is", fscore_medians)
print("The precision for normalized K medians data was for each K is", precisionmed_norm)
print("The recall for normalized K meadians data was for each K is", recallmed_norm)
print("The Fscore for Unnormalized K medians data was for each K is", fscoremed_norm)
