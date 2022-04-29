"""
Skeleton file for the project 1 of the LINGI2364 course.
Use this as your submission file. Every piece of code that is used in your program should be put inside this file.

This file given to you as a skeleton for your implementation of the Apriori and Depth
First Search algorithms. You are not obligated to use them and are free to write any class or method as long as the
following requirements are respected:

Your apriori and alternativeMiner methods must take as parameters a string corresponding to the path to a valid
dataset file and a double corresponding to the minimum frequency.
You must write on the standard output (use the print() method) all the itemsets that are frequent in the dataset file
according to the minimum frequency given. Each itemset has to be printed on one line following the format:
[<item 1>, <item 2>, ... <item k>] (<frequency>).
Tip: you can use Arrays.toString(int[] a) to print an itemset.

The items in an itemset must be printed in lexicographical order. However, the itemsets themselves can be printed in
any order.

Do not change the signature of the apriori and alternative_miner methods as they will be called by the test script.

__authors__ = "<write here your group, first name(s) and last name(s)>"
"""

from datetime import datetime
import numpy as np 
import time 

class Dataset:
	"""Utility class to manage a dataset stored in a external file."""

	def __init__(self, filepath):
		"""reads the dataset file and initializes files"""
		self.transactions = list()
		self.items = set()

		try:
			lines = [line.strip() for line in open(filepath, "r")]
			lines = [line for line in lines if line]  # Skipping blank lines
			for line in lines:
				transaction = list(map(int, line.split(" ")))
				self.transactions.append(transaction)
				for item in transaction:
					self.items.add(item)
		except IOError as e:
			print("Unable to read dataset file!\n" + e)

	def trans_num(self):
		"""Returns the number of transactions in the dataset"""
		return len(self.transactions)

	def items_num(self):
		"""Returns the number of different items in the dataset"""
		return len(self.items)

	def get_transaction(self, i):
		"""Returns the transaction at index i as an int array"""
		return self.transactions[i]

	def max_item(self):
		return max(self.items)

	def all_items(self):
		return [i for i in self.items]

# combine itemsets that are identical except for last symbol
def combine_items(itemset1, itemset2):
	# stop the comparaison of two itemsets earlier when they become long
	if len(itemset1) > 3:
		if itemset1[0] != itemset2[0]:
			return 0
		if itemset1[1] != itemset2[1]:
			return 0
		if itemset1[2] != itemset2[2]:
			return 0

	if itemset1[:-1] == itemset2[:-1]:
		try: 
			tmp1 = int(itemset1[-1])
			tmp2 = int(itemset2[-1])
			if tmp1 >= tmp2:
				ret = itemset2 + [itemset1[-1]]
				return ret
			else:
				ret = itemset1 + [itemset2[-1]]
				return ret
			
		except TypeError:
			print('must be integers')

	else:
		#print('strings must be identical except for last symbol')
		return 0

# gen candidates 
def gen_candidates(itemset):
	ret = []
	for i in range(len(itemset)):
		c = i + 1
		while(c < len(itemset)):
			if len(itemset[i]) > 1:
				tmp = combine_items(itemset[i],itemset[c])
				if tmp != 0:
					ret.append(tmp)
			else:
				tmp = itemset[i] + itemset[c]
				ret.append(tmp)
			c += 1
	return ret

def apriori(filepath, minFrequency):
	# each row represents an item
	# each column represents a transaction

	start = time.time()
	data = Dataset(filepath)
	nb_trans = data.trans_num()
	items = data.items_num()
	max_item = data.max_item()

	# create matrix with a number of rows as the max item value, need that if the sequence of items values is not complete
	matrix = np.zeros((max_item,nb_trans))
	min_sup = nb_trans * minFrequency

	# add item index for each row
	index = np.array([[i] for i in range(1,max_item+1)])
	matrix = np.concatenate((matrix,index),axis=1)

	# fill matrix with 1 for each item in a transaction
	for i in range(nb_trans):
		for j in data.get_transaction(i):
			matrix[j-1][i] = 1
		
	# remove rows which total number of 1 is less than min_sup
	elem = 0
	supports = np.array([])
	while elem < matrix.shape[0]:
		tmp = np.sum(matrix[elem][:-1])
		if tmp < min_sup:
			matrix = np.delete(matrix,elem,0)
		else:
			supports = np.append(supports,tmp)
			elem += 1

	# add the support of each item at the end of its row
	supports = np.array([[i] for i in supports])
	matrix = np.concatenate((matrix,supports),axis=1)

	f = {j:[] for j in range(items+1)}
	c = {j:[] for j in range(items+1)}

	s = 1
	while True:
		if s == 1:
			f[s] = [[int(k[0])] for k in matrix[:,-2:-1]]
			for n in matrix[:,-2:]:
				print([int(n[0])],f"({n[1]/nb_trans})")
			
		else: 
			c[s] = gen_candidates(f[s-1])
			for el in c[s]:
				mask = np.isin(matrix[:,-2],el)

				# filter the view of the dataset with the current itemset that is checked
				mat_tmp = matrix[mask]

				# logical AND between rows of the itemset
				mat_tmp = mat_tmp[:,:-2].all(axis=0)
				freq = len(mat_tmp[mat_tmp[:]])/nb_trans
				if freq >= minFrequency:
					print(el,f"({freq})")
					f[s].append(el)

		# stop if no more frequent itemset
		if not(len(f[s])>0):
			break

		s += 1
	
	end = time.time()
	t = round(end - start,3) 	
	print(f"Finished in {t:.3f} seconde(s)")
	return t


class Eclat :
    
    def __init__(self, minFrequency, total_trans):
        self.minFrequency = minFrequency
        self.total_trans = total_trans
        self.FreqItemsets = dict()
    
    def get_children(self, itemsets, tids):
        children = []
        for test_itemset, test_tids in itemsets:
            intersect_tids = tids & test_tids
            intersectFreq = self.get_frequency(len(intersect_tids))
            if intersectFreq >= self.minFrequency:
                children.append((test_itemset,intersect_tids))
        return children
    
    def get_frequency(self,num_trans):
        return num_trans/self.total_trans
    
    def runner(self, frequent_itemset, data):
        while data:
            current_itemset, current_tids = data.pop()
            currentFreq = self.get_frequency(len(current_tids)) 
            if currentFreq >= self.minFrequency:
                print(sorted(frequent_itemset + [current_itemset]), (currentFreq))
                self.FreqItemsets[tuple(frequent_itemset + [current_itemset])] = currentFreq
                children = self.get_children(data, current_tids)
                self.runner(frequent_itemset + [current_itemset], sorted(children, key=lambda item: len(item[1]), reverse=True))
    
    def get_frequent_itemsets(self):
        return self.FreqItemsets
    
    
def to_vertical(data):
    data2vert = {}
    for j in range(len(data)):
        for item in data[j]:
            if item not in data2vert:
                data2vert[item] = set()
            data2vert[item].add(j+1)
    return data2vert


def alternative_miner(filepath, minFrequency):
    """Runs the alternative frequent itemset mining algorithm on the specified file with the given minimum frequency"""
    start = time.time()
    data = Dataset(filepath)
    num_trans = data.trans_num()
    data = [data.get_transaction(i) for i in range(num_trans)]
    data = to_vertical(data)
    data = sorted(data.items(), key=lambda item: len(item[1]), reverse=True)
    frequent_itemset = []
    eclat = Eclat(minFrequency, num_trans)
    eclat.runner(frequent_itemset, data)
    end = time.time()
    t = end - start
    return t

print(apriori('./Datasets/accidents.dat',0.9))



	

