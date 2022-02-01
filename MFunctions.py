import numpy as np
from scipy.stats import spearmanr, pearsonr
from numpy.linalg import inv
import random


#Scalar products definitions

#Product of observable vectors weighted by the mean square
def M_vec_mult(M1,M2,M_mean_square):
    return np.sum(np.divide(np.multiply(M1,M2),M_mean_square))

#Product of observable deviation vectors
def M_vec_mult_dev(M1,M2,M_mean,Std_Dev):
    return np.sum((M1-M_mean)*(M2-M_mean)/(Std_Dev*Std_Dev))

#Product of observable deviation vectors with Mahalanobis (inverse covariance used as metric tensor)
def M_vec_mult_dev_Maha(M1,M2,M_mean,inv_cov):
    return ((M1-M_mean) @ inv_cov @ (M2-M_mean))

#Plain inner product
def M_vec_mult_plain(M1,M2):
    return np.sum(M1 * M2)

#Plain inner product for observable deviation vectors
def M_vec_mult_plain_dev(M1,M2,M_mean):
    return np.sum((M1 - M_mean) * (M2 - M_mean))



#Define observables object
#observables is the class of observables considered. It has a name and requires at init a list of files from which it will load the expressions of the observables
class observables:
    
    """observables contains a list of the observables as strings.
        Initialisation requires a file name or list of file names from which to load the observables.
    
    """
    
    def __init__(self,filelist):
        self.filelist=filelist
        self.obslist=[]
        
        #if filelist is a single file, i.e. a string just load that
        if isinstance(filelist,str):
            with open(filelist,'r') as myfile:
                self.obslist=self.obslist+[a.strip() for a in myfile]
                
        #if we have a list of files load the content of the single files and join together
        elif isinstance(filelist,list) and all(isinstance(i,str) for i in filelist):
            for filename in filelist:
                with open(filename,'r') as myfile:
                    self.obslist=self.obslist+[a.strip() for a in myfile]
            
        #if we have an unknown input
        #To do: convert this into a proper error message, just because I like being fancy
        else :
            try:
                raise unknowninit("observables")
            except unknowninit as error:
                print(error.message)
            #print("Unknown")
    
    def __str__(self):
        return """Observables from files {0} stored in this object""".format(self.filelist)



#Custom error message
class unknowninit(Exception):
    """Exception raised when an unknown initialisation is given to one of my custom functions
    
    name -- is the name of the function which raised the error
    
    """
    
    def __init__(self,name):
        self.name = name
        self.message = "Unknown initialisation value for the function "+self.name+", plese check input and retry."
        super().__init__(self.message)
    
    pass




#Class vector
class vector:
    """The class vector is list of numerical values associated to observables.
        Initialisation requires a word to which it is associated, a corresponding matrix mat
        and a set of observables obs as stored in the observables class.
        N.B. This class needs to be extended to multiple matrices to handle observables built from
        multiple different matrices.
    
    """
    
    def __init__(self,word,mat,obs):
        self.filelist = obs.filelist
        self.val = evaluateobs(mat,obs.obslist)
        self.word = word
        
    def __str__(self):
        return """This vector contains information about the word {0}""".format(self.word)




#Auxiliary function for the class vector. Here we use the fact that the conventional notation for our matrices is W
def evaluateobs(W,obs):
    return np.array(list(map(eval,obs)))




#OBSOLETE, USE CLASS VECTOR_DICTIONARY
#This is the function which turns a file containing the matrices into a list of vectors.
#TO DO: make it listable, so that if you give a list of files it does it on each single one
def makevectorlist(file_name,obs_obj):
    
    file_content = open(file_name,'r')
    vector_list=[]
    
    for local in file_content:
        
        #Each line in the files corresponds to a word plus the matrix a single long list, start converting into matrix/word form
        local = local.strip().split()
        #The word is the first entry of the vector, the rest of the components gives the matrix as a single vector
        word = local[0]
        mat = np.array([float(b) for b in local[1:]])
        #Reshape the vector into a matrix
        mat_size = int(np.sqrt(len(mat)))
        mat = mat.reshape(mat_size,mat_size)
        
        #Next we feed the matrix and the word, along side an observables object into a vector object and add it to the list of vectors
        vector_list=vector_list+[vector(word,mat,obs_obj)]
        
    #close the file
    file_content.close()
    
    return vector_list





#A dictionary class containing all the the vectors associated to a given file and a given set of observables
#It takes as input a file_name from which the matrices are loaded and an observables object. The file name can be a list of two lists: the first being
#a list of files the second a list of weights, then matrices are taken from the given files with respective weights

#TO DO: WRITE FUNCTION TO MERGE DICTIONARIES, so that if I have the dictionary for Linear and that for Quadratic
# I can just merge them to get the one for linear and quadratic together

# class vector_dictionary(dict):
    
#     """vector_dictionary is a subclass of the class dictionary, where each key corresponds to a vector built
#     out of the matrices stored in the file file_name and using the observables associated to the observables
#     object obs_obj. It takes as input a file_name from which the matrices are loaded and an observables object. The file name can be a list of two lists: the first being
#     a list of files the second a list of weights, then matrices are taken from the given files with respective weights. Method deviation_vec(self,workey) returns
#     the observables deviation vector of the given key."""
    
#     def __init__(self,file_name,obs_obj):
#         self.filelist = obs_obj.filelist
#         self.fname = file_name
#         self.obslist = obs_obj.obslist
        
#         #Here we start differentiating between a single file or multiple weighted files
#         if isinstance(file_name,str):
#             #we have a string corresponding to the single file
            
#             #build the content of the dictionary
#             file_content = open(file_name,'r')
    
#             for local in file_content:
        
#                 #Each line in the files corresponds to a word plus the matrix a single long list, start converting into matrix/word form
#                 local = local.strip().split()
#                 #The word is the first entry of the vector, the rest of the components gives the matrix as a single vector
#                 word = local[0]
#                 mat = np.array([float(b) for b in local[1:]])
#                 #Reshape the vector into a matrix
#                 mat_size = int(np.sqrt(len(mat)))
#                 mat = mat.reshape(mat_size,mat_size)
        
#                 #Next we feed the matrix and the word, along side an observables object into the dictionary
#                 self.__setitem__(word,vector(word,mat,obs_obj))
        
#             #close the file
#             file_content.close()
            
#         elif isinstance(file_name,list) and len(file_name) == 2 and all(isinstance(i,str) for i in file_name[0]) and all(isinstance(i,float) or isinstance(i,int) for i in file_name[1]) :
#             #we have a list of files, better a list of files and a list of weights, loop over the files and their content
#             #open all the files
#             file_content = [open(local_file) for local_file in file_name[0]]
#             #loop using zip function
#             for local in zip(*file_content):
#                 local = [var.strip().split() for var in local]
#                 #extract the word assuming list are ordered with same word in same position
#                 word = local[0][0]
#                 #dropping the words and making matrices out of the lists
#                 mat_size = int(np.sqrt(len(local[0])-1))
                
#                 #what follows is an attempt which seems not to work due to the fact that we are nesting np.array, we would need to convert
#                 #back to lists but this is something to be avoided whenever possible so we change approach
#                 #local = np.array([list(np.array([float(var2) for var2 in var1[1:]]).reshape(mat_size,mat_size)) for var1 in local])
#                 #combine the matrices with appropriate weights
#                 #mat = np.multiply(np.array([[i] for i in file_name[1]]),local)
#                 #sum the matrices together
#                 #mat = np.sum(mat,axis=0)
                
#                 #next we convert lists to matrices, combine them with the weights and sum them up
#                 #initialise mat to matrix of zeroes
#                 mat = np.zeros((mat_size,mat_size))
                
#                 local = [list(np.array([float(var2) for var2 in var1[1:]]).reshape(mat_size,mat_size)) for var1 in local]
                
#                 for a,b in zip(local,file_name[1]):
#                     mat = mat + np.multiply(a,b)
                
#                 #Next we feed the matrix and the word, along side an observables object into the dictionary
#                 self.__setitem__(word,vector(word,mat,obs_obj))
                
#             #close the files
#             for filevar in file_content:
#                 filevar.close()
                
#         else :
#             #unknown type of input
#             print("Unknown type of input, please check and retry.")
            
        
#         #define further inrinsic properties of the dictionary like the mean, standard deviation and mean squared
#         self.mean = np.mean(np.array([ a.val for a in self.values()]), axis = 0)
#         self.std = np.std(np.array([ a.val for a in self.values()]), axis = 0)
#         self.mean_sqr = np.mean(np.array([ a.val for a in self.values()]) ** 2, axis = 0)

#     #Define method which upon call returns the observable deviation vector
#     def deviation_vec(self,wordkey):
#         return self[wordkey].val - self.mean

#Vector dictionary again, this time allowing for mixing
#A dictionary class containing all the the vectors associated to a given file and a given set of observables
#TO DO: WRITE FUNCTION TO MERGE DICTIONARIES, so that if I have the dictionary for Linear and that for Quadratic
# I can just merge them to get the one for linear and quadratic together
#Again with method which returns the obsevabledeviation vector instead of the observable vector which is immediately
#accessible from the .val of vector

class vector_dictionary(dict):
    
    """vector_dictionary is a subclass of the class dictionary, where each key corresponds to a vector built
    out of the matrices stored in the file file_name and using the observables associated to the observables
    object obs_obj. Initialisation includes"""
    
    def __init__(self,file_name,obs_obj,totrans=[]):
        self.filelist = obs_obj.filelist
        self.fname = file_name
        self.obslist = obs_obj.obslist
        self.transposed = totrans
        
        #Here we start differentiating between a single file or multiple weighted files
        if isinstance(file_name,str):
            #we have a string corresponding to the single file
            
            #build the content of the dictionary
            file_content = open(file_name,'r')
    
            for local in file_content:
        
                #Each line in the files corresponds to a word plus the matrix a single long list, start converting into matrix/word form
                local = local.strip().split()
                #The word is the first entry of the vector, the rest of the components gives the matrix as a single vector
                word = local[0]
                mat = np.array([float(b) for b in local[1:]])
                #Reshape the vector into a matrix
                mat_size = int(np.sqrt(len(mat)))
                mat = mat.reshape(mat_size,mat_size)
                #If the optional argument totrans is not empty it means that the matrix needs to be transposed
                if len(totrans) > 0:
                    mat = mat.T
        
                #Next we feed the matrix and the word, along side an observables object into the dictionary
                self.__setitem__(word,vector(word,mat,obs_obj))
        
            #close the file
            file_content.close()
            
        elif isinstance(file_name,list) and len(file_name) == 2 and all(isinstance(i,str) for i in file_name[0]) and all(isinstance(i,float) or isinstance(i,int) for i in file_name[1]) :
            #we have a list of files, better a list of files and a list of weights, loop over the files and their content
            #open all the files
            file_content = [open(local_file) for local_file in file_name[0]]
            #loop using zip function
            for local in zip(*file_content):
                local = [var.strip().split() for var in local]
                #extract the word assuming list are ordered with same word in same position
                word = local[0][0]
                #dropping the words and making matrices out of the lists
                mat_size = int(np.sqrt(len(local[0])-1))
                
                #what follows is an attempt which seems not to work due to the fact that we are nesting np.array, we would need to convert
                #back to lists but this is something to be avoided whenever possible so we change approach
                #local = np.array([list(np.array([float(var2) for var2 in var1[1:]]).reshape(mat_size,mat_size)) for var1 in local])
                #combine the matrices with appropriate weights
                #mat = np.multiply(np.array([[i] for i in file_name[1]]),local)
                #sum the matrices together
                #mat = np.sum(mat,axis=0)
                
                #next we convert lists to matrices, combine them with the weights and sum them up
                #initialise mat to matrix of zeroes
                mat = np.zeros((mat_size,mat_size))
                
                local = [np.array([float(var2) for var2 in var1[1:]]).reshape(mat_size,mat_size) for var1 in local]
                
                #Next, when adding the matrices together we need to take into account that some might need a transposition
                for ii in totrans:
                    local[ii] = local[ii].T
                
                #Add the matrices together combinning with the weights
                for a,b in zip(local,file_name[1]):
                    mat = mat + np.multiply(b,a)
                
                #Next we feed the matrix and the word, along side an observables object into the dictionary
                self.__setitem__(word,vector(word,mat,obs_obj))
                
            #close the files
            for filevar in file_content:
                filevar.close()
                
        else :
            #unknown type of input
            print("Unknown type of input, please check and retry.")
            
        
        #define further intrinsic properties of the dictionary like the mean, standard deviation, mean squared and inverse covariance
        self.mean = np.mean(np.array([ a.val for a in self.values()]), axis = 0)
        self.std = np.std(np.array([ a.val for a in self.values()]), axis = 0)
        self.mean_sqr = np.mean(np.array([ a.val for a in self.values()]) ** 2, axis = 0)
        self.inv_cov = inv(np.cov(np.array([a.val for a in self.values()]).T))
        
    #Define method which upon call returns the observable deviation vector
    def deviation_vec(self,wordkey):
        return self[wordkey].val - self.mean




#A vector dictionary which takes as input files with vectors instead of matrices. From these the observable vectors are then built 
#Every line in the input file is a string whose first element is the verb and the remaining ones are the vector 

class vector_dictionary_from_vec(dict):
    
    """vector_dictionary is a subclass of the class dictionary, where each key corresponds to a vector built
    out of the vectors stored in the file file_name and using the observables associated to the observables
    object obs_obj."""
    
    def __init__(self,file_name,obs_obj):
        self.filelist = obs_obj.filelist
        self.fname = file_name
        self.obslist = obs_obj.obslist
        
        #Here we start differentiating between a single file or multiple weighted files
        if isinstance(file_name,str):
            #we have a string corresponding to the single file
            
            #build the content of the dictionary
            file_content = open(file_name,'r')
            #Each line in the file is a string of two elements separated by a space, the word and the associated vector as a list
            
            for local in file_content:
                
                local = local.strip().split()
                #No real point in defining these vars but I just like to keep the same shape as previous code...
                word = local[0]
                mat = np.array([float(e) for e in local[1:]])
        
                #Next we feed the matrix and the word, along side an observables object into the dictionary
                self.__setitem__(word,vector(word,mat,obs_obj))
            
            #close the file
            file_content.close()
            
        #At the time of coding I am a little bit in a hurry, so since there is no immediate need for the mixing of multiple
        #input files I'll leave this for a later version of the code
        elif isinstance(file_name,list) and len(file_name) == 2 and all(isinstance(i,str) for i in file_name[0]) and all(isinstance(i,float) or isinstance(i,int) for i in file_name[1]) :
            print("This functionality has not been implemented yet...")
                
        else :
            #unknown type of input
            print("Unknown type of input, please check and retry.")
            
        
        #define further intrinsic properties of the dictionary like the mean, standard deviation and mean squared
        self.mean = np.mean(np.array([ a.val for a in self.values()]), axis = 0)
        self.std = np.std(np.array([ a.val for a in self.values()]), axis = 0)
        self.mean_sqr = np.mean(np.array([ a.val for a in self.values()]) ** 2, axis = 0)
        
    #Define method which upon call returns the observable deviation vector
    def deviation_vec(self,wordkey):
        return self[wordkey].val - self.mean




#load the verb similarity file as a dictionary where the keys are SYNONYM, NONE and so on
class verb_similarity(dict):
    
    def __init__(self,file_name):
        
        #open the file and add content to the dictionary
        with open(file_name,'r') as file_content:
            for local in file_content:
                #reshape input from file
                local = list(local.strip().split())
                local.__delitem__(2)
                local[2]=float(local[2])
                #if the given key is already present append the list of verb pair plus similarity
                if local[3] in self:
                    self[local[3]].append(local[:3])
                #else make a new entry in the disctionary
                else: self[local[3]] = [local[:3]]



#Next we want to compare the verb similarities, requires a verb_similarity class object,
# a key corresponding to an element in this class, a vector_dictionary object and a specification of the type of product
def compare_similarity(verb_sym,verb_key,vec_dict,prod_type = "mean"):
    
    given_vals = np.array(verb_sym[verb_key]).T
    #np.array converts everything to the same type so when you call the numbers you have to convert tem back to float
    computed_vals = []
    #Next we compute the the mean, mean square and standard deviation of our sample
    #verb_list = [a for a in given_vals[0]] + [a for a in given_vals[1]]
    #local_mean = np.mean(np.array([ vec_dict[a].val for a in verb_list]), axis = 0)
    #loca_std = np.std(np.array([ vec_dict[a].val for a in verb_list]), axis = 0)
    #local_mean_sqr = np.mean(np.array([ vec_dict[a].val for a in verb_list]) ** 2, axis = 0)
    
    #pick out all pairs of verbs and compute the scalar product of these based on the specified option and appropriately normalised
    if prod_type == "mean":
        for a,b in zip(given_vals[0],given_vals[1]):
            #computed_vals.append(M_vec_mult(vec_dict[a].val,vec_dict[b].val,vec_dict.mean_sqr))
            computed_vals.append(M_vec_mult(vec_dict[a].val,vec_dict[b].val,vec_dict.mean_sqr)/np.sqrt(M_vec_mult(vec_dict[a].val,vec_dict[a].val,vec_dict.mean_sqr) * M_vec_mult(vec_dict[b].val,vec_dict[b].val,vec_dict.mean_sqr)))
    elif prod_type == "std_dev":
        for a,b in zip(given_vals[0],given_vals[1]):
            computed_vals.append(M_vec_mult_dev(vec_dict[a].val,vec_dict[b].val,vec_dict.mean,vec_dict.std)/np.sqrt(M_vec_mult_dev(vec_dict[a].val,vec_dict[a].val,vec_dict.mean,vec_dict.std) * M_vec_mult_dev(vec_dict[b].val,vec_dict[b].val,vec_dict.mean,vec_dict.std)))
    elif prod_type == "plain":
        for a,b in zip(given_vals[0],given_vals[1]):
            computed_vals.append(M_vec_mult_plain(vec_dict[a].val,vec_dict[b].val) / np.sqrt(M_vec_mult_plain(vec_dict[a].val,vec_dict[a].val) * M_vec_mult_plain(vec_dict[b].val,vec_dict[b].val)))
    elif prod_type == "plain_dev":
        for a,b in zip(given_vals[0],given_vals[1]):
            computed_vals.append(M_vec_mult_plain_dev(vec_dict[a].val,vec_dict[b].val,vec_dict.mean)/np.sqrt(M_vec_mult_plain_dev(vec_dict[a].val,vec_dict[a].val,vec_dict.mean) * M_vec_mult_plain_dev(vec_dict[b].val,vec_dict[b].val,vec_dict.mean)))
    else: print("Unknown type of product!")
        
    #Return the pearson and spearman correlations along with the average scalar product
    return [pearsonr(computed_vals,np.array([float(a) for a in given_vals[2]])),spearmanr(computed_vals,np.array([float(a) for a in given_vals[2]]))]



#Since often we simply want the averaged scalar product we also write a function which simply returns that instead of the correlations

#Requires a verb_similarity class object,
# a key corresponding to an element in this class, a vector_dictionary object and a specification of the type of product
def averaged_product(verb_sym,verb_key,vec_dict,prod_type = "mean"):
    
    if isinstance(verb_key,(list,tuple)):

        given_vals = [np.array(verb_sym[a]) for a in verb_key]
        given_vals = np.concatenate(tuple(given_vals)).T
    else:
        given_vals = np.array(verb_sym[verb_key]).T

    #given_vals = np.array(verb_sym[verb_key]).T
    #np.array converts everything to the same type so when you call the numbers you have to convert tem back to float
    computed_vals = []
    #Next we compute the the mean, mean square and standard deviation of our sample
    #verb_list = [a for a in given_vals[0]] + [a for a in given_vals[1]]
    #local_mean = np.mean(np.array([ vec_dict[a].val for a in verb_list]), axis = 0)
    #loca_std = np.std(np.array([ vec_dict[a].val for a in verb_list]), axis = 0)
    #local_mean_sqr = np.mean(np.array([ vec_dict[a].val for a in verb_list]) ** 2, axis = 0)
    
    #pick out all pairs of verbs and compute the scalar product of these based on the specified option and appropriately normalised
    if prod_type == "mean":
        for a,b in zip(given_vals[0],given_vals[1]):
            #computed_vals.append(M_vec_mult(vec_dict[a].val,vec_dict[b].val,vec_dict.mean_sqr))
            computed_vals.append(M_vec_mult(vec_dict[a].val,vec_dict[b].val,vec_dict.mean_sqr)/np.sqrt(M_vec_mult(vec_dict[a].val,vec_dict[a].val,vec_dict.mean_sqr) * M_vec_mult(vec_dict[b].val,vec_dict[b].val,vec_dict.mean_sqr)))
    elif prod_type == "std_dev":
        for a,b in zip(given_vals[0],given_vals[1]):
            computed_vals.append(M_vec_mult_dev(vec_dict[a].val,vec_dict[b].val,vec_dict.mean,vec_dict.std)/np.sqrt(M_vec_mult_dev(vec_dict[a].val,vec_dict[a].val,vec_dict.mean,vec_dict.std) * M_vec_mult_dev(vec_dict[b].val,vec_dict[b].val,vec_dict.mean,vec_dict.std)))
    elif prod_type == "maha":
        for a,b in zip(given_vals[0],given_vals[1]):
            computed_vals.append(M_vec_mult_dev_Maha(vec_dict[a].val,vec_dict[b].val,vec_dict.mean,vec_dict.inv_cov)/np.sqrt(M_vec_mult_dev_Maha(vec_dict[a].val,vec_dict[a].val,vec_dict.mean,vec_dict.inv_cov) * M_vec_mult_dev_Maha(vec_dict[b].val,vec_dict[b].val,vec_dict.mean,vec_dict.inv_cov)))
    elif prod_type == "plain":
        for a,b in zip(given_vals[0],given_vals[1]):
            computed_vals.append(M_vec_mult_plain(vec_dict[a].val,vec_dict[b].val) / np.sqrt(M_vec_mult_plain(vec_dict[a].val,vec_dict[a].val) * M_vec_mult_plain(vec_dict[b].val,vec_dict[b].val)))
    elif prod_type == "plain_dev":
        for a,b in zip(given_vals[0],given_vals[1]):
            computed_vals.append(M_vec_mult_plain_dev(vec_dict[a].val,vec_dict[b].val,vec_dict.mean)/np.sqrt(M_vec_mult_plain_dev(vec_dict[a].val,vec_dict[a].val,vec_dict.mean) * M_vec_mult_plain_dev(vec_dict[b].val,vec_dict[b].val,vec_dict.mean)))
    else: print("Unknown type of product!")
        
    #Return the pearson and spearman correlations along with the average scalar product
    return [np.mean(np.array(computed_vals)),np.std(np.array(computed_vals))]


# Modified function which computes the pairs of scalar products and simply returns them as a list, works exactly like averaged_product

# def averaged_product_list(verb_sym,verb_key,vec_dict,prod_type = "mean"):
    
#     given_vals = np.array(verb_sym[verb_key]).T
#     #np.array converts everything to the same type so when you call the numbers you have to convert tem back to float
#     computed_vals = []
#     #Next we compute the the mean, mean square and standard deviation of our sample
#     #verb_list = [a for a in given_vals[0]] + [a for a in given_vals[1]]
#     #local_mean = np.mean(np.array([ vec_dict[a].val for a in verb_list]), axis = 0)
#     #loca_std = np.std(np.array([ vec_dict[a].val for a in verb_list]), axis = 0)
#     #local_mean_sqr = np.mean(np.array([ vec_dict[a].val for a in verb_list]) ** 2, axis = 0)
    
#     #pick out all pairs of verbs and compute the scalar product of these based on the specified option and appropriately normalised
#     if prod_type == "mean":
#         for a,b in zip(given_vals[0],given_vals[1]):
#             #computed_vals.append(M_vec_mult(vec_dict[a].val,vec_dict[b].val,vec_dict.mean_sqr))
#             computed_vals.append(M_vec_mult(vec_dict[a].val,vec_dict[b].val,vec_dict.mean_sqr)/np.sqrt(M_vec_mult(vec_dict[a].val,vec_dict[a].val,vec_dict.mean_sqr) * M_vec_mult(vec_dict[b].val,vec_dict[b].val,vec_dict.mean_sqr)))
#     elif prod_type == "std_dev":
#         for a,b in zip(given_vals[0],given_vals[1]):
#             computed_vals.append(M_vec_mult_dev(vec_dict[a].val,vec_dict[b].val,vec_dict.mean,vec_dict.std)/np.sqrt(M_vec_mult_dev(vec_dict[a].val,vec_dict[a].val,vec_dict.mean,vec_dict.std) * M_vec_mult_dev(vec_dict[b].val,vec_dict[b].val,vec_dict.mean,vec_dict.std)))
#     elif prod_type == "plain":
#         for a,b in zip(given_vals[0],given_vals[1]):
#             computed_vals.append(M_vec_mult_plain(vec_dict[a].val,vec_dict[b].val) / np.sqrt(M_vec_mult_plain(vec_dict[a].val,vec_dict[a].val) * M_vec_mult_plain(vec_dict[b].val,vec_dict[b].val)))
#     elif prod_type == "plain_dev":
#         for a,b in zip(given_vals[0],given_vals[1]):
#             computed_vals.append(M_vec_mult_plain_dev(vec_dict[a].val,vec_dict[b].val,vec_dict.mean)/np.sqrt(M_vec_mult_plain_dev(vec_dict[a].val,vec_dict[a].val,vec_dict.mean) * M_vec_mult_plain_dev(vec_dict[b].val,vec_dict[b].val,vec_dict.mean)))
#     else: print("Unknown type of product!")
        
#     #Return the average scalar products as a dictionary
#     return computed_vals

#Second version where output is a dictionary

def averaged_product_list(verb_sym,verb_key,vec_dict,prod_type = "mean"):
    
    if isinstance(verb_key,(list,tuple)):

        given_vals = [np.array(verb_sym[a]) for a in verb_key]
        given_vals = np.concatenate(tuple(given_vals)).T
    else:
        given_vals = np.array(verb_sym[verb_key]).T
    #np.array converts everything to the same type so when you call the numbers you have to convert tem back to float
    computed_vals = {}
    #Next we compute the the mean, mean square and standard deviation of our sample
    #verb_list = [a for a in given_vals[0]] + [a for a in given_vals[1]]
    #local_mean = np.mean(np.array([ vec_dict[a].val for a in verb_list]), axis = 0)
    #loca_std = np.std(np.array([ vec_dict[a].val for a in verb_list]), axis = 0)
    #local_mean_sqr = np.mean(np.array([ vec_dict[a].val for a in verb_list]) ** 2, axis = 0)
    
    #pick out all pairs of verbs and compute the scalar product of these based on the specified option and appropriately normalised
    if prod_type == "mean":
        for a,b in zip(given_vals[0],given_vals[1]):
            computed_vals.update({(a,b) : M_vec_mult(vec_dict[a].val,vec_dict[b].val,vec_dict.mean_sqr)/np.sqrt(M_vec_mult(vec_dict[a].val,vec_dict[a].val,vec_dict.mean_sqr) * M_vec_mult(vec_dict[b].val,vec_dict[b].val,vec_dict.mean_sqr))})
    elif prod_type == "std_dev":
        for a,b in zip(given_vals[0],given_vals[1]):
            computed_vals.update({(a,b) : M_vec_mult_dev(vec_dict[a].val,vec_dict[b].val,vec_dict.mean,vec_dict.std)/np.sqrt(M_vec_mult_dev(vec_dict[a].val,vec_dict[a].val,vec_dict.mean,vec_dict.std) * M_vec_mult_dev(vec_dict[b].val,vec_dict[b].val,vec_dict.mean,vec_dict.std))})
    elif prod_type == "maha":
        for a,b in zip(given_vals[0],given_vals[1]):
            computed_vals.update({(a,b) : M_vec_mult_dev_Maha(vec_dict[a].val,vec_dict[b].val,vec_dict.mean,vec_dict.inv_cov)/np.sqrt(M_vec_mult_dev_Maha(vec_dict[a].val,vec_dict[a].val,vec_dict.mean,vec_dict.inv_cov) * M_vec_mult_dev_Maha(vec_dict[b].val,vec_dict[b].val,vec_dict.mean,vec_dict.inv_cov))})
    elif prod_type == "plain":
        for a,b in zip(given_vals[0],given_vals[1]):
            computed_vals.update({(a,b) : M_vec_mult_plain(vec_dict[a].val,vec_dict[b].val) / np.sqrt(M_vec_mult_plain(vec_dict[a].val,vec_dict[a].val) * M_vec_mult_plain(vec_dict[b].val,vec_dict[b].val))})
    elif prod_type == "plain_dev":
        for a,b in zip(given_vals[0],given_vals[1]):
            computed_vals.update({(a,b) : M_vec_mult_plain_dev(vec_dict[a].val,vec_dict[b].val,vec_dict.mean)/np.sqrt(M_vec_mult_plain_dev(vec_dict[a].val,vec_dict[a].val,vec_dict.mean) * M_vec_mult_plain_dev(vec_dict[b].val,vec_dict[b].val,vec_dict.mean))})
    else: print("Unknown type of product!")
        
    #Return the average scalar products as a dictionary
    return computed_vals



# Functions to compute the balanced accuracy.
# Required input is the name of the vector dictionary here misleadingly called file_name, the verb_similarity object from which to extract the pairs (simdict), two lexical classes in simdict (lex1 and lex2) and the type of product to use (prod, takes the same values as in averaged_product) 

def balanced_accuracy(file_name, simdict, lex1, lex2, prod):

    '''Required input is the name of the vector dictionary here misleadingly called file_name, the verb_similarity object from which to extract the pairs (simdict), two lexical classes in simdict (lex1 and lex2) and the type of product to use (prod, takes the same values as in averaged_product) '''

    #Using observable vectors with cosine distance

    #Compute the mean and the standard deviation for the SYNONYMS
    syn1 = averaged_product(simdict,lex2,file_name,prod)

    #Compute the mean and the standard deviation for the ANTONYMS
    ant1 = averaged_product(simdict,lex1,file_name,prod)
    
    #Use the means and deviations to set the divide between synonyms and antonyms
    divide1 = ant1[0] + 4/np.pi * np.arctan(ant1[1]/syn1[1]) * (syn1[0]-ant1[0])/2
    
    # Use the earlier defined function to compute the list of scalar products of the verbs and compare with divide
    prodlist1 = averaged_product_list(simdict,lex2,file_name,prod)

    prodlist5 = averaged_product_list(simdict,lex1,file_name,prod)
    
    # Compare and count the number of synonyms we get right
    syncomparison1 = [a > divide1 for a in list(prodlist1.values())]

    # Compare and count the number of antonyms we get right
    antcomparison1 = [a < divide1 for a in list(prodlist5.values())]

    #Compute the recall
    recall1 = 1/2*(syncomparison1.count(True)/len(syncomparison1) + antcomparison1.count(True)/len(antcomparison1))
    
    #precision
    #precision1 = 1/2*(syncomparison1.count(True)/(syncomparison1.count(True)+antcomparison1.count(False))+antcomparison1.count(True)/(antcomparison1.count(True)+syncomparison1.count(False)))
    
    #f1 score
    #precision1 = 2 * precision1 * recall1 /(precision1 + recall1)
    
    return recall1


# Same function as above but with 3 labels

def balanced_accuracy3(file_name, simdict, lex1, lex2, lex3, prod):

    '''Same required input and output as balanced_accuracy, just works with 3 lexical classes instead of two.'''
    #Using observable vectors with cosine distance

    #Compute the mean and the standard deviation for the SYNONYMS
    syn1 = averaged_product(simdict,lex3,file_name,prod)
    
    #Compute the mean and the standard deviation for the NONE
    non1 = averaged_product(simdict,lex2,file_name,prod)

    #Compute the mean and the standard deviation for the ANTONYMS
    ant1 = averaged_product(simdict,lex1,file_name,prod)

    #Use the means and deviations to set the divide between synonyms and none
    divide1 = non1[0] + 4/np.pi * np.arctan(non1[1]/syn1[1]) * (syn1[0]-non1[0])/2

    #Use the means and deviations to set the divide between none and antonyms
    divide5 = ant1[0] + 4/np.pi * np.arctan(ant1[1]/non1[1]) * (non1[0]-ant1[0])/2
    
    # Use the earlier defined function to compute the list of scalar products of the verbs and compare with divide
    prodlist1 = averaged_product_list(simdict,lex3,file_name,prod)

    prodlist5 = averaged_product_list(simdict,lex1,file_name,prod)

    prodlist9 = averaged_product_list(simdict,lex2,file_name,prod)

    # Compare and count the number of synonyms we get right
    syncomparison1 = [a > divide1 for a in list(prodlist1.values())]

    # Compare and count the number of none we get right
    noncomparison1 = [a > divide5 and a < divide1 for a in list(prodlist9.values())]

    # Compare and count the number of antonyms we get right
    antcomparison1 = [a < divide5 for a in list(prodlist5.values())]

    #Compute the average precision
    precision1 = 1/3*(syncomparison1.count(True)/len(syncomparison1) + antcomparison1.count(True)/len(antcomparison1) + noncomparison1.count(True)/len(noncomparison1))

    return precision1


# Balanced accuracy taking into account subsets

def balanced_accuracy_subsets_aux(file_name, simdict, lex1, lex2, prod, myseed1, samplesize2, samplesize1):
    #Since here we use just a subset of the pairs of synonyms and antonyms to determine the means and deviation, we have to do this by hand

    #Compute the products for all the observable deviation vectors associated to the SYNONYMS pairs
    all1 = averaged_product_list(simdict,lex2,file_name,prod)

    #The subset from which we copute the means is randomly selected, we seed the random generator for repeatable results. Chosen seed 7
    random.seed(myseed1)

    mysample = random.sample(list(all1),samplesize1)

    #Find complementary sample where we will predict and test
    mycomplsample = set(all1.keys()) - set(mysample)

    #Compute the means and standard deviations
    syn1 = [np.mean(np.array(list(map(all1.get,mysample)))),np.std(np.array(list(map(all1.get,mysample))))]

    #Now the same thing for the antonyms

    #Compute the products for all the observable deviation vectors associated to the SYNONYMS pairs
    all5 = averaged_product_list(simdict,lex1,file_name,prod)

    mysample2 = random.sample(list(all5),samplesize2)

    #Find complementary sample where we will predict and test
    mycomplsample2 = set(all5.keys()) - set(mysample2)

    #Compute the means and standard deviations
    ant1 = [np.mean(np.array(list(map(all5.get,mysample2)))),np.std(np.array(list(map(all5.get,mysample2))))]

    #Use the means and deviations to set the divide between synonyms and antonyms
    divide1 = ant1[0] + 4/np.pi * np.arctan(ant1[1]/syn1[1]) * (syn1[0]-ant1[0])/2
    
    # Next we test on the complementary sample
    # Compare and count the number of synonyms we get right
    syncomparison1 = [a > divide1 for a in list(map(all1.get,mycomplsample))]

    # Compare and count the number of antonyms we get right
    antcomparison1 = [a < divide1 for a in list(map(all5.get,mycomplsample2))]

    #Compute the average precision
    precision1 = 1/2*(syncomparison1.count(True)/len(syncomparison1) + antcomparison1.count(True)/len(antcomparison1))

    return precision1


# Next the external wrapper which does the same thing 20 times and averages

def balanced_accuracy_subsets(file_name, simdict, lex1, lex2, prod, myseed1, samplesize1, samplesize2):

    #seed the random generator before genearting the integer seeds for the random samples
    random.seed(myseed1)

    seed_list = [(random.randint(0,5000),random.randint(0,5000)) for _ in range(20)]

    return (lambda x1 : np.array([np.mean(x1,axis = 0),np.std(x1,axis=0)]).T)([balanced_accuracy_subsets_aux(file_name, simdict, lex1, lex2, prod, a, samplesize1, samplesize2) for a in seed_list ])


# Again balanced accuracy taking subsest to train and then test on complementary subset, but for 3 lexical classes

def balanced_accuracy_subsets3_aux(file_name, simdict, lex1, lex2, lex3, prod, myseed1, samplesize1, samplesize2, samplesize3):
    
    #Since here we use just a subset of the pairs of synonyms and antonyms to determine the means and deviation, we have to do this by hand

    #Compute the products for all the observable deviation vectors associated to the SYNONYMS pairs
    all1 = averaged_product_list(simdict,lex3,file_name,prod)

    #The subset from which we copute the means is randomly selected, we seed the random generator for repeatable results. Chosen seed 7
    random.seed(myseed1)

    mysample = random.sample(list(all1),samplesize3)

    #Find complementary sample where we will predict and test
    mycomplsample = set(all1.keys()) - set(mysample)

    #Compute the means and standard deviations
    syn1 = [np.mean(np.array(list(map(all1.get,mysample)))),np.std(np.array(list(map(all1.get,mysample))))]

    #Now the same thing for the antonyms

    #Compute the products for all the observable deviation vectors associated to the SYNONYMS pairs
    all5 = averaged_product_list(simdict,lex1,file_name,prod)


    mysample2 = random.sample(list(all5),samplesize1)

    #Find complementary sample where we will predict and test
    mycomplsample2 = set(all5.keys()) - set(mysample2)

    #Compute the means and standard deviations
    ant1 = [np.mean(np.array(list(map(all5.get,mysample2)))),np.std(np.array(list(map(all5.get,mysample2))))]
    
    #Finally the same for None

    #Compute the products for all the observable deviation vectors associated to the SYNONYMS pairs
    all9 = averaged_product_list(simdict,lex2,file_name,prod)


    mysample3 = random.sample(list(all9),samplesize2)

    #Find complementary sample where we will predict and test
    mycomplsample3 = set(all9.keys()) - set(mysample3)

    #Compute the means and standard deviations
    non1 = [np.mean(np.array(list(map(all9.get,mysample3)))),np.std(np.array(list(map(all9.get,mysample3))))]
    
    #Use the means and deviations to set the divide between synonyms and none
    divide1 = non1[0] + 4/np.pi * np.arctan(non1[1]/syn1[1]) * (syn1[0]-non1[0])/2

    #Use the means and deviations to set the divide between none and antonyms
    divide5 = ant1[0] + 4/np.pi * np.arctan(ant1[1]/non1[1]) * (non1[0]-ant1[0])/2
    
    # Next we test on the complementary sample
    # Compare and count the number of synonyms we get right
    syncomparison1 = [a > divide1 for a in list(map(all1.get,mycomplsample))]


    # Compare and count the number of none we get right
    noncomparison1 = [a > divide5 and a < divide1 for a in list(map(all9.get,mycomplsample3))]
    
    # Compare and count the number of antonyms we get right
    antcomparison1 = [a < divide5 for a in list(map(all5.get,mycomplsample2))]
    
    #Compute the average precision
    precision1 = 1/3*(syncomparison1.count(True)/len(syncomparison1) + antcomparison1.count(True)/len(antcomparison1) + noncomparison1.count(True)/len(noncomparison1))
    
    return precision1


# Next the external wrapper which does the same thing 20 times and averages

def balanced_accuracy_subsets3(file_name, simdict, lex1, lex2, lex3, prod, myseed1, samplesize1, samplesize2, samplesize3):

    #seed the random generator before genearting the integer seeds for the random samples
    random.seed(myseed1)

    seed_list = [(random.randint(0,5000),random.randint(0,5000)) for a in range(20)]

    return (lambda x1 : np.array([np.mean(x1,axis = 0),np.std(x1,axis=0)]).T)([balanced_accuracy_subsets3_aux(file_name, simdict, lex1, lex2, lex3, prod, a, samplesize1, samplesize2, samplesize3) for a in seed_list ])