import warnings
import numpy as np
#from scipy.stats import permutation_test
from mlxtend.evaluate import permutation_test
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.spatial.distance import cosine
from sklearn.preprocessing import normalize
import gensim 
from collections import Counter
import csv 
class weat:
    def __init__(self, model_path, attribute1, attribute2, target1, target2, year):
        self.model_path = model_path
        self.attribute1 = attribute1
        self.attribute2 = attribute2
        self.target1 = target1
        self.target2 = target2
        self.year=year
        #self.model = None

        # Word Embedding
        self.model = None
        self.word_vectors = None
        self.vocabulary = None
        self.word_ids= None

        self.ar_grammar_gender_direction=None

    #def load_model(self):
    #    self.model = gensim.models.Word2Vec.load(self.model_path)

    def load_model(self):
        self.model = gensim.models.Word2Vec.load(self.model_path)
        self.word_vectors = self.model.wv.vectors
        self.word_vectors_normalized = normalize(self.model.wv.vectors)
        self.vocabulary = self.model.wv.index_to_key
        
    def cosine_similarity(self, v, u):
        v_norm = np.linalg.norm(v)
        u_norm = np.linalg.norm(u)
        similarity = v @ u / (v_norm * u_norm)
        return similarity
    

    
    def cosine_similarities_by_words(self, word, words):

        vec =[ self.word_vectors_normalized[self.vocabulary.index(w)] for w in [word] if w in self.vocabulary]
         
        vecs = [self.word_vectors_normalized[self.vocabulary.index(w)] for w in words if w in self.vocabulary]
        #try:
        #print(np.mean([self.cosine_similarity(vec, vec2)  for vec2 in vecs]))
        return [self.cosine_similarity(vec, vec2)  for vec2 in vecs]
        #except Exception as e:
            #print(f'word {word} not found')
    
    def _calc_association_target_attributes(self, target_word, first_attribute_words, second_attribute_words):

        try:
            first_mean = np.mean(self.cosine_similarities_by_words(
                                                   target_word,
                                                   first_attribute_words))
                      #.mean())

            second_mean = np.mean(self.cosine_similarities_by_words(
                                                    target_word,
                                                    second_attribute_words))
                       #.mean())
            #print(first_mean)
            #print(second_mean)
            

            return (first_mean) - (second_mean)
        
        except Exception as e:
            print(f'word {target_word} not found')



    
    def _calc_association_all_targets_attributes(self, target_words, first_attribute_words, second_attribute_words):
        return [self._calc_association_target_attributes(target_word, first_attribute_words, second_attribute_words)
                for target_word in target_words]

    def _calc_weat_score(self, first_target_words, second_target_words, first_attribute_words, second_attribute_words):
        first_associations = self._calc_association_all_targets_attributes(first_target_words, first_attribute_words, second_attribute_words)
        second_associations = self._calc_association_all_targets_attributes(second_target_words, first_attribute_words, second_attribute_words)
        return sum(first_associations) - sum(second_associations)

    def _calc_weat_pvalue(self, first_associations, second_associations, method='approximate'):
        RANDOM_STATE = 42
        pvalue = permutation_test(first_associations, second_associations,
                            func=lambda x, y: sum(x) - sum(y),
                            method=method,
                            num_rounds=1000,
                            seed=RANDOM_STATE)  # if exact - no meaning
        #print(first_associations)
        #print(pvalue)
        return pvalue


    def calculate_average_word_frequency(self, words):
        total_frequency = 0
        count = 0
        for word in words:
            #arabic_word = get_display(arabic_reshaper.reshape(word))  # Convert back to Arabic for model evaluation
            if word in self.model.wv.key_to_index:
                word_count = self.model.wv.get_vecattr(word, 'count')
                total_frequency += word_count
                count += 1

        if count == 0:
            return 0  # Avoid division by zero

        average_frequency = total_frequency / count
        return average_frequency
    def calc_single_weat(self, with_pvalue=True):
        #self.load_model()
        if (len(self.attribute1)>len(self.attribute2)):
            self.attribute1=self.attribute1[:len(self.attribute2)]
        else:
            self.attribute2=self.attribute2[:len(self.attribute1)]
        first_associations = self._calc_association_all_targets_attributes(self.target1, self.attribute1, self.attribute2)
        #print(first_associations)
        first_associations=  [x for x in first_associations if x is not None]
        
        second_associations = self._calc_association_all_targets_attributes(self.target2, self.attribute1, self.attribute2)
        #print(second_associations)
        second_associations=  [x for x in second_associations if x is not None]
        
        if (len(second_associations)>len(first_associations)):
            second_associations=second_associations[:len(first_associations)]
        else:
            first_associations=first_associations[:len(second_associations)]
        
        #print(first_associations)
        #print(second_associations)
        if first_associations and second_associations:
            score = sum(first_associations) - sum(second_associations)
            std_dev = np.std(first_associations + second_associations, ddof=0)
            effect_size = (np.mean(first_associations) - np.mean(second_associations)) / std_dev
            
            pvalue = None
            if with_pvalue:
                pvalue = self._calc_weat_pvalue(first_associations, second_associations)

        else:
            score, std_dev, effect_size, pvalue = None, None, None, None

        return {
            's': score,
            'd': effect_size,
            'p': pvalue,
            'year':self.year
        }
    
    def load_nouns(self, filename):
        nouns = []
        with open(filename, "r", encoding="utf-8") as f:
            for line in f: 
                word = line.strip()
                if word not in nouns and word in self.vocabulary:
                    nouns.append(word)
        return nouns

    def load_expanded_nouns(self,masculine_nouns_file,feminine_nouns_file):
        ar_expanded_m_nouns = self.load_nouns(masculine_nouns_file)
        ar_expanded_f_nouns = self.load_nouns(feminine_nouns_file)
        return ar_expanded_m_nouns, ar_expanded_f_nouns

    def create_grammar_pairs(self, m_nouns, f_nouns):
        grammar_pairs = []
        for f, m in zip(f_nouns, m_nouns):
            pair = [f, m]
            grammar_pairs.append(pair)
        # Check the length of ar_grammar_pair_expanded
        if len(grammar_pairs) % 2 != 0:
            # If the length is odd, subtract 1 to make it even
            grammar_pairs = grammar_pairs[:-1]
        return grammar_pairs

    def shorten_pairs(self, grammar_pairs, length):
        shortened = grammar_pairs[:length]
        return shortened

    def prepare_data(self, grammar_pairs):
        X = np.zeros((len(grammar_pairs) * 2, 300))
        y = np.tile([1, 2], len(grammar_pairs))
        counter = 0
        for pair in grammar_pairs:
            try:
                X[counter] = self.word_vectors_normalized[self.vocabulary.index(pair[0])]
                counter += 1
                X[counter] = self.word_vectors_normalized[self.vocabulary.index(pair[1])]
                counter += 1
                #print(counter)
            except Exception as e:
                #print(e)
                continue
        return X, y

    def train_linear_svm(self, X, y):
        clf = LinearSVC()
        clf.fit(X, y)
        return clf

    def train_linear_discriminant_analysis(self, X, y):
        clf = LinearDiscriminantAnalysis(n_components=1)
        clf.fit(X, y)
        return clf
    
    # Get semantic gender component for a vector by removing the grammatical gender component
    def get_SG_component(self,vector, ggd):
        return vector - (vector.dot(ggd)*ggd/ggd.dot(ggd))


    def get_grammatical_direction(self,masculine_nouns_file, feminine_nouns_file ):
        self.load_model()
        ar_expanded_m_nouns, ar_expanded_f_nouns = self.load_expanded_nouns(masculine_nouns_file,feminine_nouns_file)
        ar_grammar_pair_expanded = self.create_grammar_pairs(ar_expanded_m_nouns, ar_expanded_f_nouns)

        X, y = self.prepare_data(ar_grammar_pair_expanded)

        clf_linear_svm = self.train_linear_svm(X, y)
        scores = cross_val_score(clf_linear_svm, X, y, cv=5)
        print("Accuracy: %0.4f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))

        self.ar_grammar_gender_direction = np.reshape(clf_linear_svm.coef_ / np.linalg.norm(clf_linear_svm.coef_), (300,))

        embedding=np.zeros((len(self.vocabulary),300))
        for word in self.vocabulary:
            word_vec = self.word_vectors_normalized[self.vocabulary.index(word)]
            embedding[self.vocabulary.index(word)] = self.drop(u=word_vec, v=self.ar_grammar_gender_direction) 
        self.word_vectors_normalized = normalize(embedding)
        


    def drop(self,u, v):
        return u - v * u.dot(v) / v.dot(v)


    def sc_weat(self,w):
        """Compute the association of a target word 'w' with attribute sets A and B."""
        sum_similarity_A = 0
        sum_similarity_B = 0

        for a in self.male_word_pairs:
            
            similarity_A = self.cosine_similarity(self.word_vectors_normalized[self.vocabulary.index(w)],self.word_vectors_normalized[self.vocabulary.index(a)])
            sum_similarity_A += similarity_A

        for b in self.female_word_pairs:
            similarity_B = self.cosine_similarity(self.word_vectors_normalized[self.vocabulary.index(w)],self.word_vectors_normalized[self.vocabulary.index(b)])
            sum_similarity_B += similarity_B

        s_wAB = sum_similarity_A/len(self.male_word_pairs) - sum_similarity_B/len(self.female_word_pairs)
        return s_wAB

