import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import gensim
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.spatial.distance import cosine
from sklearn.preprocessing import normalize
import gensim 


class GenderPCA:
    def __init__(self, model_path, male_terms, female_terms, male_word_pairs, female_word_pairs):
        self.model_path = model_path

        # Terms to define word pairs
        self.male_term = male_terms
        self.female_term = female_terms
        # Word pairs
        self.male_word_pairs = male_word_pairs
        self.female_word_pairs = female_word_pairs
        # Terms to compute bias 
        self.male_bias= None
        self.female_bias = None

        # Word Embedding
        self.model = None
        self.word_vectors = None
        self.vocabulary = None
        self.word_ids= None

        self.ar_gender_direction = None
        self.ar_grammar_gender_direction=None
        self.ar_semantic_gender_direction=None

        self.neutral_embedding=None
        


    def load_model(self):
        self.model = gensim.models.Word2Vec.load(self.model_path)
        self.word_vectors = self.model.wv.vectors
        self.word_vectors_normalized = normalize(self.model.wv.vectors)
        self.vocabulary = self.model.wv.index_to_key

    def do_pca(self, num_components=2):
        matrix = []
        self.ar_gender_direction=None
        for i in range(len(self.male_word_pairs)):
            male_word = self.male_word_pairs[i]
            female_word = self.female_word_pairs[i]

            if male_word in self.vocabulary and female_word in self.vocabulary:
                male_vector = self.word_vectors_normalized[self.vocabulary.index(male_word)]
                female_vector = self.word_vectors_normalized[self.vocabulary.index(female_word)]
                center = (male_vector + female_vector) / 2
                matrix.append(female_vector - center)
                matrix.append(male_vector - center)
        matrix = np.array(matrix)
        pca = PCA(n_components=num_components)
        pca.fit(matrix) 
        self.ar_gender_direction = pca.components_[0]

        plt.bar(range(num_components), pca.explained_variance_ratio_)
        plt.xlabel('Components', fontsize=20)
        plt.ylabel('Percentage of variance', fontsize=20)
        plt.margins(tight=True)
        plt.show()
    
    def cosine_similarity(self, v, u):
        v_norm = np.linalg.norm(v)
        u_norm = np.linalg.norm(u)
        similarity = v @ u / (v_norm * u_norm)
        return similarity

    def check_pairs(self):
        pairs_f=[]
        pairs_m=[]
        for i in range(0, len(self.male_word_pairs)):
            female_pair = self.female_word_pairs[i]
            male_pair=self.male_word_pairs[i]
            if (female_pair  in  self.vocabulary and male_pair in self.vocabulary):
                pairs_f.append(female_pair)
                pairs_m.append(male_pair)
        self.female_word_pairs=pairs_f
        self.male_word_pairs=pairs_m

    def check_words(self, female_words, male_words):
        female_words_modified=[]
        male_words_modified=[]
        for i in range(0, len(male_words)):
            female_term = female_words[i]
            male_term=male_words[i]
            if (female_term  in  self.vocabulary and male_term in self.vocabulary):
                female_words_modified.append(female_term)
                male_words_modified.append(male_term)
            else:
                female_term="ال"+female_term
                male_term="ال"+male_term
                #print(female_term)
                if (female_term  in  self.vocabulary and male_term in self.vocabulary):
                    female_words_modified.append(female_term)
                    male_words_modified.append(male_term)
        self.female_bias=female_words_modified
        self.male_bias=male_words_modified

    
    def check_terms(self):
        female_terms=[]
        male_terms=[]
        for i in range(0, len(self.male_term)):
            female_term = self.female_term[i]
            male_term=self.male_term[i]
            if (female_term  in  self.vocabulary and male_term in self.vocabulary):
                female_terms.append(female_term)
                male_terms.append(male_term)
            else:
                female_term="ال"+female_term
                male_term="ال"+male_term
                #print(female_term)
                if (female_term  in  self.vocabulary and male_term in self.vocabulary):
                    female_terms.append(female_term)
                    male_terms.append(male_term)
            
        self.female_term=female_terms
        self.male_term=male_terms
        
    def modify_pairs(self, female=True):
        f_modified_pairs=[]
        m_modified_pairs=[]
        for i in range(len(self.female_word_pairs)):
            male_pair_word = self.male_word_pairs[i]
            male_pair_word_vector=self.word_vectors[self.vocabulary.index(self.male_word_pairs[i])]
            female_pair_word = self.female_word_pairs[i]
            female_pair_word_vector=self.word_vectors[self.vocabulary.index(self.female_word_pairs[i])]
            count = 0
            w_count = 0

            if(female==True):
                for word in self.female_term:
                    if word in self.vocabulary:
                        w_count += 1
                        word_vector=self.word_vectors[self.vocabulary.index(word)]
                        if (self.cosine_similarity(word_vector, female_pair_word_vector) -
                                self.cosine_similarity(word_vector, male_pair_word_vector)) > 0:
                            count += 1
                print(female_pair_word, " ", male_pair_word, " ", count/w_count)
                if (count/w_count >= 0.70):
                    f_modified_pairs.append(female_pair_word)
                    m_modified_pairs.append(male_pair_word)
            else: 
                for word in self.male_term:
                    if word in self.vocabulary:
                        w_count += 1
                        word_vector=self.word_vectors[self.vocabulary.index(word)]
                        if (self.cosine_similarity(word_vector, male_pair_word_vector) -
                                self.cosine_similarity(word_vector, female_pair_word_vector)) > 0:
                            count += 1
                print(female_pair_word, " ", male_pair_word, " ", count/w_count)
                if (count/w_count>= 0.70):
                    f_modified_pairs.append(female_pair_word)
                    m_modified_pairs.append(male_pair_word)

        return f_modified_pairs, m_modified_pairs
        

    def modify_word_pairs(self):
        modified_female_pairs1, modified_male_pairs1 = self.modify_pairs(female=True)
        modified_female_pairs2, modified_male_pairs2= self.modify_pairs(female=False)
        female_word_pair = [x for x in modified_female_pairs1 if x in modified_female_pairs2]
        male_word_pair = [x for x in modified_male_pairs1 if x in modified_male_pairs2]
        if(len(female_word_pair)>1):
            self.female_word_pairs=female_word_pair
            self.male_word_pairs=male_word_pair
            
        else:
            female_word_pair=["صديقة","لها","نفسها","هي","سيدة"]
            male_word_pair=["صديق","له","نفسه","هو","سيد"]
            self.female_word_pairs=female_word_pair
            self.male_word_pairs=male_word_pair
        self.do_pca()

    def compute_bias(self, w, direction):
        return self.word_vectors_normalized[self.vocabulary.index(w)].dot(direction)

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
                print(e)
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
        ar_expanded_m_nouns, ar_expanded_f_nouns = self.load_expanded_nouns(masculine_nouns_file,feminine_nouns_file)
        ar_grammar_pair_expanded = self.create_grammar_pairs(ar_expanded_m_nouns, ar_expanded_f_nouns)

        X, y = self.prepare_data(ar_grammar_pair_expanded)

        clf_linear_svm = self.train_linear_svm(X, y)
        scores = cross_val_score(clf_linear_svm, X, y, cv=5)
        print("Accuracy: %0.4f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))

        self.ar_grammar_gender_direction = np.reshape(clf_linear_svm.coef_ / np.linalg.norm(clf_linear_svm.coef_), (300,))

        self.ar_semantic_gender_direction = self.get_SG_component(self.ar_gender_direction, self.ar_grammar_gender_direction)

    def drop(self,u, v):
        return u - v * u.dot(v) / v.dot(v)
    

    def get_SG_component_word(self):
        embedding=np.zeros((len(self.vocabulary),300))
        for word in self.vocabulary:
            word_vec = self.word_vectors_normalized[self.vocabulary.index(word)]
            embedding[self.vocabulary.index(word)] = self.drop(u=word_vec, v=self.ar_grammar_gender_direction) 
        self.word_vectors=embedding

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