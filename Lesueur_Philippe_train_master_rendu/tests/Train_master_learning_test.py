import pandas as pd
import sys
sys.path.insert(1, '../main')
from Train_master_learning import Train_master_learning
import unittest

class Train_master_learning_test(unittest.TestCase):

    df = pd.DataFrame()
    tml = Train_master_learning()
    df["review_content"] = ["ALL CAPS NO PONCT LEN 24","no cap. two sentences, len 30.","Some Caps! all Poncts ? 4 caps : len 30; 3 Sentences, \"quote\" - tez'(par)."]
    df["review_title"] = ["TITLE1", "title2", "Title3"]
    df["review_stars"] = [1,3,5]
    
    def test_add_FirstClassif_ClassifCreated(self):
        new_df = self.df.copy()
        self.tml.add_FirstClassif(new_df)
        try:
            new_df.text_clf
            return True
        except AttributeError:
            return False
    
    def test_add_length_LengthCreated(self):
        new_df = self.df.copy()
        self.tml.add_length(new_df)
        try:
            new_df.review_length
            new_df.title_length
            return True
        except AttributeError:
            return False
    
    def test_add_NBupper_UpperCreated(self):
        new_df = self.df.copy()
        self.tml.add_NBupper(new_df)
        try:
            new_df.review_NBupper
            new_df.title_NBupper
            return True
        except AttributeError:
            return False
    
    def test_add_NBlower_LowerCreated(self):
        new_df = self.df.copy()
        self.tml.add_NBlower(new_df)
        try:
            new_df.review_NBlower
            new_df.title_NBlower
            return True
        except AttributeError:
            return False
            
    def test_add_NBponct_PonctCreated(self):
        new_df = self.df.copy()
        self.tml.add_NBponct(new_df)
        try:
            new_df.review_NBponct
            new_df.title_NBponct
            return True
        except AttributeError:
            return False
    
    def test_add_NBnumber_NumberCreated(self):
        new_df = self.df.copy()
        self.tml.add_NBnumber(new_df)
        try:
            new_df.review_NBnumber
            new_df.title_NBnumber
            return True
        except AttributeError:
            return False
    
    def test_add_NBsentences_SentencesCreated(self):
        new_df = self.df.copy()
        self.tml.add_NBsentences(new_df)
        try:
            new_df.review_NBsentences
            return True
        except AttributeError:
            return False
    
##################################################################
    def test_add_FirstClassif_ClassifNumbers(self):
        new_df = self.df.copy()
        self.tml.add_FirstClassif(new_df)
        unic = new_df.text_clf.unique()
        return all(unic == [0, 1]) or all(unic == [1, 0]) or unic == [1] or unic == [0]
    
    def test_add_length_LengthNumbers(self):
        new_df = self.df.copy()
        self.tml.add_length(new_df)
        return all(new_df.review_length == [24, 30, 74])
    
    def test_add_NBupper_UpperNumbers(self):
        new_df = self.df.copy()
        self.tml.add_NBupper(new_df)
        return all(new_df.review_NBupper == [17, 0, 4]) and all(new_df.title_NBupper == [5, 0, 1])
    
    def test_add_NBlower_LowerNumbers(self):
        new_df = self.df.copy()
        self.tml.add_NBlower(new_df)
        return all(new_df.review_NBlower ==[0, 20, 40]) and all(new_df.title_NBlower == [0, 5, 4]) 
    def test_add_NBponct_PonctNumbers(self):
        new_df = self.df.copy()
        self.tml.add_NBponct(new_df)
        return all(new_df.review_NBponct == [0, 3, 12]) and all(new_df.title_NBponct == [0, 0, 0])
    
    def test_add_NBnumber_NumberNumbers(self):
        new_df = self.df.copy()
        self.tml.add_NBnumber(new_df)
        return all(new_df.review_NBnumber == [2, 2, 4]) and all(new_df.title_NBnumber == [1, 1, 1])
    
    def test_add_NBsentences_SentencesNumbers(self):
        new_df = self.df.copy()
        self.tml.add_NBsentences(new_df)
        return all(new_df.review_NBsentences == [1, 2, 3])
    
if __name__ == '__main__':
    unittest.main(verbosity=2)
