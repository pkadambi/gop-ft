import pandas as pd

ENGLISH_PHONEME_LIST = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D',
       'DH', 'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY',
        'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S',
       'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z']

VOWELS = []
PLOSIVES = []
FRICATIVES = []
APPROXIMANTS = []


PHONEME_INFO_DF = pd.DataFrame([{'phoneme': 'AA', 'type': 'vowel', 'voiced': 'voiced', },
                                       {'phoneme': 'AE', 'type': 'vowel', 'voiced': 'voiced', },
                                       {'phoneme': 'AH', 'type': 'vowel', 'voiced': 'voiced', },
                                       {'phoneme': 'AO', 'type': 'vowel', 'voiced': 'voiced', },
                                       {'phoneme': 'AW', 'type': 'vowel', 'voiced': 'voiced', },
                                       {'phoneme': 'AY', 'type': 'vowel', 'voiced': 'voiced', },
                                       {'phoneme': 'EH', 'type': 'vowel', 'voiced': 'voiced', },
                                       {'phoneme': 'ER', 'type': 'vowel', 'voiced': 'voiced', },
                                       {'phoneme': 'EY', 'type': 'vowel', 'voiced': 'voiced', },
                                       # {'phoneme': 'I', 'type': 'vowel', 'voiced': 'voiced', },
                                       {'phoneme': 'IH', 'type': 'vowel', 'voiced': 'voiced', },
                                       {'phoneme': 'IY', 'type': 'vowel', 'voiced': 'voiced', },
                                       {'phoneme': 'OW', 'type': 'vowel', 'voiced': 'voiced', },
                                       {'phoneme': 'OY', 'type': 'vowel', 'voiced': 'voiced', },
                                       {'phoneme': 'UH', 'type': 'vowel', 'voiced': 'voiced', },
                                       {'phoneme': 'UW', 'type': 'vowel', 'voiced': 'voiced', },
                                       {'phoneme': 'B', 'type': 'bilabial plosive stop', 'voiced': 'voiced', },
                                       {'phoneme': 'CH', 'type': 'alveolar affricate fricative', 'voiced': 'unvoiced', },
                                       {'phoneme': 'D', 'type': 'alveolar plosive', 'voiced': 'voiced', },
                                       {'phoneme': 'DH', 'type': 'biabial plosive', 'voiced': 'voiced', },
                                       {'phoneme': 'F', 'type': 'labiodental fricative', 'voiced': 'unvoiced', },
                                       {'phoneme': 'G', 'type': 'velar plosive', 'voiced': 'voiced', },
                                       {'phoneme': 'HH', 'type': 'gottal fricative', 'voiced': 'unvoiced', },
                                       {'phoneme': 'JH', 'type': 'alveolar fricative', 'voiced': 'voiced', },
                                       {'phoneme': 'K', 'type': 'velar plosive stop', 'voiced': 'unvoiced', },
                                       {'phoneme': 'L', 'type': 'lateral approximant', 'voiced': 'voiced', },
                                       {'phoneme': 'M', 'type': 'bilabial nasal', 'voiced': 'voiced', },
                                       {'phoneme': 'N', 'type': 'alveolar nasal', 'voiced': 'voiced', },
                                       {'phoneme': 'NG', 'type': 'velar nasal', 'voiced': 'voiced', },
                                       {'phoneme': 'P', 'type': 'bilabial plosive stop', 'voiced': 'unvoiced', },
                                       {'phoneme': 'R', 'type': 'retroflex approximant', 'voiced': 'voiced', },
                                       {'phoneme': 'S', 'type': 'alveolar fricative', 'voiced': 'unvoiced', },
                                       {'phoneme': 'SH', 'type': 'palatoalveolar fricative', 'voiced': 'unvoiced', },
                                       {'phoneme': 'T', 'type': 'alveolar flap stop', 'voiced': 'unvoiced', },
                                       {'phoneme': 'TH', 'type': 'dental fricative', 'voiced': 'unvoiced', },
                                       {'phoneme': 'V', 'type': 'labiodental fricative', 'voiced': 'voiced', },
                                       {'phoneme': 'W', 'type': 'labiovelar approximant', 'voiced': 'voiced', },
                                       {'phoneme': 'Y', 'type': 'palatal sonorant vowel', 'voiced': 'voiced', },
                                       {'phoneme': 'Z', 'type': 'alveolar fricative', 'voiced': 'voiced', },
                                       ])