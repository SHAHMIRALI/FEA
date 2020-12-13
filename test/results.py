from common.constants import EMOTION_MAP, EMOTION_KEY_MAP

# Control test result data and its summary
class Result:
    TOTAL_EMOTIONS = 7
    _COL_0_SIZE = 10
    _COL_2_SIZE = 8

    get_col_1_size = lambda x: max(len('hits'), len(str(x)))

    def __init__(self, emotion):
        self.emotion = emotion
        self.total = -1
        self.predictions = [0 for _ in range(Result.TOTAL_EMOTIONS)]
    
    def add(self, prediction):
        self.total += 1
        self.predictions[EMOTION_KEY_MAP[prediction]] += 1
    
    def summarize(self):
        # Title
        fat_hrule(20)
        print(f"Test results: {self.emotion}")
        fat_hrule()

        # Summary of whole test for this emotion
        print("SUMMARY")
        hrule()
        print(f"Total tests:\t{self.total}\nCorrect:\t{self.predictions[EMOTION_KEY_MAP[self.emotion]]}")
        print("Accuracy:\t%.2f%%" % (100 * self.predictions[EMOTION_KEY_MAP[self.emotion]] / self.total))
        hrule()

        # For all images of this emotion, what is the distribution of our model's predictions' spread?
        print("DETAILED")
        hrule()
        col_1_size = Result.get_col_1_size(self.total)
        print_row(['Emotion', 'Hits', 'Share'], col1=col_1_size)
        for i in range(Result.TOTAL_EMOTIONS):
            emotion = ('*' if EMOTION_MAP[i] == self.emotion else ' ') + EMOTION_MAP[i]
            share = '%.2f%%' % (100 * self.predictions[i] / self.total)
            print_row([emotion, self.predictions[i], share], col1=col_1_size)
        fat_hrule(20)

### HELPER FUNCTIONS ###
def _hrule(length=7, char='-'):
    print("="*length)

# -------
def hrule(length=7):
    _hrule(length)

# ===========
def fat_hrule(length=10):
    _hrule(length, "=")

# Add spaces to the right to set fixed length for table columns
def pad(string, length):
    return '{:<{size}}'.format(string, size=length)

# Print a table row
def print_row(row, col0=Result._COL_0_SIZE, col1=5, col2=Result._COL_2_SIZE):
    print(*map(pad, row, [col0, col1, col2]), sep='\t')