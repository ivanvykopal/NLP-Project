# Create LASER embdings for the dataset

from embeddings.laser.laser_vectorizer import LaserVectorizer


class LASERSentenceEmbeddings:
    def __init__(self, dir_path):
        self.model = LaserVectorizer(dir_path=dir_path)
        self.model.load()

    def __call__(self, texts):
        self.embeddings = self.model.vectorize(texts)
        return self.embeddings

    def save(self):
        self.model.save()
