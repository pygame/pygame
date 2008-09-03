from ZODB import DB, FileStorage
import transaction

class ZDB(object):
    def __init__(self, path):
        self.storage = FileStorage.FileStorage(path)
        self.db = DB(self.storage)
        self.connection = self.db.open()
        self.root = self.connection.root()
    
    def commit(self):
        transaction.commit()

    def close(self, commit=True):
        if commit: self.commit()
        self.connection.close()
        self.db.close()
        self.storage.close()

__all__ = ['ZDB']