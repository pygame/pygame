#!/usr/bin/env python
import dbapi20
import unittest
import msidb
import os

db_path = 'testdb.msi'

class test_msidb(dbapi20.DatabaseAPI20Test):
    driver = msidb
    connect_args = ()
    connect_kw_args = {'dsn': db_path, 'persist': msidb.OPEN_TRANSACT}

    table_prefix = ''
    ddl1 = 'create table %sbooze (name char(20) primary key name)' % table_prefix
    ddl2 = 'create table %sbarflys (name char(20) primary key name)' % table_prefix
    xddl1 = 'drop table %sbooze' % table_prefix
    xddl2 = 'drop table %sbarflys' % table_prefix

    def setUp(self):
        # Call superclass setUp In case this does something in the
        # future
        dbapi20.DatabaseAPI20Test.setUp(self) 

        if not os.path.exists(db_path):
            con = self.driver.connect(db_path, persist=msidb.OPEN_CREATE)
            con.close()

    def tearDown(self):
        dbapi20.DatabaseAPI20Test.tearDown(self)

    def test_nextset(self): pass
    def test_setoutputsize(self): pass
    def test_callproc(self): pass

if __name__ == '__main__':
    unittest.main()
