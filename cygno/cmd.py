############# general TOOL for file #############

def mv_file(filein, fileout):
    import os
    command = '/bin/mv '+ filein +' '+ fileout
    return os.system(command)

def cp_file(filein, fileout):
    import os
    command = '/bin/cp '+ filein +' '+ fileout
    return os.system(command)

def rm_file(filein):
    import os
    command = '/bin/rm '+ filein
    return os.system(command)

def grep_file(what, filein):
    import subprocess
    command = '/usr/bin/grep ' + what +' '+filein
    status, output = subprocess.getstatusoutput(command)
    return output

def mkdir_file(folder):
    import os
    command = '/bin/mkdir -p '+ folder
    return os.system(command)


def append2file(line, filein):
    import os
    command = 'echo '+ line + ' >> '+filein
    return os.system(command)

def reporthook(blocknum, blocksize, totalsize, verbose=True):
    import sys
    readsofar = blocknum * blocksize
    if totalsize > 0:
        percent = readsofar * 1e2 / totalsize
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(totalsize)), readsofar, totalsize)
        if verbose: sys.stderr.write(s)
        if readsofar >= totalsize: # near the end
            if verbose: sys.stderr.write("\n")
    else: # total size is unknown
        if verbose: sys.stderr.write("read %d\n" % (readsofar,))


def cache_file(url, cachedir='/tmp/', verbose=False):
    import os
    import sys
    from platform import python_version
    if not os.path.exists(cachedir):
        os.mkdir(cachedir)
    tmpname = cachedir+url.split('/')[-1]
    if not os.path.exists(tmpname):
        if verbose: print("downloading: "+tmpname)
        if python_version().split('.')[0]=='3':
            from urllib.request import urlretrieve
            urlretrieve(url, tmpname, reporthook)
#             import urllib
#             urllib.request.urlretrieve(url, tmpname, reporthook)
        else:
            import urllib
            urllib.urlretrieve(url, tmpname, reporthook)
    else:
        if verbose: sys.stderr.write('file '+tmpname+' cached')
        
    return tmpname

# SQL and File storage

def file_as_bytes(file):
    with file:
        return file.read()
    
def file_md5sum(file):
    import hashlib
    try:
        return hashlib.md5(file_as_bytes(open(file, 'rb'))).hexdigest()
    except:
        print ("md5sum error", file)
        return False
    
def push_panda_table_sql(connection, table_name, df):
    
    mycursor=connection.cursor()
    mycursor.execute("SHOW TABLES LIKE '"+table_name+"'")
    result = mycursor.fetchone()
    if not result:
        cols = "`,`".join([str(i) for i in df.columns.tolist()])
        db_to_crete = "CREATE TABLE `"+table_name+"` ("+' '.join(["`"+x+"` REAL," for x in df.columns.tolist()])[:-1]+")"
        print ("[Table {:s} created into SQL Server]".format(table_name))
        mycursor = connection.cursor()
        mycursor.execute(db_to_crete)

    cols = "`,`".join([str(i) for i in df.columns.tolist()])

    for i,row in df.iterrows():
        sql = "INSERT INTO `"+table_name+"` (`" +cols + "`) VALUES (" + "%s,"*(len(row)-1) + "%s)"
        mycursor.execute(sql, tuple(row.astype(str)))
        connection.commit()

    mycursor.close()

    
def update_sql_value(connection, table_name, row_element, row_element_condition, 
                     colum_element, value, verbose=False):
    if isinstance(value, str):
        svalue="\""+value+"\""
    else:
        svalue=str(value)
    if isinstance(row_element_condition, str):
        scondition="\""+row_element_condition+"\""
    else:
        scondition=str(row_element_condition)
    sql = "UPDATE `"+table_name+"` SET `"+colum_element+"` = "+ \
    svalue+" WHERE `"+row_element+"` = "+scondition+";"
    if verbose: print(sql)
    try:
        mycursor = connection.cursor()
        mycursor.execute(sql)
        connection.commit()
        if verbose: print(mycursor.rowcount, "Update done")
        mycursor.close()
        return 0
    except:
        return 1