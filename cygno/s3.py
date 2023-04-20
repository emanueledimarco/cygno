#
################## General TOOL for S3 ##############
#

BAKET_POSIX_PATH = '/jupyter-workspace/cloud-storage/'
BAKET_REST_PATH = 'https://s3.cloud.infn.it/v1/AUTH_2ebf769785574195bde2ff418deac08a/'

def kb2valueformat(val):
    import numpy as np
    if int(val/1024./1024/1024.)>0:
        return val/1024./1024./1024., "Gb"
    if int(val/1024./1024.)>0:
        return val/1024./1024., "Mb"
    if int(val/1024.)>0:
        return val/1024., "Kb"
    return val, "byte"

def open_aws_session(session, number_of_try=3, wait=10, verbose=False):
    import boto3
    from boto3sts import credentials as creds
    import time
    ntry=1
    while ntry<=number_of_try:
        try:
            aws_session = creds.assumed_session(session)
            return aws_session
        except:
            print ("ERROR opening AWS session retry # {:d} in {:d}".format(ntry, wait))
            time.sleep(wait)
        ntry+=1
    if ntry==number_of_try:
        return 0
    
############################    

# def root_file(run, tag='LAB', posix=False, verbose=False):
#     if posix:
#         BASE_URL  = BAKET_POSIX_PATH+'cygno-data/'
#         if run <= 4504:
#             BASE_URL  = BAKET_POSIX_PATH+'cygno/Data/'
#     else:
#         BASE_URL  = BAKET_REST_PATH+'cygno-data/'
#         if run <= 4504:
#             BASE_URL  =  BAKET_REST_PATH+'cygnus/Data/'
    
#     file_root = (tag+'/histograms_Run%05d.root' % run)
#     if verbose: print(BASE_URL+file_root)
#     return BASE_URL+file_root


def root_file(run, tag='LAB', cloud=False, verbose=False):
    if cloud:
        BASE_URL  = BAKET_REST_PATH+'cygno-data/'
        if run <= 4504:
            BASE_URL  =  BAKET_REST_PATH+'cygnus/Data/'
        f  = BASE_URL+(tag+'/histograms_Run%05d.root' % run)
    else:
        f = (tag+'/histograms_Run%05d.root' % run)
        
    if verbose: print(f)
    return f

def mid_file(run, tag='LNGS', cloud=False, verbose=False):
    if cloud:
        BASE_URL  = BAKET_REST_PATH+'cygno-data/'
        f = BASE_URL+(tag+'/run%05d.mid.gz' % run)
    else:
        f = ('/run%05d.mid.gz' % run)
    if verbose: print(f)
    return f



def bucket_list(tag, bucket='cygno-sim', session="infncloud-iam", filearray=False, verbose=False):
    import boto3
    from boto3sts import credentials as creds
    # from mypy_boto3_sts import credentials as creds
    
    endpoint='https://minio.cloud.infn.it/'
    version='s3v4'
    key = tag+'/'
    if verbose: print(">> listing", tag, "on bucket", bucket, "for session",  session, "\n")
    aws_session = open_aws_session(session, number_of_try=3, wait=10, verbose=False)
    if aws_session:
        s3 = aws_session.client('s3', endpoint_url=endpoint,
                                config=boto3.session.Config(signature_version=version),verify=True)
        lsarray=[]
        IsTruncated = True
        NextMarker  = ''
        while IsTruncated:
            response    = s3.list_objects(Bucket=bucket, Marker=NextMarker)
            IsTruncated = response['IsTruncated']
            contents = response['Contents']
            for i, file in enumerate(contents):
                if key in str(file['Key']):
                    if filearray:
                        lsarray.append(file['Key'])
                    else:
                        print("{0:20s} {1:s}".format(str(file['LastModified']).split(".")[0].split("+")[0], file['Key']))
            if IsTruncated:
                Marker      = response['Marker']
                NextMarker  = response['NextMarker']
            if verbose: print("bucket troncato? "+str(IsTruncated),"Marker: ", Marker,"NextMarker: ", NextMarker)
        # return array 
        return lsarray
    else:
        return [0]

def obj_put(filename, tag, bucket='cygno-sim', session="infncloud-iam", verbose=False):
    # https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3-uploading-files.html#uploading-files
    import boto3
    from boto3sts import credentials as creds
    # from mypy_boto3_sts import credentials as creds
    import logging
    import botocore
    import requests
    import os
    #
    endpoint='https://minio.cloud.infn.it/'
    version='s3v4'
    #
    if verbose: print(">> upload", filename,"taged", tag, "on bucket", bucket, "for session",  session, "\n")
    aws_session = open_aws_session(session, number_of_try=3, wait=10, verbose=False)
    if aws_session:
        s3 = aws_session.client('s3', endpoint_url=endpoint, config=boto3.session.Config(signature_version=version),verify=True)

        key = tag+'/'

        # Upload the file

        try:
            response=s3.head_object(Bucket=bucket,Key=key+filename)
            value, unit = kb2valueformat(response['ContentLength'])
            print("The file already exists and has a dimension of {:.2f} {:s}".format(value, unit))
            #print("The file already exists and has a dimension of "+str(response['ContentLength']/1024./1024.)+' MB')
            return True, False

        except (botocore.exceptions.ConnectionError, requests.exceptions.ConnectionError):
            print("There was a connection error or failed")
            return False, False

        except botocore.exceptions.ClientError:

            if verbose: print('No file with this name was found on the cloud, it will be uploaded')
            try:
                out = key+os.path.basename(filename)
                response = s3.upload_file(filename, bucket, out)
                if verbose: print ('Uploaded file: '+out)
            except Exception as e:
                logging.error(e)
                return False, False
            return True, True
    else:
        return False, False

def obj_get(filein, fileout, tag, bucket='cygno-sim', session="infncloud-iam", verbose=False):
    # https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3-uploading-files.html#uploading-files
    import boto3
    from boto3sts import credentials as creds
    import logging
    import botocore
    import requests
    import os
    #
    endpoint='https://minio.cloud.infn.it/'
    version='s3v4'
    #
    if verbose: print(">> get", filein, fileout,"taged", tag, "on bucket", bucket, "for session",  session, "\n")
    aws_session = open_aws_session(session, number_of_try=3, wait=10, verbose=False)
    if aws_session:
        s3 = aws_session.client('s3', endpoint_url=endpoint, config=boto3.session.Config(signature_version=version),verify=True)

        key = tag+'/'

        # Download the file

        try:
            response=s3.head_object(Bucket=bucket,Key=key+filein)
            value, unit = kb2valueformat(response['ContentLength'])
            print("downloading file of {:.2f} {:s}...".format(value, unit))    
        except (botocore.exceptions.ConnectionError, requests.exceptions.ConnectionError):
            print("There was a connection error or failed")
            return False

        except botocore.exceptions.ClientError:
            print('No file with this'+fielneme+'was found on the cloud')
            return False
        try:
            object_in = key+filein
            response = s3.download_file(bucket, object_in, fileout)
            if verbose: print ('Downloaded file: '+fileout)
            return True
        except Exception as e:
            logging.error(e)
            return False
    return False
    
def obj_size(filein, tag, bucket='cygno-sim', session="infncloud-iam", verbose=False):
    # https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3-uploading-files.html#uploading-files
    import boto3
    from boto3sts import credentials as creds
    import logging
    import botocore
    import requests
    import os
    #
    endpoint='https://minio.cloud.infn.it/'
    version='s3v4'
    #
    if verbose: print(">> get", filein,"taged", tag, "on bucket", bucket, "for session",  session, "\n")
    aws_session = open_aws_session(session, number_of_try=3, wait=10, verbose=False)
    if aws_session:
        s3 = aws_session.client('s3', endpoint_url=endpoint, config=boto3.session.Config(signature_version=version),verify=True)

        key = tag+'/'


        try:
            response=s3.head_object(Bucket=bucket,Key=key+filein)
            value, unit = kb2valueformat(response['ContentLength'])
            if verbose: print("File of {:.2f} {:s} size".format(value, unit)) 
            return int(response['ContentLength'])

        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                if verbose: print("File not found")
            else:
                if verbose: print("S3 failure")
            return 0
    return 0
    
def obj_rm(filename, tag, bucket='cygno-sim', session="infncloud-iam", verbose=False):
    # https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3-uploading-files.html#uploading-files
    import boto3
    from boto3sts import credentials as creds
    import logging
    import botocore
    import requests
    import os
    #
    endpoint='https://minio.cloud.infn.it/'
    version='s3v4'
    #
    if verbose: print(">> get", filename, "tag", tag, "on backet", bucket, "for session",  session, "\n")
    aws_session = open_aws_session(session, number_of_try=3, wait=10, verbose=False)
    if aws_session:
        s3 = aws_session.client('s3', endpoint_url=endpoint, config=boto3.session.Config(signature_version=version),verify=True)

        key = tag+'/'

        # Download the file

        try:
            response=s3.head_object(Bucket=bucket,Key=key+filename)
            value, unit = kb2valueformat(response['ContentLength'])
            print("removing file of {:.2f} {:s}...".format(value, unit))    
        except (botocore.exceptions.ConnectionError, requests.exceptions.ConnectionError):
            print("There was a connection error or failed")
            return False

        except botocore.exceptions.ClientError:
            print('No file with this'+filename+'was found on the cloud')
            return False
        try:
            object_in = key+filename
            response = s3.delete_object(Bucket=bucket,Key=object_in)
            print ('removed file: '+filename)
            return True
        except Exception as e:
            logging.error(e)
            return False
    return False