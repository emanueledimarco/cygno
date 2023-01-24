# CYGNO LIB
middle software tools to handle cygno data, repository, images, ecc.

1. [install](https://github.com/CYGNUS-RD/cygno/edit/main/README.md#install-the-cygno-library)
2. [Command line tools](https://github.com/CYGNUS-RD/cygno/edit/main/README.md#cygno-cli-tools)
   * [Open ID and cygno_repo ](https://github.com/CYGNUS-RD/cygno/edit/main/README.md#cygno_repo)
   * [stored data info](https://github.com/CYGNUS-RD/cygno/edit/main/README.md#cygno_runs)
   * [convert HIS Camera files 2 root T2H inage](https://github.com/CYGNUS-RD/cygno/edit/main/README.md#cygno_his2root)
   * [convert Midas files 2 root T2H inage](https://github.com/CYGNUS-RD/cygno/edit/main/README.md#cygno_mid2root)
3. [Library functions](https://github.com/CYGNUS-RD/cygno/edit/main/README.md#cygno-library-functions)
   * [files handling](https://github.com/CYGNUS-RD/cygno/blob/main/README.md#files)
   * [SQL](https://github.com/CYGNUS-RD/cygno/edit/main/README.md#storage--sql)
   * [old stuff](https://github.com/CYGNUS-RD/cygno/edit/main/README.md#old-api-before-midas-raw-data)
   * [logbook tools](https://github.com/CYGNUS-RD/cygno/edit/main/README.md#logbook)
   * [storage S3 tools](https://github.com/CYGNUS-RD/cygno/edit/main/README.md#s3-repo)
   * [images tools](https://github.com/CYGNUS-RD/cygno/edit/main/README.md#images-tools)

## install the CYGNO library:

requirements:
* Pyroot: https://root.cern/manual/python/ 
* oidc-agent: https://indigo-dc.gitbook.io/oidc-agent/installation
* boto3sts: https://github.com/DODAS-TS/boto3sts
* MIDAS: https://github.com/CYGNUS-RD/middleware/tree/master/midas
      pip install git+https://github.com/CYGNUS-RD/cygno.git -U

## CYGNO CLI Tools 
### *cygno_repo*

tool to operate on CYGNO backet in S3 exeperiment repository

requirements:

* configure *oidc-agent* on your machine: https://codimd.web.cern.ch/s/SL-cWzDZB (DAQ setup, expert only https://codimd.web.cern.ch/s/_XqFfF_7V)
* example for osx

      brew tap indigo-dc/oidc-agent
      brew install oidc-agent

* install the IAM-Profile (not WLCG-Profile token) as reported in the second part of the guide https://codimd.web.cern.ch/s/SL-cWzDZB

* install python library  (https://github.com/DODAS-TS/boto3sts): 

      pip install git+https://github.com/DODAS-TS/boto3sts
      pip install git+https://github.com/CYGNUS-RD/cygno.git
      
* see https://boto3.amazonaws.com/v1/documentation/api/latest/index.html for S3 documentation

before run the script crate the iam token:

      eval `oidc-agent`
      oidc-token infncloud-iam (to generate or see your active token)
 
or refresh the token
 
      eval `oidc-agent`
      oidc-gen --reauthenticate --flow device infncloud-iam (if you alrady have the token)
      
you can also add in your bush (or equivalent) profile

	echo "CLOUD storage setup: infncloud-iam"
	export REFRESH_TOKEN="xxx"
	export IAM_CLIENT_SECRET="yyy"
	export IAM_CLIENT_ID="zzz"
	export IAM_SERVER=https://iam.cloud.infn.it/
	unset OIDC_SOCK; unset OIDCD_PID; eval `oidc-keychain`
	oidc-gen --client-id $IAM_CLIENT_ID --client-secret $IAM_CLIENT_SECRET --rt $REFRESH_TOKEN --manual --issuer $IAM_SERVER --pw-cmd="echo pwd" --redirect-uri="edu.kit.data.oidc-agent:/redirect http://localhost:34429 http://localhost:8080 http://localhost:4242" --scope "iam openid email profile offline_access" infncloud-iam
	
to get setup info, type:

	oidc-gen -p infncloud-iam


usage

	Usage: cygno_repo	 [-tsv] [ls backet]
				 [put backet filename]
				 [[get backet filein] fileout]
				 [rm backet fileneme]
	
	
	Options:
	  -h, --help            	show this help message and exit
	  -t TAG, --tag=TAG     	tag where dir for data;
	  -s SESSION, --session=SESSION	token profile [infncloud-iam];
	  -v, --verbose         	verbose output;
                   
example:

      Giovannis-MacBook-Air-2:script mazzitel$ cygno_repo ls cygno-sim -t test
      2021-10-17 10:03:21  test/s3_list.py
      Giovannis-MacBook-Air-2:script mazzitel$ cygno_repo put cygno-sim s3_function.py -t test
      Giovannis-MacBook-Air-2:script mazzitel$ cygno_repo ls cygno-sim -t test
      2021-10-26 16:36:03  test/s3_function.py
      2021-10-17 10:03:21  test/s3_list.py
      Giovannis-MacBook-Air-2:script mazzitel$ cygno_repo get cygno-sim s3_function.py -t test
      downloading file of 5.82 Kb...
      Giovannis-MacBook-Air-2:script mazzitel$ cygno_repo ls cygno-sim -t test
      2021-10-26 16:36:03  test/s3_function.py
      2021-10-17 10:03:21  test/s3_list.py
      Giovannis-MacBook-Air-2:script mazzitel$ cygno_repo rm cygno-sim s3_function.py -t test
      removing file of 5.82 Kb...
      removed file: s3_function.py
      Giovannis-MacBook-Air-2:script mazzitel$ cygno_repo ls cygno-sim -t test
      2021-10-17 10:03:21  test/s3_list.py

Data are also shared in CYGNO CLOUD resources via the CYGNO application: https://notebook.cygno.cloud.infn.it:8888/ (jupyter notebook, python, root and terminal use dodasts/cygno-jupyter:v2.1 image) and availeble via web broser https://minio.cloud.infn.it/
      
###  *cygno_runs*

tool to show SQL runs infromation stored in the logbook

	Usage: cygno_runs        [-ajv] run number

	Options:
	  -h, --help     show this help message and exit
	  -a, --all      all runs in DBs;
	  -j, --json     json output;
	  -v, --verbose  verbose output;
		
example:

	cygno_runs 368 -j (new logbook json output)
	cygno_runs -a (dump all the dadabase)
	
HTTP access
	
* sigle run query output: http://lnf.infn.it/~mazzitel/php/cygno_sql_query.php?run=368 
* full database table: http://lnf.infn.it/~mazzitel/php/cygno_sql_query.php?table=on 
	
	
###  *cygno_his2root*

convert HIS HoKaWo output file in CYGNO root histograms data files

	Usage: cygno_his2root	 [-d] DIRECTORY
	
	Options:
	  -h, --help     show this help message and exit
	  -d, --delete   delete HIS file after conversion;
	  -v, --verbose  verbose output;
	  
###  *cygno_mid2root*

convert MIDAS output file in CYGNO root histograms data files. Required:

	pip install 'https://github.com/CYGNUS-RD/middleware/blob/master/midas/midaslib.tar.gz?raw=true'

tool:

	Usage: cygno_mid2root	 <RUN number>

	Options:
	  -h, --help            show this help message and exit
	  -p PATH, --path=PATH  path to file or cache directory
	  -v, --verbose         verbose output;

## CYGNO library functions

### files

* open_mid(run, path='/tmp/',  cloud=True,  tag='LNGS', verbose=False): open/cache MIDAS form cloud in path, return *poiner to file* ([see example](https://github.com/CYGNUS-RD/cygno/blob/main/dev/readMidasFile.ipynb))
* daq_cam2array(bank, verbose=False): decode daq equipment CAM, return *image (2D array) shape_x_image (int), shape_y_image (int)*
* daq_dgz2header(bank, verbose=False): decode daq equipment header DGZ, return *number_events (int), number_channels (int), number_samples (int)*
* daq_dgz2array(bank, header, verbose=False): decode daq equipment data DGZ, return *waveform array of #number_channels * #number_samples dimesion* 
* daq_slow2array(bank, verbose=False): decode daq equipment INPUT

### Storage & SQL
* daq_sql_cennection(verbose=False): return SQL connection
* daq_update_runlog_replica_checksum(connection, run_number, md5sum, verbose=False): return run checksum 
* daq_update_runlog_replica_tag(connection, run_number, TAG, verbose=False): return run tag
* daq_update_runlog_replica_size(connection, run_number, size, verbose=False): return run size
* daq_update_runlog_replica_status(connection, run_number, storage, status=-1, verbose=False): update run replica status
* daq_read_runlog_replica_status(connection, run_number, storage, verbose=False): return run replica status
* daq_not_on_tape_runs(connection, verbose=False): return array of file not on tape
* daq_run_info(connection, run, verbose=False): return run info

### old api before midas raw data
* open_root(run, path='/tmp/',  cloud=True,  tag='LAB', verbose=False)  open/cache ROOT form cloud in path
class cfile:
```	
		def __init__(self, file, pic, wfm, max_pic, max_wfm, x_resolution, y_resolution):
			self.file         = file
			self.pic          = pic 
			self.wfm          = wfm
			self.max_pic      = max_pic
			self.max_wfm      = max_wfm
			self.x_resolution = x_resolution
			self.y_resolution = y_resolution 
```
* open_(run, tag='LAB', posix=False, verbose=True) - open cygno ROOT/MID file from remote or on cloud posix like access and return cfile class type
* read_(f, iTr) - return image array from file poiter
* pic_(cfile, iTr=0) - return immage array of track iTr from cfile
* wfm_(cfile, iTr=0, iWf=0) - return amplitude and time of iTr track and iWr waveform from cfile
* ped_(run, path='./ped/', tag = 'LAB', posix=False, min_image_to_read = 0, max_image_to_read = 0, verbose=False) - cerate (if not exist) root file image of mean and sigma for each pixel and return main and sigma imege of pedestal runs

### logbook 
* read_cygno_logbook(verbose=False) 		ruturn pandas db old google sheet logbook info
* run_info_logbook(run, sql=True, verbose=True)	return pandas db google/sql run [int] info

### s3 repo
* s3.root_file(run, tag='LAB', posix=False, verbose=False)
* s3.backet_list(tag, bucket='cygno-sim', session="infncloud-iam", verbose=False)
* s3.obj_put(filename, tag, bucket='cygno-sim', session="infncloud-iam", verbose=False)
* s3.obj_get(filein, fileout, tag, bucket='cygno-sim', session="infncloud-iam", verbose=False)
* s3.obj_rm(filename, tag, bucket='cygno-sim', session="infncloud-iam", verbose=False)

### images tools
* cluster_par(xc, yc, image): return intesity and dimestion
* n_std_rectangle(x, y, ax, image = np.array([]), n_std=3.0, facecolor='none', kwargs): return rettagle confindece level image 
* confidence_ellipse(x, y, ax, image = np.array([]), n_std=3.0, facecolor='none', kwargs): return ellips confidence level, rimage
* confidence_ellipse_par(x, y, image = np.array([]), n_std=3.0, facecolor='none', kwargs): return image quantity width, height, pearson, sum, size
* cluster_elips(points): return points_3d
* rebin(a, shape): return rebined shape
* smooth(y, box_pts): return smooted array of box_pts dimesion
* img_proj(img, vmin, vmax, log=False): retrun plot of image projection

NB usefull and primitive UNIX and SQL function are available in the library https://github.com/CYGNUS-RD/cygno/blob/main/cygno/cmd.py 
