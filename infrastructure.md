# Computing infrastrcture
The CYGNO exepriment develop a facility based on the [INFN cloud](https://www.cloud.infn.it/) to host:
- data experiment storage ([S3 based](https://it.wikipedia.org/wiki/Amazon_S3)) - [status](https://monitoring.cloud.infn.it:3000/d/cygno-s3/storage-s3-cygno?orgId=1)
- tape backup storage - [status](https://t1metria.cr.cnaf.infn.it/d/ZArHZvEMz/storage-usage-per-experiment?orgId=18&var-exp=cygn&var-vo=CYGNO&from=now-30d&to=now) (NB for disk space refer [S3](https://monitoring.cloud.infn.it:3000/d/cygno-s3/storage-s3-cygno?orgId=1) starage status)
- two notebooksinterface with python and root kernles, bach resources, posix acces to data storage - [status nb1](https://90.147.174.175:3000), [status nb2](https://192.135.24.178:3000/login)
- bach resources accesible via condor queues - [status](https://grafana.131.154.96.115.myip.cloud.infn.it/login)

CYGNO status of usage of [pledged resources](https://monitoring.cloud.infn.it:3000/d/u1sBcydVk/cygno-cloud-capacities?orgId=1)

Moreover, computing resources are available at LNF and LNGS (Cygno VM login and U-LITE nodes) and two [DAQ server](https://drive.google.com/file/d/1kEzvfJK7WSXK2Y1vfEwRqcH9uSmoYsXl/view?usp=sharing) equipped with GPU


### Signup on computing ressources (needed for all resources: CLOUD, LNGS, LNF)
* if you are not associeted/hosted/employed of INFN plese signup on: https://signup.app.infn.it/ (tips for foreign users see below)
* accept the security policy  https://userportal.app.infn.it/;
* follow the traning on computing security;
* for foreign users no CODICE FISCALE (CF) is needed;
<!---
  * for foreign users to be reggistred on AAI you need a CODICE FISCALE (CF) that you can generate with the tool https://quifinanza.it/strumenti/codice-fiscale 
  ```
      (Provincia: “Stato Estero")
      (LUOGO di NASCITA: BRASILE)
  ```
--->
  * up to now also a local username in INFN local sites computing resources is required (see following point)

### Computing resources on INFN Cloud
* signup as **user** on [INFN CLOUD](https://guides.cloud.infn.it/docs/users-guides/en/latest/users_guides/getting_started.html), as responsable put everyware **Giovanni Mazzitelli**
* when approved follow the [HOWTO](https://github.com/CYGNUS-RD/cygno/blob/main/infrastructure.md#usage-of-the-cygno-notebook-web-interface-and-cloud-services) to exploit the resources


### Computing resources and OPEN VPN @ LNF (test DAQ server, ecc)
* send an email to: giovanni.mazzitelli@lnf.infn.it to be autorized
* when aproved install the profile http://www.lnf.infn.it/computing/networking/openvpn-en.php
* if you need also local computing resesources plese fill http://www.lnf.infn.it/computing/cgi-bin/newaccountrequest.pl 

### Computing resources and OPEN VPN @ LNGS (DAQ, shift, ecc)
* send an email to: giovanni.mazzitelli@lnf.infn.it to be autorized
* if you need also local computing resesources **Cygno VM login and U-LITE nodes** (deprecated) plese specify in the mail.
* when aproved install the profile install the profile https://www.lngs.infn.it/en/vpn

### DAQ and Middle Ware ###
* Data are collected by DAQ at LNF and LNGS [server configuration](https://drive.google.com/file/d/1kEzvfJK7WSXK2Y1vfEwRqcH9uSmoYsXl/view?usp=sharing) 
* Exeperiment data are monitored by the quasi-online recostracion by the [Middle Ware](https://github.com/CYGNUS-RD/middleware)

### CYGNO CLOUD Storage
Data collected by the experiment DAQ are automatically pushed on INFN [S3 cloud](https://it.wikipedia.org/wiki/Amazon_S3) storage. The storage data and the experiment area for analysis and simulation can be acces and manage via: 

* Web Tool: https://minio.cloud.infn.it/minio/login
* Cloud CYGNO web interface tool: https://notebook.cygno.cloud.infn.it:8888/ 
* CLI tool: https://github.com/CYGNUS-RD/cygno#cygno-cli-tool-cygno_repo

the cloud-storage/ contain tree backet:
* cloud-data: daq stored data, read only
* cloud-sim: simulation input and output data, read and write
* cloud-analysis: analysis input and output data, read and write
* (cygno - old data repository, USERNAME private repository on cloud, scratch repository on cloud)



### Usage of the CYGNO notebook web interface and Cloud services
Two VM offer acces to cloud infrastrucure via web services based on jupyter notebook interface
* production: 
  - CYGNO notebook 01 https://notebook.cygno.cloud.infn.it:8888/ ([16 CPU](https://novabench.com/parts/cpu/intel-core-broadwell-ibrs)/32 GB); [usage monitor](https://90.147.174.175:3000/login)
  - CYGNO notebook 02 https://notebook02.cygno.cloud.infn.it:8888/ (16 CPU/32 GB);
* test environment https://192.135.24.159:8888/ ([8 CPU](https://www.intel.it/content/www/it/it/products/platforms/details/cascade-lake.html)/16 GB); [usage monitor](https://192.135.24.159:3000)
* the web inteface offer the possibility to run a specific software configuration. In general:
  * tag [dodas](https://github.com/DODAS-TS/dodas-docker-images) realises are the official one and approved by INFN
  * tag [gmazzitelli](https://github.com/gmazzitelli/dodas-docker-images) are realisesed fork of official project under [development](https://github.com/gmazzitelli/cloud_docker_dev) and test

### Tag v1.0.17 ###
  - python 3.9.10 (not deafult), emacs, screen, root_numpy, uproot, pydot, tensorflow, opencv-python, graphviz
  - [full packege list](https://raw.githubusercontent.com/CYGNUS-RD/cygno/main/requirements.txt)
### Tag < v1.0.17 ###
  - ROOT 6.24/06
  - Python 2/3.6 ([Default package list notebook >= 16](https://raw.githubusercontent.com/CYGNUS-RD/cygno/main/img/PackageListV16.txt))
  - Garfield 
  - GEANT 4.10.5
  - https://gitlab.cern.ch/RooUnfold
  - https://github.com/christopherpoole/CADMesh
  - access to CYGNO cluster (~ 50 cores), condor queues, via the notebook terminal or via any computer by means of [dedicated container](https://github.com/CYGNUS-RD/mycondor)

### Usage:
* to access the resource login with AAI credentials (see above to be athorized) 
<img src="https://github.com/CYGNUS-RD/cygno/blob/main/img/login.png" alt="login" style="width:400px;"/>
<img src="https://github.com/CYGNUS-RD/cygno/blob/main/img/aai.png" alt="login" style="width:400px;"/>

* start your notebook choosing version and RAM needed. That RAM is the maximum your interactive job can exploit. if there are concurred interactive job form other users draining the ram you can have your job killed. so don't ask the maximum of RAM if you don't relay need, and use condor queue instead of interactive jobs: https://github.com/CYGNUS-RD/mycondor#cygno-condor-queue 

<img src="https://github.com/CYGNUS-RD/cygno/blob/main/img/resorce.png" alt="login" style="width:400px;"/>

* run/edit your notebook Python/ROOT or script via the available buttons
<img src="https://github.com/CYGNUS-RD/cygno/blob/main/img/buttos.png" alt="login" style="width:400px;"/>

* the file system is availbale at the dafult path **/jupyter-workspace** that is divided in:
  * **/jupyter-workspace/cloud-storage/**: POSIX simulated access to S3 CYGNO storage system (experiment data, simulation and analysis repository, see also [CYGNO cloud storage](https://github.com/CYGNUS-RD/cygno/blob/main/infrastructure.md#cygno-cloud-storage)) 
  * under **/jupyter-workspace/cloud-storage/**: is also available a *USERNAME* directory (accessible only by user) and a *screach* area (accesible by anybody). Those directories are on S3 and permanent.
  * **/jupyter-workspace/private/**: working directory; this access to a local file system in case of cloud fault data can be lost (from v17 this foleder is atomaticaly backuped in **/jupyter-workspace/cloud-storage/USERNAME/private**, safe and always reachbele by [MINIO](https://minio.cloud.infn.it/))
  * **/jupyter-workspace/shared/**: shared working directory on lacal system

* it's strogly sujest to develop and run your code from **/jupyter-workspace/private** use private folder to develop and store your code
* all paths to exploit installed softwares, and condor queues since realese v17, are configured by dafault. A personal setup can be configured editing the file **/jupyter-workspace/cloud-storage/USERNAME/.bashrc** (example git personal config)
