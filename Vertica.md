Installing Vertica
==================

**Vertica** is a columnar data storage platform designed to handle large volumes of data, which enables very fast query performance in traditionally intensive scenarios. 
The product improves query performance over traditional database relational database systems, provides high-availability, and petabyte scalability on commodity enterprise servers.

HP Vertica’s platform design features include:

* Columnar data storage organization
* Standard SQL interface with many analytics capabilities built in
* Compression to reduce storage costs
* Support for standard programming interfaces
* High performance and parallel data transfer
* Ability to store machine learning models and use them for database scoring

**<center>Scheme</center>**

![scheme](https://i.imgur.com/UJuTs4M.png)

## Basic Installation on Ubuntu

> :warning: First of all, check the requirements [Before You Install Vertica.](https://www.vertica.com/docs/8.1.x/HTML/Content/Authoring/InstallationGuide/BeforeYouInstall/BeforeYouInstallVertica.htm)

1. As root (or sudo) run the install script. The script must be run by a BASH shell as root or as a user with sudo privileges. You can configure many options when running the install script. See Basic Installation Parameters below for the complete list of options.

    If the installer fails due to any requirements not being met, you can correct the issue and then run the installer again with the same command line options.

    To perform a basic installation:

    * As root:

    ```ubuntu
    # /opt/vertica/sbin/install_vertica --hosts  host_list --rpm package_name --dba-user dba_username
    ```

    * Using sudo:
    ```ubuntu
    $ sudo /opt/vertica/sbin/install_vertica --hosts host_list --rpm package_name --dba-user dba_username
    ```
    
> :warning: If you place install_vertica somewhere other than /opt/vertica, you need to create a symlink from that location to /opt/vertica. You need to create this symlink on all nodes in the cluster, otherwise the database will not start.

#### Basic Installation Parameters

<table>
<tr><td> Option </td> 
    <td> Description </td>
</tr>
<tr><td> --hosts <em>host_list</em> </td>
<td>A comma-separated list of IP addresses to include in the cluster; do not include space characters in the list. Examples:
            
```
--hosts 127.0.0.1
--hosts 192.168.233.101,192.168.233.102,192.168.233.103
```
</td></tr>
<tr><td> --rpm <em>package_name</em><br>
--deb <em>package_name</em> </td>
<td>The path and name of the Vertica RPM package. Example:
    
`--rpm /tmp/vertica_8.1.x.x86_64.RHEL6.rpm`
    
For Debian and Ubuntu installs, provide the name of the Debian package, for example:
    
`--deb /tmp/vertica_7.2.x86.deb`
<tr><td>--dba-user <em>dba_username</em>
</td>
<td>The name of the <b>Database Superuser</b> system account to create. Only this account can run the Administration Tools. If you omit the <code>--dba-user parameter</code>, then the default database administrator account name is <code>dbadmin</code>.<br><br>
This parameter is optional for new installations done as root but must be specified when upgrading or when installing using sudo. If upgrading, use the <code>-u</code> parameter to specify the same DBA account name that you used previously. If installing using sudo, the user must already exist.
<br><br>
<b>Note:</b> if you manually create the user, modify the user's .bashrc file to include the line: <code>PATH=/opt/vertica/bin:$PATH</code> so that the Vertica tools such as vsql and admintools can be easily started by the dbadmin user.
</tr>
</td>
</tr>
</table>

2. When prompted for a password to log into the other nodes, provide the requested password. Doing so allows the installation of the package and system configuration on the other cluster nodes.

    * If you are root, this is the root password.
    * If you are using sudo, this is the sudo user password.
    
    The password does not echo on the command line. For example:

    ```
    Vertica Database 9.2. Installation Tool
    Please enter password for root@host01:password
    ```

3. If the dbadmin user, or the user specified in the argument `--dba-user`, does not exist, then the install script prompts for the password for the user. Provide the password. For example:

    ```
    Enter password for new UNIX user dbadmin:password
    Retype new UNIX password for user dbadmin:password
    ```

4. Carefully examine any warnings or failures returned by `install_vertica` and correct the problems.

    For example, insufficient RAM, insufficient network throughput, and too high readahead settings on the file system could cause performance problems later on. Additionally, LANG warnings, if not resolved, can cause database startup to fail and issues with VSQL. The system LANG attributes must be UTF-8 compatible. **Once you fix the problems, re-run the install script.**

5. When installation is successful, disconnect from the **Administration Host**, as instructed by the script. Then, complete the required post-installation steps.

    At this point, root privileges are no longer needed and the database administrator can perform any remaining steps.
    
<small>Source: [www.vertica.com](https://www.vertica.com/docs/9.2.x/HTML/Content/Authoring/InstallationGuide/InstallingVertica.htm?tocpath=Installing%20Vertica%7CInstalling%20Manually%7CInstalling%20Vertica%7C_____0), [insightsoftware.com](https://insightsoftware.com/encyclopedia/hp-vertica/), [Fundamentals of columnar thinking](https://youtu.be/P4OXCNAKMfA)</small>