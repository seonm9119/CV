The SIFT function can only be used in openv 3.4.2.16 version.
The openv 3.4.2.16 version can be installed in Python 3.7.
In order to execute the program, the version of each module must be matched.

In order to use the calibrated_fivepoint.m given to the task, this program interlocked matlab with python. 
Therefore, matlab.engine should be installed.
In order to use the matlab.engine in python 3.7, MATLAB R2019a or higher must be installed.

Installation matla.engine
https://kr.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html
matlab.engine must be installed in the program execution environment variable.


[Requirements]
python=3.7
opencv-python=3.4.2.16
opencv-contrib-python=3.4.2.16
matlab-kernel=0.16.11
matlab >=R2019a (Program execution environment : R2021b )


The given SavePLY.m file was replaced with python code and used. (SavePLY.py)


