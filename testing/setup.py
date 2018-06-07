from cx_Freeze import setup, Executable

import os



base = None

#os.environ['TCL_LIBRARY'] = r'\Users/yuanpu/local/python3/anaconda3/tcl/tcl8.6'
#os.environ['TK_LIBRARY'] = r'\Users/yuanpu/local/python3/anaconda3/tcl/tk8.6'

os.environ['TCL_LIBRARY'] = r'C:\ProgramData\Anaconda3\tcl\tcl8.6'
os.environ['TK_LIBRARY'] = r'C:\ProgramData\Anaconda3\tcl\tk8.6'


executables = [Executable("testing_exe.py", base=base)]

#build_exe_options = {"packages": ["errno", "os", "re", "stat", "subprocess","collections", "pprint","shutil", "humanize","pycallgraph",'scipy','numpy.core._methods', 'numpy.lib.format']}


packages = [ "os", "re", "stat","scipy"]  # idan

options = {

 'build_exe': {

     'packages': packages,

     'excludes': ['scipy.spatial.cKDTree'],

      'includes': ['scipy.spatial.ckdtree'],

 },



}



setup(

 name="he",

 options=options,

 version="0.1",

 description='123',

 executables=executables

)

