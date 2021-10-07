import setuptools, sys, os

# module info
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
ext_par = {
    'name' : 'nugas',
    'version' : '1.1',
    'description' : 'Python package that computes flavor oscillations in dense neutrino gases.',
    'url' : 'https://github.com/NuCO-UNM/nugas',
    'author' : 'Huaiyu Duan',
    'author_email' : 'duan@unm.edu',
    'license' : 'MIT',
    'long_description' : long_description,
    'long_description_content_type' : "text/markdown",
    'classifiers' : [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    'package_dir' : {"": "src"},
    'packages' : setuptools.find_packages(where="src"),
    'python_requires' : ">=3.7",
    'install_requires' : [ 'numpy', 'scipy' ],
    'zip_safe' : False
}
    

try: # try to build the c++ extensions
    from pybind11.setup_helpers import Pybind11Extension, build_ext, intree_extensions

    # list of extensions
    ext_modules = intree_extensions(
        ["src/nugas/misc/pdz_c.cpp",
        "src/nugas/f2e0d1a/eom_c.cpp", 
        "src/nugas/f2e0d1a/lax42.cpp"]
    )

    # additional flags depending on the compiler
    if sys.platform.startswith("darwin"): # MacOS, assuming clang
        extra_compile_args = ['-Xpreprocessor', '-fopenmp']
        extra_link_args = []
    elif 'icpc' in os.environ['CXX']:
        extra_compile_args = ['-xHost', '-qopenmp']
        extra_link_args = ['-qopenmp']
    else: # assuming g++
        extra_compile_args = ['-fopenmp']
        extra_link_args = ['-lgomp'] 
    for m in ext_modules:
        m._add_cflags(extra_compile_args)
        m._add_ldflags(extra_link_args)

    setuptools.setup(
        ext_modules=ext_modules,
        cmdclass={"build_ext": build_ext},
        **ext_par
    )
except:
    setuptools.setup(**ext_par)
    print("** FAILED TO BUILD THE C++ EXTENSIONS. INSTALLED THE PURE PYTHON PACKAGE INSTEAD. **", file=sys.stderr)
