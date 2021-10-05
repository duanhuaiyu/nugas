'''Python module provides I/O of NuGas through netCDF.
Author: Huaiyu Duan (UNM)
'''

class FlavorHistory:
    '''I/O object of NuGas.
    '''
    def __init__(self, filename, clobber=False, attr={}, dim={}, var={}, var_ini={}, load=False):
        '''Initialization of the I/O object.
        filename : name of the file.
        clobber : whether to overwrite the existing file. Default is False.
        attr : dictionary of global attributes in the format {"name1" : value1, ...}
        dim : dictionary of dimensions in the format {"name1" : length1, ...}. 
        var : dictionary of variables in the format {"name1" : description1, ...}, where the description of a variable is of the format {
            "type" : "data_type", # 'i4', 'i8', 'f4', 'f8', etc
            "dimensions" : ("dimension1", ...),
            "attributes" : {"attribute_name" : value, ...}
        }. Only the first dimension of a variable can be unlimited (length=None).
        var_ini : dictionary of the initial values of the variables in the format {"name1": value1, ...}. If the first dimension of the variable is unlimited, the value is stored to the first row of the record.
        load : whether to load and append to an existing history. Default is False.
        '''
        import netCDF4 as nc
        if load: # append to an existing history
            self.data = nc.Dataset(filename, "r+") # data object
            # find the unlimited dimension
            udim = None
            for name, dim in self.data.dimensions.items():
                if dim.isunlimited():
                    assert not udim, "Only one unlimited dimension is allowed."
                    udim = name
            assert udim, "No unlimited dimension is defined."

            # find the variables with the unlimited dimension
            self.uvar = [] # list of variables with an unlimited dimension
            for name, var in self.data.variables.items():
                if var.dimensions and var.dimensions[0] == udim: # the first dimension of the variable is unlimited
                    self.uvar.append(name)
            assert self.uvar, "No variable with unlimited dimension is defined."
            self.Nt = len(self.data.variables[self.uvar[0]]) # number of records stored

        else:
            self.data = nc.Dataset(filename, "w", clobber=clobber) # data object
            # create attributes
            for name, val in attr.items():
                setattr(self.data, name, val)

            # create dimensions
            udim = None # unlimited dimension
            for name, val in dim.items():
                if not val: # unlimited dimension
                    assert not udim, "Only one unlimited dimension is allowed."
                    udim = name
                self.data.createDimension(name, val)
            assert udim, "No unlimited dimension is defined."

            # create variables
            self.uvar = [] # list of variable names with unlimited dimensions
            for name, desc in var.items():
                if "dimensions" in desc: # array
                    # check unlimited dimensions
                    assert udim not in desc["dimensions"][1:], "Only the first dimension of a variable can be unlimited."
                    if desc["dimensions"][0] == udim: # has unlimited dimension
                        self.uvar.append(name)
                    v = self.data.createVariable(name, desc["type"], desc["dimensions"])
                else: # scalar
                    v = self.data.createVariable(name, desc["type"])
                if "attributes" in desc: # add attributes there there are any
                    for key, val in desc["attributes"]:
                        setattr(v, key, val)
            assert len(self.uvar)>0, "No variable with unlimited dimension is defined."
        
            # set initial values
            self.Nt = 0 # number of records along the unlimited dimension
            for name, val in var_ini.items():
                if name in self.uvar: # variable has an unlimited dimension
                    self.data.variables[name][0] = val # add the first record
                    self.Nt = 1 
                else:
                    self.data.variables[name][:] = val

            # finish up
            self.data.sync()


    def addSnapshot(self, var, flush=False):
        '''Add new records along the unlimited dimension.
        var : dictionary of the initial values of the variables in the format {"name1": value1, ...}.
        flush : whether to flush the data to disk. Default is True.
        '''
        for name, val in var.items():
            assert name in self.uvar, f"Variable {name} does not have an unlimited dimension."
            self.data.variables[name][self.Nt] = val

        self.Nt += 1
        if flush: self.data.sync()

    def flush(self):
        'Flush the data to the disk.'
        self.data.sync()

    def __del__(self):
        try:
            self.data.close()
        except:
            pass