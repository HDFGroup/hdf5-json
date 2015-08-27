import json
import posixpath as pp
from fast_strconcat import StringStore


class PyCode(object):
    """
    Produce Python code that generates HDF5 file based on given JSON input.
    """

    def __init__(self, h5json, fname, ext='h5'):
        """Initialize a PyCode instance.

        :arg dict h5json: HDF5/JSON content.
        :arg str fname: Name of the HDF5 file the generated code will create.
        :arg str ext: Generated HDF5 file's extension. **Without the comma!**
        """
        if len(fname) == 0:
            raise ValueError('Missing file name.')
        if 'root' not in h5json:
            raise KeyError('"root" key not found.')
        self._h5j = h5json
        self._dimensions = list()
        self._dimscales = dict()
        self._fname = '{}.{}'.format(fname, ext)
        self._file_var = 'f'
        self._p = StringStore()

    def get_code(self):
        """Generate Python code for supplied HDF5/JSON."""
        root_uuid = self._h5j["root"]

        self._p.append(
            "import h5py\n"
            "import numpy as np\n\n"
            "# creating file: {1}\n"
            "{0} = h5py.File('{1}', 'w')\n\n"
            .format(self._file_var, self._fname)
        )

        group_json = self._h5j["groups"]
        root_json = group_json[root_uuid]
        self.doAttributes(root_json, '/', self._file_var)
        self.doLinks(self._h5j, root_json, 0)
        self.doDimensions(self._h5j, self._dimensions, self._dimscales,
                          self._file_var)

        return self._p.dump()

    def getNumpyTypename(self, hdf5TypeName, typeClass=None):
        predefined_int_types = {
              'H5T_STD_I8':  'i1',
              'H5T_STD_U8':  'u1',
              'H5T_STD_I16': 'i2',
              'H5T_STD_U16': 'u2',
              'H5T_STD_I32': 'i4',
              'H5T_STD_U32': 'u4',
              'H5T_STD_I64': 'i8',
              'H5T_STD_U64': 'u8'
        }
        predefined_float_types = {
              'H5T_IEEE_F32': 'f4',
              'H5T_IEEE_F64': 'f8'
        }
        if len(hdf5TypeName) < 3:
            raise TypeError("%s: invalid type" % hdf5TypeName)
        endian = '<'  # default endian
        key = hdf5TypeName
        if hdf5TypeName.endswith('LE'):
            key = hdf5TypeName[:-2]
        elif hdf5TypeName.endswith('BE'):
            key = hdf5TypeName[:-2]
            endian = '>'

        if key in predefined_int_types and (typeClass is None or
                                            typeClass == 'H5T_INTEGER'):
            return endian + predefined_int_types[key]
        if key in predefined_float_types and (typeClass is None or
                                              typeClass == 'H5T_FLOAT'):
            return endian + predefined_float_types[key]
        raise TypeError("%s: invalid type" % hdf5TypeName)

    def getBaseDataType(self, typeItem):
        code = "dt = "
        if type(typeItem) == str or type(typeItem) == unicode:
            # should be one of the predefined types
            dtName = self.getNumpyTypename(typeItem)
            code += "np.dtype('{}')\n".format(dtName)
            return code

        if type(typeItem) != dict:
            raise TypeError("{}: invalid type".format(typeItem))

        code += 'np.dtype({})\n'.format(self._dtype(typeItem))

        return code

    def _dtype(self, typeItem, compound=False):
        """Helper function for generating numpy.dtype() code.

        :arg dict typeItem: HDF5/JSON datatype description.
        :arg bool compound: Flag indicating the datatype is part of a compound
            datatype.
        """
        typeClass = typeItem['class']
        shape = ''
        if 'dims' in typeItem:
            shp_key = 'dims'
        else:
            shp_key = 'shape'
        if shp_key in typeItem:
            dims = None
            if type(typeItem[shp_key]) == int:
                dims = (typeItem[shp_key],)  # make into a tuple
            elif type(typeItem[shp_key]) not in (list, tuple):
                raise TypeError("expected list or integer for %s" % shp_key)
            else:
                dims = typeItem[shp_key]
            shape = str(tuple(dims))

        code = ''
        if typeClass == 'H5T_INTEGER':
            if 'base' not in typeItem:
                raise KeyError("'base' not provided")
            baseType = self.getNumpyTypename(typeItem['base'],
                                             typeClass='H5T_INTEGER')
            code += "'{}{}'".format(shape, baseType)
        elif typeClass == 'H5T_FLOAT':
            if 'base' not in typeItem:
                raise KeyError("'base' not provided")
            baseType = self.getNumpyTypename(typeItem['base'],
                                             typeClass='H5T_FLOAT')
            code += "'{}{}'".format(shape, baseType)
        elif typeClass == 'H5T_STRING':
            if 'length' not in typeItem:
                raise KeyError("'length' not provided")
            if 'charSet' not in typeItem:
                raise KeyError("'charSet' not provided")

            if typeItem['length'] == 'H5T_VARIABLE':
                if shape:
                    raise TypeError(
                        "ArrayType is not supported for variable len types")
                if typeItem['charSet'] == 'H5T_CSET_ASCII':
                    code += "h5py.special_dtype(vlen=str)"
                elif typeItem['charSet'] == 'H5T_CSET_UTF8':
                    code += "h5py.special_dtype(vlen=unicode)"
                else:
                    raise TypeError("unexpected 'charSet' value")
            else:
                # fixed size ascii string
                nStrSize = typeItem['length']
                if type(nStrSize) != int:
                    raise TypeError("expecting integer value for 'length'")
                code += "'{}S{}'".format(shape, nStrSize)
        elif typeClass == 'H5T_VLEN':
            if shape:
                raise TypeError(
                    "ArrayType is not supported for variable len types")
            if 'base' not in typeItem:
                raise KeyError("'base' not provided")
            vlenBaseType = typeItem['base']
            baseType = self.getNumpyTypename(vlenBaseType['base'],
                                             typeClass=vlenBaseType['class'])
            code += "h5py.special_dtype(vlen=np.dtype('" + baseType + "'))"
        elif typeClass == 'H5T_OPAQUE':
            if shape:
                raise TypeError(
                    "Opaque Type is not supported for variable len types")
            if 'size' not in typeItem:
                raise KeyError("'size' not provided")
            nSize = int(typeItem['size'])
            if nSize <= 0:
                raise TypeError("'size' must be non-negative")
            code += "'V{}'".format(nSize)
        elif typeClass == 'H5T_ARRAY':
            if not shape:
                raise KeyError("'shape' must be provided for array types")
            if 'base' not in typeItem:
                raise KeyError("'base' not provided")
            baseType = self._dtype(typeItem['base'])
            if type(baseType) not in (str, unicode):
                raise TypeError(
                    "Array type is only supported for predefined base types")
            # should be one of the predefined types
            code += "{1}, {0}".format(shape, baseType)
            if not compound:
                code = "({})".format(code)
        elif typeClass == 'H5T_COMPOUND':
            if 'fields' not in typeItem:
                raise KeyError("'fields' must be provided for compound types")
            if type(typeItem['fields']) is not list:
                raise TypeError("compound 'fields' value must be a list")
            dt_arg = list()
            for fld in typeItem['fields']:
                dt_arg.append(
                    "('{}', {})".format(
                        fld['name'], self._dtype(fld['type'], compound=True)))
            code = '[' + ', '.join(dt_arg) + ']'
        else:
            raise TypeError("%s: Invalid type class" % typeClass)

        return code

    def valueToString(self, attr_json):
        value = attr_json["value"]
        return json.dumps(value)

    def doAttribute(self, attr_json, parent_var):
        if attr_json['type']['class'] == 'H5T_STRING':
            self._p.append(parent_var + ".attrs['" + attr_json['name']
                           + "'] = " + self.valueToString(attr_json) + '\n')
        else:
            dt = 'np.dtype({})'.format(self._dtype(attr_json["type"]))
            shape_json = attr_json["shape"]
            if shape_json['class'] == 'H5S_SIMPLE':
                shape = self._dims2str(shape_json['dims'])
            elif shape_json['class'] == 'H5S_SCALAR':
                shape = ', ()'
            else:
                raise NotImplementedError('{}: Dataspace not supported yet'
                                          .format(shape_json['class']))
            self._p.append("{}.attrs.create('{}', {}{}, dtype={})\n"
                           .format(parent_var, attr_json['name'],
                                   self.valueToString(attr_json), shape, dt))

    def getObjectName(self, obj_json, obj_title):
        name = obj_title
        if "alias" in obj_json:
            alias = obj_json["alias"]
            try:
                name = alias[0]
            except (TypeError, IndexError):
                name = alias
        return name

    def group_var_name(self, level, next=False):
        """Determine the name of the group variable based on its HDF5 tree
        depth.

        :arg int level: HDF5 tree depth level (root = 0).
        :arg bool next: Provide group variable for the next level, i.e.
            subgroup.
        """
        grp_fmt = 'grp_{:d}'
        if next:
            return grp_fmt.format(level+1)
        if level == 0:
            pvar = self._file_var
        else:
            pvar = grp_fmt.format(level)
        return pvar

    def doAttributes(self, obj_json, obj_name, parent_var, is_dimscale=False,
                     is_dimension=False):
        if len(obj_json.get('attributes', [])) == 0:
            return
        attrs_json = obj_json["attributes"]

        self._p.append("# Creating attributes for {}\n".format(obj_name))
        for attr_json in attrs_json:
            if is_dimscale and attr_json['name'] in ('CLASS', 'REFERENCE_LIST',
                                                     'NAME'):
                    continue
            if is_dimension and attr_json['name'] == 'DIMENSION_LIST':
                continue
            self.doAttribute(attr_json, parent_var)

    def doGroup(self, h5json, group_id, group_name, level):
        parent_var = self.group_var_name(level)
        groups = h5json["groups"]
        group_json = groups[group_id]
        group_path = self.getObjectName(group_json, group_name)
        self._p.append("\n\n# Group: {}\n".format(group_path))
        group_var = self.group_var_name(level, next=True)
        self._p.append("{0} = {1}.create_group('{2}')\n"
                       .format(group_var, parent_var, group_name))
        self.doAttributes(group_json, group_path, group_var)
        self.doLinks(h5json, group_json, level+1)

    def _dims2str(self, dims, kwd=''):
        """Convert a list of integers to a string representing a dimension tuple.

        :arg list dims: A dimension list.
        :arg str kwd: Optional keyword string.
        """
        dims = [str(d) if d != 'H5S_UNLIMITED' else 'None' for d in dims]
        if len(dims) == 1:
            # Produce correct tuple when dim rank is 1...
            dims.append('')
        if kwd:
            kwd += '='
        return ', {}({})'.format(kwd, ','.join(dims))

    def doDataset(self, h5json, dset_id, dset_name, parent_var):
        datasets = h5json["datasets"]
        dset_json = datasets[dset_id]
        dset_path = dset_json.get('alias', [dset_name])[0]
        self._p.append("\n# Dataset: {}\n".format(dset_path))
        dset_var = 'dset'
        try:
            dtLine = self.getBaseDataType(dset_json["type"])  # "dt = ..."
            self._p.append(dtLine)

            shape = ''
            maxshape = ''
            chunks = ''
            flt = ''
            cp = dset_json.get('creationProperties', {})
            if "shape" in dset_json:
                shape_json = dset_json["shape"]
                if shape_json["class"] == "H5S_SIMPLE":
                    shape = self._dims2str(shape_json["dims"])

                    if 'maxdims' in shape_json:
                        maxshape = self._dims2str(shape_json['maxdims'],
                                                  kwd='maxshape')

                    layout = cp.get('layout', {}).get('class',
                                                      'H5D_CONTIGUOUS')
                    if layout == 'H5D_CHUNKED':
                        chunks = self._dims2str(cp['layout']['dims'],
                                                kwd='chunks')

                        filters = cp.get('filters', [])
                        for f in filters:
                            if f['class'] == 'H5Z_FILTER_DEFLATE':
                                flt += (
                                    ", compression='gzip', "
                                    "compression_opts={:d}").format(f['level'])
                            elif f['class'] == 'H5Z_FILTER_FLETCHER32':
                                flt += ', fletcher32=True'
                            elif f['class'] == 'H5Z_FILTER_SHUFFLE':
                                flt += ', shuffle=True'
                            elif f['class'] == 'H5Z_FILTER_SCALEOFFSET':
                                flt += (', scaleoffset={:d}'
                                        .format(f['scaleOffset']))
                            else:
                                raise NotImplementedError(
                                    '{}: Filter not supported yet'
                                    .format(f['class']))

                elif shape_json['class'] == 'H5S_SCALAR':
                    shape = ', ()'

            if 'fillValue' in cp:
                if dset_json['type']['class'] == 'H5T_STRING':
                    fv = ", fillvalue='{}'".format(cp['fillValue'])
                else:
                    fv = ', fillvalue={}'.format(cp['fillValue'])
            else:
                fv = ''

            code_line = ("{} = {}.create_dataset('{}'{}{}{}{}{}"
                         ", dtype=dt)\n").format(dset_var, parent_var,
                                                 dset_name, shape, maxshape,
                                                 chunks, fv, flt)
            self._p.append(code_line)
            self._p.append("# initialize dataset values here\n")

            dscale = self._is_dimscale(dset_json.get('attributes', []))
            if dscale:
                # Find out dimension scale's name, if available...
                for a in dset_json.get('attributes', []):
                    if a['name'] == 'NAME':
                        ds_name = a['value']
                        break
                else:
                    ds_name = None
                self._dimscales.update({dset_id: ds_name})
            dim = self._is_dimlist(dset_json.get('attributes', []))
            if dim:
                self._dimensions.append(dset_id)

            self.doAttributes(dset_json, dset_path, dset_var,
                              is_dimscale=dscale, is_dimension=dim)
        except Exception as e:
            raise type(e)('{}: {}'.format(dset_path, str(e)))

    def doLink(self, h5json, link_json, level):
        parent_var = self.group_var_name(level)

        if link_json["class"] == "H5L_TYPE_EXTERNAL":
            self._p.append("{0}['{1}'] = h5py.ExternalLink('{2}', '{3}')\n"
                           .format(parent_var, link_json["title"],
                                   link_json["file"], link_json["h5path"])
                           )

        elif link_json["class"] == "H5L_TYPE_SOFT":
            self._p.append("{0}['{1}'] = h5py.SoftLink('{2}')\n"
                           .format(parent_var, link_json["title"],
                                   link_json["h5path"])
                           )

        elif link_json["class"] == "H5L_TYPE_HARD":
            if link_json["collection"] == "groups":
                self.doGroup(h5json, link_json["id"], link_json["title"],
                             level)
            elif link_json["collection"] == "datasets":
                self.doDataset(h5json, link_json["id"], link_json["title"],
                               parent_var)
            elif link_json["collection"] == "datatypes":
                raise NotImplementedError(
                    'committed datatypes not supported yet')
            else:
                raise Exception(
                    "unexpected collection name: " + link_json["collection"])

        elif link_json["class"] == "H5L_TYPE_UDLINK":
            self._p.append("# ignoring user defined link: '{0}'\n"
                           .format(link_json["title"]))

        else:
            raise Exception("unexpected link type: " + link_json["class"])

    def doLinks(self, h5json, group_json, level):
        links = group_json.get("links", [])
        for link in links:
            self.doLink(h5json, link, level)

    def _is_dimscale(self, attrs):
        """Check if the dataset is a dimension scale.

        :arg list attrs: All dataset's attributes.
        """
        # Check if REFERENCE_LIST attribute is present...
        ref_list = any(a['name'] == 'REFERENCE_LIST'
                       and a['type']['class'] == 'H5T_COMPOUND'
                       for a in attrs)

        # Check if CLASS attribute is present...
        cls_ = any(a['name'] == 'CLASS'and a['value'] == 'DIMENSION_SCALE'
                   for a in attrs)

        if ref_list and cls_:
            return True
        else:
            return False

    def _is_dimlist(self, attrs):
        """Check if the dataset has dimension scales attached.

        :arg list attrs: All dataset's attributes.
        """
        # Check if DIMENSION_LIST attribute is present...
        dim_list = any(a['name'] == 'DIMENSION_LIST'
                       and a['type']['class'] == 'H5T_VLEN'
                       for a in attrs)

        return True if dim_list else False

    def _dset_paths(self, h5json):
        grps = [{'id': h5json['root'], 'path': '/'}]
        dsets = {}

        def get_hard_links(grpjson, collection):
            links = list()
            for l in grpjson.get('links', []):
                if (l['class'] == 'H5L_TYPE_HARD'
                        and l['collection'] == collection):
                    links.append(l)
            return links

        def tree_walker(ginfo):
            dlinks = get_hard_links(h5json['groups'][ginfo['id']], 'datasets')
            for dl in dlinks:
                dsets.update({dl['id']: pp.join(ginfo['path'], dl['title'])})

            glinks = get_hard_links(h5json['groups'][ginfo['id']], 'groups')
            chld_grps = list()
            for gl in glinks:
                chld_grps.append({'id': gl['id'],
                                  'path': pp.join(ginfo['path'], gl['title'])})
            grps.extend(chld_grps)
            for cg in chld_grps:
                tree_walker(cg)

        tree_walker(grps[0])

        return dsets

    def doDimensions(self, h5json, dimensions, dimscales, parent_var):
        if len(dimensions) == 0:
            return

        # Generate HDF5 paths for all datasets...
        dset_path = self._dset_paths(h5json)

        self._p.append('\n\n'
                       '#\n'
                       '# Adding dimensions\n'
                       '#\n'
                       '\n')

        self._p.append('# Creating dimension scales\n')
        for dsid, name in dimscales.iteritems():
            if dimscales[dsid]:
                name = ", '{}'". format(dimscales[dsid])
            else:
                name = ''
            self._p.append("h5py.h5ds.set_scale({}['{}'].id{})\n"
                           .format(parent_var, dset_path[dsid], name))

        for dsid in dimensions:
            dsid_path = dset_path[dsid]
            self._p.append("\n# Attaching dimension scales to dataset: {}\n"
                           .format(dsid_path))
            for attr in h5json['datasets'][dsid].get('attributes', []):
                if attr['name'] == 'DIMENSION_LIST':
                    dim_list = attr['value']
                    break
            else:
                raise ValueError('%s: DIMENSION_LIST attribute not found'
                                 % dsid_path)

            for idx, ds in enumerate(dim_list):
                for d in ds:
                    did = pp.split(d)[-1]
                    did_path = dset_path[did]
                    self._p.append(
                        "{}['{}'].dims[{:d}].attach_scale({}['{}'])\n"
                        .format(parent_var, dsid_path, idx, parent_var,
                                did_path)
                    )
