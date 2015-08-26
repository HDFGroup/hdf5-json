from string import Template
import posixpath as pp
from util import StringStore


class MCode(object):
    """Generate MATLAB source code (as an .m file) that produces a template
    HDF5 file."""

    def __init__(self, tinfo, fname, ext='h5'):
        """Initialize an MCode instance.

        :arg dict tinfo: Template content information.
        :arg str fname: Name of the file to use when generating the source
            code.
        :arg str ext: HDF5 file extension. **Without the comma!**
        """
        if len(fname) == 0:
            raise ValueError('Missing file name.')
        if 'root' not in tinfo:
            raise KeyError('"root" key not found.')

        # Helper variables...
        self._d = tinfo
        self._root = self._d['root']
        self._fname = fname + '.' + ext
        self._dset_path = dict()  # map dataset ID to HDF5 paths
        self._dimlist = []  # List of dataset IDs that have dimscales attached

        # Variable for the generated source code...
        self._m = StringStore()

    def _get_hard_links(self, gid, collection):
        links = list()
        for l in self._d['groups'][gid].get('links', []):
            if l['class'] == 'H5L_TYPE_HARD' and l['collection'] == collection:
                links.append(l)
        return links

    def _order_groups(self):
        pid = [{'id': self._root, 'path': '/'}]

        def tree_walker(ginfo):
            glinks = self._get_hard_links(ginfo['id'], 'groups')
            chld_grps = list()
            for gl in glinks:
                chld_grps.append({'id': gl['id'],
                                  'path': pp.join(ginfo['path'], gl['title'])})
            pid.extend(chld_grps)
            for cg in chld_grps:
                tree_walker(cg)

        tree_walker(pid[0])

        return pid

    def matlab_dtype(self, h5type):
        """Find appropriate MATLAB numerical data type for HDF5 predefined
        datatype.

        :arg str h5type: HDF5 predefined datatype.
        :return: MATLAB data type.
        :rtype: str
        """
        conv_map = {
            'H5T_STD_I8': 'int8',
            'H5T_STD_U8': 'uint8',
            'H5T_STD_I16': 'int16',
            'H5T_STD_U16': 'uint16',
            'H5T_STD_I32': 'int32',
            'H5T_STD_U32': 'uint32',
            'H5T_STD_I64': 'int64',
            'H5T_STD_U64': 'uint64',
            'H5T_IEEE_F32': 'single',
            'H5T_IEEE_F64': 'double'
        }
        try:
            return conv_map[h5type[:-2]]
        except KeyError:
            raise ValueError('%s: Invalid predefined datatype' % h5type)

    def _create_file(self):
        """Code to create an HDF5 file."""
        tmplt = Template(
            "% Create the HDF5 file. It will overwrite any file with same "
            "name.\n"
            "fname = '$filename';\n"
            "fcpl = H5P.create('H5P_FILE_CREATE');\n"
            "fapl = H5P.create('H5P_FILE_ACCESS');\n"
            "fid = H5F.create(fname, 'H5F_ACC_TRUNC', fcpl, fapl);\n"
        )
        vars = {'filename': self._fname}
        self._m.append(tmplt.substitute(vars))

    def _close_file(self):
        """Code to close an HDF5 file."""
        tmplt = (
            "\n\n"
            "% Close the HDF5 file.\n"
            "H5F.close(fid);\n"
            "% Template file is ready!\n"
        )
        self._m.append(tmplt)

    def _dims2str(self, dims):
        """Stringify dimension list with support for the unlimited size.

        :arg list dims: Dimension size list.
        """
        dim_str = []
        for d in dims:
            if d == 'H5S_UNLIMITED':
                # Unlimited dimension...
                dim_str.append("H5ML.get_constant_value('H5S_UNLIMITED')")
            else:
                dim_str.append('{:d}'.format(d))
        return '[{}]'.format(', '.join(dim_str))

    def _dspace(self, shape):
        """Generate dataspace code.

        :arg dict shape: HDF5/JSON shape information.
        """
        if shape['class'] == 'H5S_SCALAR':
            return "sid = H5S.create('H5S_SCALAR');\n"
        elif shape['class'] == 'H5S_SIMPLE':
            rank = len(shape['dims'])
            if rank == 1:
                tmplt = Template(
                    "sid = H5S.create_simple($rank, $dims, $maxdims);\n"
                )
            else:
                tmplt = Template(
                    "sid = H5S.create_simple($rank, fliplr($dims), "
                    "fliplr($maxdims));\n"
                )
            vars = {'dims': self._dims2str(shape['dims']),
                    'maxdims': self._dims2str(shape['maxdims']),
                    'rank': rank}
            return tmplt.substitute(vars)
        else:
            raise NotImplementedError('%s: Not supported' % shape['class'])

    def _dtype(self, t, var='tid'):
        """Generate datatype code.

        :arg dict t: HDF5/JSON datatype information.
        :arg str var: Default name of the datatype variable.
        """
        tcls = t['class']
        if tcls == 'H5T_COMPOUND':
            tmplt = ''

            # Go over each compound field...
            field_cnt = 0
            for f in t['fields']:
                field_cnt += 1
                dt = "field_tid(%d)" % field_cnt
                tmplt += self._dtype(f['type'], var=dt)
                tmplt += "field_size(%d) = H5T.get_size(%s);\n" % (field_cnt,
                                                                   dt)

            # Compute field byte offsets...
            num_fields = len(t['fields'])
            tmplt += (
                "field_offset = [0 cumsum(field_size(1:%d))];\n"
            ) % (num_fields - 1)

            # Create the compound datatype...
            tmplt += "tid = H5T.create('H5T_COMPOUND', sum(field_size));\n"
            for n in xrange(num_fields):
                tmplt += ("H5T.insert(tid, '%s', field_offset(%d), "
                          "field_tid(%d));\n") % (t['fields'][n]['name'],
                                                  n + 1, n + 1)

            # Close field datatypes...
            for n in xrange(num_fields):
                tmplt += "H5T.close(field_tid(%d));\n" % (n + 1)

            return tmplt

        elif tcls == 'H5T_VLEN':
            if t['base']['class'] == 'H5T_STRING':
                base_type = 'H5T_C_S1'
                if t['base']['length'] not in (1, 'H5T_VARIABLE'):
                    raise NotImplementedError(
                        'MATLAB only allows vlen strings of variable or '
                        'fixed length of 1.')
            elif t['base']['class'] == 'H5T_REFERENCE':
                raise NotImplementedError(
                    'MATLAB does not support H5T_REFERENCE for vlen datatype')
            else:
                base_type = t['base']['base']
            return "tid = H5T.vlen_create('%s');\n" % base_type

        elif tcls == 'H5T_ARRAY':
            tmplt = Template(
                "${base}"
                "$var = H5T.array_create(base_tid, fliplr($dims));\n"
                "H5T.close(base_tid);\n"
            )
            return tmplt.substitute(
                {'base': self._dtype(t['base'], var='base_tid'),
                 'dims': t['dims'],
                 'var': var})

        else:
            return var + " = " + self._atomic_dtype(t, var=var)

    def _atomic_dtype(self, t, var='tid'):
        """Handle HDF5 atomic datatypes.

        :arg dict t: HDF5/JSON datatype information.
        "arg str tid: Default name of the datatype variable.
        """
        tcls = t['class']
        if tcls == 'H5T_STRING':
            tmplt = Template(
                "H5T.copy('H5T_C_S1');\n"
                "H5T.set_size($var, $length);\n"
                "H5T.set_strpad($var,'$strpad');\n"
                "H5T.set_cset($var, H5ML.get_constant_value('$cset'));\n"
            )
            if isinstance(t['length'], basestring):
                length = "'%s'" % t['length']
            else:
                length = t['length']
            return tmplt.substitute({'length': length,
                                     'strpad': t['strPad'],
                                     'cset': t['charSet'],
                                     'var': var})

        elif tcls in ('H5T_FLOAT', 'H5T_INTEGER'):
            return "H5T.copy('%s');\n" % t['base']

        elif tcls == 'H5T_REFERENCE':
            return "H5T.copy('%s');\n" % t['base']

        else:
            raise NotImplementedError('%s: Datatype not supported'
                                      % t['class'])

    def _create_attr(self, attr, locid, dimscale=False):
        """Generate code for one attribute of the ``locid`` object.

        :arg dict attr: Attribute information.
        :arg str locid: Attribute's parent variable name.
        :arg bool dimscale: Indicates attribute's parent is a dimension scale.
        """
        if dimscale and attr['name'] in ('REFERENCE_LIST', 'DIMENSION_LIST',
                                         'NAME', 'CLASS'):
            return

        dataspace = self._dspace(attr['shape'])
        datatype = self._dtype(attr['type'])

        tmplt = Template(
            "\n% Attribute: $name\n"
            "${datatype}"
            "${dataspace}"
            "acpl = H5P.create('H5P_ATTRIBUTE_CREATE');\n"
            "aid = H5A.create($lid, '$name', tid, sid, acpl, 'H5P_DEFAULT');\n"
            "${value}"
            "H5T.close(tid);\n"
            "H5S.close(sid);\n"
            "H5A.close(aid);\n"
        )
        if attr['type']['class'] == 'H5T_STRING':
            def prep_vals(vals):
                values = list()
                with_sprintf = False
                for v in vals:
                    temp = v.encode('ascii')
                    if "'" in temp:
                        temp = temp.replace("'", "''")
                        with_sprintf = True
                    if '%' in temp:
                        temp = temp.replace('%', '%%')
                        with_sprintf = True
                    if '\n' in temp:
                        temp = temp.replace('\n', '\\n')
                        with_sprintf = True
                    if with_sprintf:
                        values.append("sprintf('{}')".format(temp))
                    else:
                        values.append("'{}'".format(temp))
                return values

            if attr['shape']['class'] == 'H5S_SCALAR':
                value = prep_vals([attr['value']])[0]
            else:
                if len(attr['shape']['dims']) > 1:
                    raise NotImplementedError(
                        'Rank > 1 for string data not supported')
                value = prep_vals(attr['value'])

            if attr['type']['length'] == 'H5T_VARIABLE':
                val_str = ('{'
                           + ', '.join(value)
                           + '}')
            else:
                # Left-justified, fixed-length string format...
                # fmt = '{{:<{0:d}.{0:d}}}'.format(attr['type']['length'])

                if attr['shape']['class'] == 'H5S_SCALAR':
                    # val_str = "'%s'" % fmt.format(value)
                    val_str = value
                else:
                    # for i in xrange(len(value)):
                    #     value[i] = fmt.format(value[i])
                    val_str = (
                        '[' + '; '.join(value)
                        + "]'")
        else:
            val_str = attr['value']
        val = Template(
            "H5A.write(aid, 'H5ML_DEFAULT', $value);\n"
        ).substitute({'value': val_str})

        vars = {'lid': locid,
                'name': attr['name'],
                'datatype': datatype,
                'dataspace': dataspace,
                'value': val}
        self._m.append(tmplt.substitute(vars))

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

    def _dimscale(self, attrs, locid):
        """Generate the code that sets a dimension scale.

        :arg list attrs: All dataset's attributes.
        :arg str locid: MATLAB variable name of the attributes' parent object.
        """
        # Check if NAME attribute is present...
        for a in attrs:
            if a['name'] == 'NAME'and a['type']['class'] == 'H5T_STRING':
                scale_name = "'%s'" % a['value']
                break
        else:
            scale_name = '[]'

        return "H5DS.set_scale(%s, %s);\n" % (locid, scale_name)

    def _is_dimlist(self, attrs):
        """Check if the dataset has dimension scales attached.

        :arg list attrs: All dataset's attributes.
        """
        # Check if DIMENSION_LIST attribute is present...
        dim_list = any(a['name'] == 'DIMENSION_LIST'
                       and a['type']['class'] == 'H5T_VLEN'
                       for a in attrs)

        return True if dim_list else False

    def _is_dimscale_related(self, attrs):
        """Check if the attributes of a dataset indicate it is related to
        dimension scales.

        :arg list attrs: All dataset's attributes.
        """
        dimscale = self._is_dimscale(attrs)
        dimlist = self._is_dimlist(attrs)

        if dimscale or dimlist:
            return True
        else:
            return False

    def _create_attrs(self, attrs, locid, dimscale=False):
        """Generate code for all the attributes of the ``locid`` object.

        :arg dict attrs: HDF5/JSON information about the attributes of
            the parent ``locid`` object.
        :arg str locid: MATLAB variable name of the attributes' parent object.
        :arg bool dimscale: Indicates whether the attributes belong to a
            dimension scale.
        """
        for a in attrs:
            self._create_attr(a, locid, dimscale=dimscale)

    def _create_dset(self, id, name, ds, locid):
        """Generate code for one dataset of the ``locid`` group.

        : arg str id: Dataset's identifier.
        :arg str name: Dataset's name.
        :arg dict ds: Dataset information.
        :arg str locid: Varable name of the dataset's parent group.
        """
        dataspace = self._dspace(ds['shape'])
        datatype = self._dtype(ds['type'])

        tmplt = Template(
            "\n"
            "% Dataset: $name\n"
            "${datatype}"
            "${dataspace}"
            "${dcpl}"
            "dsid = H5D.create($locid, '$name', tid, sid, '$plist', dcpl, "
            "'$plist');\n"
            "H5S.close(sid);\n"
            "H5T.close(tid);\n"
            "${dimscale}"
        )

        # Dataset creation property list...
        dcpl = "dcpl = H5P.create('H5P_DATASET_CREATE');\n"

        # Layout...
        layout = ds.get('creationProperties', {}).get('layout', {})
        if layout.get('class', 'H5D_CONTIGUOUS') == 'H5D_CONTIGUOUS':
            pass
        elif layout['class'] == 'H5D_COMPACT':
            dcpl = ("dcpl = H5P.create('H5P_DATASET_CREATE');\n"
                    "H5P.set_layout(dcpl, 'H5D_COMPACT');\n")
        elif layout['class'] == 'H5D_CHUNKED':
            if len(layout['dims']) == 1:
                chunks = layout['dims']
            else:
                chunks = 'fliplr(%s)' % layout['dims']
            dcpl = Template(
                "dcpl = H5P.create('H5P_DATASET_CREATE');\n"
                "H5P.set_layout(dcpl, H5ML.get_constant_value('H5D_CHUNKED'));"
                "\n"
                "H5P.set_chunk(dcpl, $chunks);\n"
            ).substitute({'chunks': chunks})
        else:
            raise ValueError('%s: Invalid layout class'
                             % layout['class'])

        # Filters...
        filters = ds.get('creationProperties', {}).get('filters', [])
        for f in filters:
            if f['class'] == 'H5Z_FILTER_DEFLATE':
                dcpl += "H5P.set_deflate(dcpl, %s);\n" % f['level']
            elif f['class'] == 'H5Z_FILTER_FLETCHER32':
                dcpl += "H5P.set_fletcher32(dcpl);\n"
            elif f['class'] == 'H5Z_FILTER_SHUFFLE':
                dcpl += "H5P.set_shuffle(dcpl);\n"
            elif f['class'] == 'H5Z_FILTER_SCALEOFFSET':
                dcpl += ("H5P.set_scaleoffset(dcpl, '%s', %d);\n"
                         % (f['scaleType'], f['scaleOffset']))
            elif f['class'] == 'H5Z_FILTER_NBIT':
                dcpl += "H5P.set_nbit(dcpl);\n"
            else:
                raise NotImplementedError('%s: Filter not supported yet'
                                          % f['class'])

        # Fill value...
        fv = ds.get('creationProperties', {}).get('fillValue', None)
        if fv:
            if type(fv) is list:
                raise NotImplementedError(
                    'Non-scalar fill value not supported yet')
            else:
                # Use dataset's datatype for fill value...
                if ds['type']['class'] == 'H5T_STRING':
                    fv = "'%s'" % fv
                else:
                    fv = str(fv)
                    # Remove an "L" suffix if present...
                    if fv[-1] == 'L':
                        fv = fv[:-1]
                    fv = '{}({})'.format(self.matlab_dtype(ds['type']['base']),
                                         fv)
                dcpl += "H5P.set_fill_value(dcpl, tid, %s);\n" % fv

        attrs = ds.get('attributes', [])
        is_dimscale = self._is_dimscale(attrs)
        if is_dimscale:
            dimscale = self._dimscale(attrs, 'dsid')
        else:
            dimscale = str()

        vars = {'locid': locid,
                'name': name,
                'datatype': datatype,
                'dataspace': dataspace,
                'plist': 'H5P_DEFAULT',
                'dcpl': dcpl,
                'dimscale': dimscale}
        self._m.append(tmplt.substitute(vars))

        is_dimlist = self._is_dimlist(attrs)
        if is_dimlist:
            self._dimlist.append(id)
        self._create_attrs(attrs, 'dsid',
                           dimscale=(is_dimscale or is_dimlist))

        self._m.append("H5D.close(dsid);\n")

    def _create_dsets(self, dsets, locid, path):
        """Generate code for all the datasets of the ``locid`` object.

        :arg dict dsets: HDF5/JSON information about the datasets of
            the parent ``locid`` object.
        :arg str locid: MATLAB variable name of the datasets' parent object.
        :arg str path: HDF5 path of the datasets' parent object.
        """
        for d in dsets:
            # Record the full HDF5 path to the dataset...
            self._dset_path[d['id']] = pp.join(path, d['title'])
            # Generate dataset code...
            self._create_dset(d['id'], d['title'],
                              self._d['datasets'][d['id']], locid)

    def _create_group(self, g):
        """Code for all group content.

        :arg dict g: Group id and full name.
        """
        grpid = g['id']
        path = g['path']

        if grpid == self._root:
            locid = 'fid'
        else:
            tmplt = Template(
                "\n\n"
                "%\n"
                "% Group: $path\n"
                "%\n"
                "gid = H5G.create(fid, '$path', '$plist', '$plist', '$plist');"
                "\n"
            )
            vars = {'path': path, 'plist': 'H5P_DEFAULT'}
            self._m.append(tmplt.substitute(vars))
            locid = 'gid'

        self._create_attrs(self._d['groups'][grpid].get('attributes', []),
                           locid)
        self._create_dsets(self._get_hard_links(grpid, 'datasets'), locid,
                           path)

        if grpid != self._root:
            self._m.append("H5G.close(gid);\n")

    def _dimensions(self):
        """Generate code connecting dimension scales and their datasets."""
        # List for storing dimension scale's IDs...
        dscales = list()

        self._m.append(
            "\n\n"
            "%\n"
            "% Datasets and their dimension scales\n"
            "%\n"
        )
        for dset_id in self._dimlist:
            tmp = Template(
                "\n% Dataset with dimension scales: $name\n"
                "dset_id = H5D.open(fid, '$name', 'H5P_DEFAULT');\n"
            ).substitute({'name': self._dset_path[dset_id]})
            self._m.append(tmp)

            dset = self._d['datasets'][dset_id]

            # Get DIMENSION_LIST attribute value...
            dims = next(a['value'] for a in dset['attributes']
                        if a['name'] == 'DIMENSION_LIST')

            # Iterate over dataset's dimension scales in reversed order
            # (because of using fliplr() when defining dataset's shape)...
            for index, dimscales in enumerate(reversed(dims)):
                for n, ds in enumerate(dimscales):
                    ds_id = pp.basename(ds)
                    dscales.append(ds_id)
                    tmp = Template(
                        "\n% Dimension scale: $name\n"
                        "dscl_id = H5D.open(fid, '$name', 'H5P_DEFAULT');\n"
                        "H5DS.attach_scale(dset_id, dscl_id, $idx);\n"
                        "H5D.close(dscl_id);\n"
                    ).substitute({'name': self._dset_path[ds_id],
                                  'idx': index})
                    self._m.append(tmp)

            self._m.append(
                "\nH5D.close(dset_id);\n"
            )

    def get_code(self):
        """Generate MATLAB source code."""
        self._create_file()

        # Order groups by hierarchy...
        groups = self._order_groups()

        for g in groups:
            self._create_group(g)

        if self._dimlist:
            # Handle dimension scales...
            self._dimensions()

        self._close_file()

        return self._m.dump()
