from string import Template
import posixpath as pp
from fast_strconcat import StringStore


class IdlCode(object):
    """Produce IDL source code (as an .m file) to produce a template HDF5
    file."""

    def __init__(self, tinfo, fname, ext='h5'):
        """Initialize an IdlCode instance.

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
        self._c = StringStore()

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

    def _create_file(self):
        """Code to create an HDF5 file."""
        tmplt = Template(
            "; Create the HDF5 file. It will overwrite any file with same "
            "name.\n"
            "fname = '$filename';\n"
            "fid = H5F_CREATE(fname)\n"
        )
        vars = {'filename': self._fname}
        self._c.append(tmplt.substitute(vars))

    def _close_file(self):
        """Code to close an HDF5 file."""
        tmplt = (
            "\n\n"
            "; Close the HDF5 file.\n"
            "H5F_CLOSE, fid\n"
            "; Template file is ready!\n"
        )
        self._c.append(tmplt)

    def _dims2str(self, dims):
        """Stringify dimension list with support for the unlimited size.

        :arg list dims: Dimension size list.
        """
        dim_str = []
        for d in dims:
            if d == 'H5S_UNLIMITED':
                # Unlimited dimension...
                dim_str.append("-1")
            else:
                dim_str.append('{:d}ULL'.format(d))
        return '[{}]'.format(', '.join(dim_str))

    def _dspace(self, shape):
        """Generate dataspace code.

        :arg dict shape: HDF5/JSON shape information.
        """
        if shape['class'] == 'H5S_SCALAR':
            return "sid = H5S_CREATE_SCALAR()\n"
        elif shape['class'] == 'H5S_SIMPLE':
            rank = len(shape['dims'])
            if rank == 1:
                tmplt = Template(
                    "sid = H5S_CREATE_SIMPLE($dims, MAX_DIMENSIONS=$maxdims)\n"
                )
            else:
                tmplt = Template(
                    "sid = H5S_CREATE_SIMPLE(REVERSE($dims), "
                    "MAX_DIMENSIONS=REVERSE($maxdims))\n"
                )
            maxdims = []
            for d in shape['maxdims']:
                if d == 'H5S_UNLIMITED':
                    # Unlimited dimension
                    maxdims.append(-1)
                else:
                    maxdims.append(d)
            vars = {'dims': self._dims2str(shape['dims']),
                    'maxdims': self._dims2str(shape['maxdims'])}
            return tmplt.substitute(vars)
        else:
            raise NotImplementedError('%s: Not supported' % shape['class'])

    def _dtype(self, t, var='tid'):
        """Generate datatype code.

        :arg dict t: HDF5/JSON datatype information.
        :arg str tid: Default name of the datatype variable.
        """
        tcls = t['class']
        if tcls == 'H5T_COMPOUND':
            tmplt = ''

            # Go over each compound field...
            field_cnt = 0
            field_tid_fmt = 'tid%d'
            for f in t['fields']:
                field_cnt += 1
                field_tid = field_tid_fmt % field_cnt
                tmplt += self._dtype(f['type'], var=field_tid)

            # Create the compound datatype...
            array = [field_tid_fmt % i for i in range(1, field_cnt + 1)]
            array = ', '.join(array)
            names = [f['name'].encode('ascii') for f in t['fields']]
            tmplt += "%s = H5T_COMPOUND_CREATE([%s], %s)\n" % (var, array,
                                                               names)

            # Close field datatypes...
            for i in range(1, field_cnt + 1):
                tmplt += "H5T_CLOSE, tid%d\n" % i

            return tmplt

        elif tcls == 'H5T_VLEN':
            return "%s = H5T_VLEN_CREATE(%s)\n" \
                % (var, self._atomic_dtype(t['base']))

        elif tcls == 'H5T_ARRAY':
            tmplt = Template(
                "${base}"
                "$var = H5T_ARRAY_CREATE(base_tid, $dims)\n"
                "H5T_CLOSE, base_tid\n"
            )
            if len(t['dims']) > 1:
                dims = 'REVERSE(%s)' % t['dims']
            else:
                dims = t['dims']
            return tmplt.substitute(
                {'base': self._dtype(t['base'], var='base_tid'),
                 'dims': dims,
                 'var': var})

        else:
            return var + " = " + self._atomic_dtype(t) + "\n"

    def _atomic_dtype(self, t, var='tid'):
        """Handle HDF5 atomic datatypes.

        :arg dict t: HDF5/JSON datatype information.
        """
        tcls = t['class']
        if tcls == 'H5T_STRING':
            if isinstance(t['length'], basestring):
                raise NotImplementedError('Variable length string datatype '
                                          'not supported yet.')
            else:
                tmplt = Template(
                    "H5T_IDL_CREATE(STRING('a', FORMAT='(A${n})'))"
                )
                return tmplt.substitute({'n': t['length']})

        elif tcls in ('H5T_FLOAT', 'H5T_INTEGER'):
            type_map = {'H5T_STD_U8': 'BYTE',
                        'H5T_STD_U16': 'UINT',
                        'H5T_STD_U32': 'ULONG',
                        'H5T_STD_U64': 'ULONG64',
                        'H5T_STD_I16': 'FIX',
                        'H5T_STD_I32': 'LONG',
                        'H5T_STD_I64': 'LONG64',
                        'H5T_IEEE_F32': 'FLOAT',
                        'H5T_IEEE_F64': 'DOUBLE'}
            base = t['base'][:-2]
            try:
                return "H5T_IDL_CREATE(%s(0))" % type_map[base]
            except KeyError:
                raise NotImplementedError('IDL does not support datatype: %s'
                                          % t['base'])

        elif tcls == 'H5T_REFERENCE':
            if t['base'] == 'H5T_STD_REF_OBJ':
                region = ''
            else:
                region = '/REGION'
            return "H5T_REFERENCE_CREATE(%s)" % region

        else:
            raise NotImplementedError('%s: Datatype class not supported yet'
                                      % t['class'])

    def _create_attr(self, attr, locid, dimscale=False):
        """Generate code for one attribute of the ``locid`` object.

        :arg dict attr: Attribute information.
        :arg str locid: Attribute's parent variable name.
        :arg bool dimscale: Parent object type: ``group`` or ``dataset``.
        """
        dataspace = self._dspace(attr['shape'])
        datatype = self._dtype(attr['type'])

        tmplt = Template(
            "\n; Attribute: $name\n"
            "${datatype}"
            "${dataspace}"
            "aid = H5A_CREATE($locid, '$name', tid, sid)\n"
            "${value}"
            "H5T_CLOSE, tid\n"
            "H5S_CLOSE, sid\n"
            "H5A_CLOSE, aid\n"
        )
        if dimscale and attr['name'] in ('REFERENCE_LIST', 'DIMENSION_LIST'):
            val = ("; This is a dimension scale attribute. It's value is "
                   "written later in the code.\n")
        else:
            if attr['type']['class'] == 'H5T_STRING':
                def prep_vals(vals):
                    values = list()
                    for v in vals:
                        temp = v.encode('ascii')
                        temp = temp.replace("'", "''")
                        if '\n' in temp:
                            # Replace "\n" with "STRING(10B)"
                            temp = "'+STRING(10B)+'".join(temp.split('\n'))
                        values.append(temp)
                    return values

                if attr['shape']['class'] == 'H5S_SCALAR':
                    value = "%s" % prep_vals([attr['value']])[0]
                else:
                    if len(attr['shape']['dims']) > 1:
                        raise NotImplementedError(
                            'Rank > 1 for string data not supported')
                    value = prep_vals(attr['value'])

                if attr['type']['length'] == 'H5T_VARIABLE':
                    val_str = \
                        '[' + ', '.join(["'%s'" % v for v in value]) + ']'
                    value = "H5T_VLEN_TO_STR(%s)" % val_str
                else:
                    # Left-justified, fixed-length string format...
                    # fmt = '{{:<{0:d}.{0:d}}}'.format(attr['type']['length'])

                    if attr['shape']['class'] == 'H5S_SCALAR':
                        # value = "'%s'" % fmt.format(value)
                        value = "'%s'" % value
                    else:
                        # for i in xrange(len(value)):
                        #     value[i] = fmt.format(value[i])
                        value = ('['
                                 + ', '.join(["'%s'" % v for v in value])
                                 + "]")
            else:
                value = attr['value']
            val = Template(
                "H5A_WRITE, aid, $value\n"
            ).substitute({'value': value})

        vars = {'locid': locid,
                'name': attr['name'],
                'datatype': datatype,
                'dataspace': dataspace,
                'value': val}
        self._c.append(tmplt.substitute(vars))

    def _is_dimscale(self, attrs):
        """Check if the dataset is a dimension scale.

        :arg list attrs: All dataset's attributes.
        """
        # Check if REFERENCE_LIST attribute is present...
        ref_list = any(a['name'] == 'REFERENCE_LIST'
                       and a['type']['class'] == 'H5T_COMPOUND'
                       for a in attrs)

        # Check if CLASS attribute is present...
        cls = any(a['name'] == 'CLASS'and a['value'] == 'DIMENSION_SCALE'
                  for a in attrs)

        if ref_list and cls:
            return True
        else:
            return False

    def _is_dimlist(self, attrs):
        """Check if the dataset has dimension scales atteched.

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
        :arg bool dimscale: Boolean indicating whether the attributes belong to
            a dimension scale.
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
            "; Dataset: $name\n"
            "${datatype}"
            "${dataspace}"
            "dsid = H5D_CREATE($locid, '$name', tid, sid${layout}${filter})\n"
            "H5S_CLOSE, sid\n"
            "H5T_CLOSE, tid\n"
        )

        # Layout...
        layout = ds.get('creationProperties', {}).get('layout', {})
        lyt = ''
        if layout.get('class', 'H5D_CONTIGUOUS') == 'H5D_CONTIGUOUS':
            pass
        elif layout['class'] == 'H5D_COMPACT':
            pass
        elif layout['class'] == 'H5D_CHUNKED':
            if len(layout['dims']) == 1:
                chunks = '$chunks'
            else:
                chunks = 'REVERSE($chunks)'
            lyt = Template(
                ", CHUNK_DIMENSIONS=" + chunks
            ).substitute({'chunks': self._dims2str(layout['dims'])})
        else:
            raise ValueError('%s: Invalid layout class'
                             % layout['class'])

        # Filters...
        filters = ds.get('creationProperties', {}).get('filters', [])
        fltr = ''
        for f in filters:
            if f['class'] == 'H5Z_FILTER_DEFLATE':
                fltr += ", GZIP=%s" % f['level']
            elif f['class'] == 'H5Z_FILTER_SHUFFLE':
                fltr += ", /SHUFFLE"
            else:
                raise NotImplementedError('%s: Filter not supported yet'
                                          % f['class'])

        vars = {'locid': locid,
                'name': name,
                'datatype': datatype,
                'dataspace': dataspace,
                'layout': lyt,
                'filter': fltr}
        self._c.append(tmplt.substitute(vars))

        attrs = ds.get('attributes', [])
        dimscale = self._is_dimscale_related(attrs)
        if self._is_dimlist(attrs):
            self._dimlist.append(id)
        self._create_attrs(attrs, 'dsid', dimscale=dimscale)

        self._c.append("H5D_CLOSE, dsid\n")

    def _create_dsets(self, dsets, locid, path):
        """Generate code for all the datasets of the ``locid`` object.

        :arg dict dsets: HDF5/JSON information about the datasets of
            the parent ``locid`` object.
        :arg str locid: IDL variable name of the datasets' parent object.
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
                ";\n"
                "; Group: $path\n"
                ";\n"
                "gid = H5G_CREATE(fid, '$path')\n"
            )
            vars = {'path': path}
            self._c.append(tmplt.substitute(vars))
            locid = 'gid'

        self._create_attrs(self._d['groups'][grpid].get('attributes', []),
                           locid)
        self._create_dsets(self._get_hard_links(grpid, 'datasets'), locid,
                           path)

        if grpid != self._root:
            self._c.append("H5G_CLOSE, gid\n")

    def _dimscales(self):
        """Generate code connecting dimension scales and their datasets."""
        # List for storing dimension scales IDs...
        dscales = set()

        self._c.append(
            "\n\n"
            ";\n"
            "; Datasets and their dimension scales\n"
            ";\n"
        )
        for dset_id in self._dimlist:
            dset = self._d['datasets'][dset_id]

            # Get DIMENSION_LIST attribute value...
            dims = next(a['value'] for a in dset['attributes']
                        if a['name'] == 'DIMENSION_LIST')

            tmp = Template(
                "\n; Dataset with dimension scales: $name\n"
                "dset_id = H5D_OPEN(fid, '$name')\n"
                "aid = H5A_OPEN_NAME(dset_id, 'DIMENSION_LIST')\n"
                "dims = REPLICATE({IDL_H5_VLEN}, $n)\n"
            ).substitute({'name': self._dset_path[dset_id],
                          'n': len(dims)})
            self._c.append(tmp)

            # Iterate over dataset's dimension scales...
            for index, dimscales in enumerate(dims):
                self._c.append("ref = INDGEN(%d)\n" % len(dimscales))
                for n, ds in enumerate(dimscales):
                    ds_id = pp.basename(ds)
                    dscales.add(ds_id)
                    self._c.append("ref[%d] = H5R_CREATE(fid, '%s')\n"
                                   % (n, self._dset_path[ds_id]))
                self._c.append("dims[%d].pdata = PTR_NEW(ref);\n" % index)

            self._c.append(
                "H5A_WRITE, aid, dims\n"
                "H5A_CLOSE, aid\n"
                "H5D_CLOSE, dset_id\n"
            )

        for dset_id in dscales:
            tmp = Template(
                "\n; Dimension scale: $name\n"
                "dset_id = H5D_OPEN(fid, '$name')\n"
                "aid = H5A_OPEN_NAME(dset_id, 'REFERENCE_LIST')\n"
            ).substitute({'name': self._dset_path[dset_id]})
            self._c.append(tmp)

            dset = self._d['datasets'][dset_id]

            # Get REFERENCE_LIST attribute value and fields...
            refs, fields = next((a['value'], a['type']['fields'])
                                for a in dset['attributes']
                                if a['name'] == 'REFERENCE_LIST')

            # Field names...
            f_names = [f['name'] for f in fields]

            self._c.append("ref = REPLICATE({%s:1, %s:1}, %d)\n"
                           % tuple(f_names + [len(refs)]))
            for n, r in enumerate(refs):
                ds_id = pp.basename(r[0])
                tmp = Template(
                    "ref[$n].${d} = H5R_CREATE(fid, '$path')\n"
                    "ref[$n].${i} = $index;\n"
                ).substitute({'n': n,
                              'path': self._dset_path[ds_id],
                              'd': f_names[0],
                              'i': f_names[1],
                              'index': r[1]})
                self._c.append(tmp)
            self._c.append(
                "H5A_WRITE, aid, ref\n"
                "H5A_CLOSE, aid\n"
                "H5D_CLOSE, dset_id\n"
            )

    def get_code(self):
        """Generate IDL source code."""
        self._create_file()

        # Order groups by hierarchy...
        groups = self._order_groups()

        for g in groups:
            self._create_group(g)

        if self._dimlist:
            # Handle dimension scales...
            self._dimscales()

        self._close_file()

        return self._c.dump()
